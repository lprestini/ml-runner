import os
import numpy as np
import torch
import cv2

# import sam3
from sam3.model_builder import build_sam3_video_predictor_only
from ml_runner.utils.logs import write_stats_file, calc_progress, check_for_abort_render


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")

print(f"using device: {device}")


class runSAM3(object):
    # TODO move config reading to model class
    def __init__(
        self,
        sam2_checkpoint_path,
        video_dir,
        numpy_img_list,
        render_dir,
        render_name,
        boxes_filt,
        ann_frame_idx,
        first_frame_sequence,
        pred_phrases,
        shot_name,
        text_prompt,
        logger,
        uuid,
        H=None,
        W=None,
        use_gdino=True,
        limit_range=False,
        inference_state=None,
        name_idx=0,
        delimiter=".",
    ):
        self.sam3_checkpoint = sam2_checkpoint_path
        self.predictor = build_sam3_video_predictor_only(
            checkpoint_path=self.sam3_checkpoint
        )
        self.numpy_img_list = numpy_img_list
        self.video_dir = video_dir
        self.render_dir = render_dir
        self.render_name = render_name
        self.boxes_filt = boxes_filt
        self.ann_frame_idx = ann_frame_idx
        self.first_frame_sequence = first_frame_sequence
        self.inference_state = None
        self.pred_phrases = pred_phrases
        self.use_gdino = use_gdino
        self.shot_name = shot_name
        self.logger = logger
        self.uuid = uuid
        self.H = H
        self.W = W
        self.limit_range = limit_range
        self.pred_phrases = (
            self.pred_phrases if self.use_gdino else ["" for i in self.pred_phrases]
        )
        self.inference_state = inference_state
        self.name_idx = name_idx
        self.delimiter = delimiter
        self.is_limit = self.limit_range
        self.text_prompt = text_prompt

        ##Debug paramters
        self.render = True  ## This is for debug only
        self.plot_results = False

    def flip_boxes(self, box, H, W):
        if box[0] > box[2] or box[1] < box[3]:
            box = [
                min(box[0], box[2]),
                min(box[1], box[3]),
                max(box[0], box[2]),
                max(box[1], box[3]),
            ]
        box = [
            box[0] / self.W,
            (H - box[3] - 1) / self.H,
            box[2] / self.W,
            (H - box[3] - 1 + (box[3] - box[1])) / self.H,
        ]
        return box

    def run(self):
        ## If the model returns an image that you don't expect, its probable that the error comes from here
        # I've edited this function so that it takes in shot name as well
        if not self.inference_state:
            self.logger.info("Loading video")
            self.inference_state = self.predictor.init_state(
                resource_path=self.video_dir, numpy_img_list=self.numpy_img_list
            )
            self.logger.info("Video loaded")
        else:
            self.logger.info("Using cached video!")

        shot_name, ext = os.path.splitext(self.shot_name)
        self.render_name = self.render_name if not self.render_name == "" else shot_name

        # If we limited the range, the annotation frame will have a different index
        # This is only used if we are not reading the preloaded frames
        if self.is_limit and len(self.numpy_img_list) == 0:
            self.ann_frame_idx -= self.limit_range[0]

        # Flip boxes if they're coming from Nuke
        if all(i is not None for i in self.boxes_filt):
            for idx, b in enumerate(self.boxes_filt):
                self.boxes_filt[idx] = np.array(
                    [self.flip_boxes(b, self.H, self.W)]
                ).astype("float32")

        # Get total num frames for progress tracking purposes
        total_steps = self.inference_state["num_frames"]
        total_boxes = len(self.boxes_filt)
        filenames = []

        for idx, b in enumerate(self.boxes_filt):
            # Check for abort render file at major break points
            if check_for_abort_render(
                self.render_dir, self.shot_name, self.uuid, self.logger
            ):
                break

            self.predictor.reset_session("0", inference_state=self.inference_state)
            self.logger.info(
                f"Segmenting object ID {idx} out of {len(self.boxes_filt)}"
            )
            ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
            box_labels = [1] if self.boxes_filt[idx] else None
            inference_dict = self.predictor.add_prompt(
                inference_state=self.inference_state,
                frame_idx=self.ann_frame_idx,
                obj_id=ann_obj_id,
                text=self.text_prompt,
                bounding_boxes=self.boxes_filt[idx],
                bounding_box_labels=box_labels,
                session_id=0,
            )

            # Check for abort render file at major break points
            if check_for_abort_render(
                self.render_dir, self.shot_name, self.uuid, self.logger
            ):
                break

            # run propagation throughout the video and collect the results in a dict
            video_segments = {}  # video_segments contains the per-frame segmentation results

            # Cant specify to start from frame 0 so we have to do this in 2 passes if ann frame idx != 0
            # First pass
            self.logger.info("Track mask forward")
            for inference_dict in self.predictor.propagate_in_video(
                inference_state=self.inference_state,
                start_frame_idx=self.ann_frame_idx,
                max_frame_num_to_track=total_steps,
                propagation_direction="forward",
                session_id=0,
            ):
                out_frame_idx = inference_dict["frame_index"]

                out_obj_ids = inference_dict["outputs"]["out_obj_ids"]
                out_mask_logits = inference_dict["outputs"]["out_binary_masks"]

                video_segments[out_frame_idx] = {
                    out_obj_id: out_mask_logits[i]
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

                # Compute progress tracking forward
                if out_frame_idx % 10 == 0:
                    track_progress = calc_progress(
                        total_boxes,
                        idx,
                        (out_frame_idx - self.ann_frame_idx) + 1,
                        total_steps - self.ann_frame_idx,
                    )
                    write_stats_file(
                        self.render_dir,
                        shot_name,
                        self.uuid,
                        "0%",
                        track_progress,
                        False,
                    )

                # If we find file while tracking we break loop but not delete the file
                if check_for_abort_render(
                    self.render_dir,
                    self.shot_name,
                    self.uuid,
                    self.logger,
                    is_tracking=True,
                ):
                    break
            # And we delete the file here
            if check_for_abort_render(
                self.render_dir, self.shot_name, self.uuid, self.logger
            ):
                break

            if self.ann_frame_idx != 0:
                self.logger.info("Track mask backward")
                for inference_dict in self.predictor.propagate_in_video(
                    inference_state=self.inference_state,
                    start_frame_idx=self.ann_frame_idx,
                    max_frame_num_to_track=total_steps,
                    propagation_direction="backward",
                    session_id=0,
                ):
                    out_frame_idx = inference_dict["frame_index"]
                    out_obj_ids = inference_dict["outputs"]["out_obj_ids"]
                    out_mask_logits = inference_dict["outputs"]["out_binary_masks"]

                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                    # Compute progress tracking forward
                    if out_frame_idx % 10 == 0:
                        track_progress = calc_progress(
                            total_boxes,
                            idx,
                            (self.ann_frame_idx - out_frame_idx) + 1,
                            total_steps,
                        )
                        write_stats_file(
                            self.render_dir,
                            shot_name,
                            self.uuid,
                            "0%",
                            track_progress,
                            False,
                        )

                    # And we repeat here
                    if check_for_abort_render(
                        self.render_dir,
                        self.shot_name,
                        self.uuid,
                        self.logger,
                        is_tracking=True,
                    ):
                        break
                # And we delete the file here
                if check_for_abort_render(
                    self.render_dir, self.shot_name, self.uuid, self.logger
                ):
                    break

            if self.render:
                if not os.path.isdir(self.render_dir):
                    os.mkdir(self.render_dir)

            max_detection = max([len(video_segments[i].keys()) for i in video_segments])
            total_boxes += max_detection * total_steps
            # render the segmentation results every few frames
            self.vis_frame_stride = 100
            self.vis_frame_stride = (
                self.vis_frame_stride if not self.render else 1
            )  # interval to check mask set to 0 to render whole sequence
            sam3_det_idx = 0
            for out_frame_idx in range(0, total_steps, self.vis_frame_stride):
                # And we delete the file here
                if check_for_abort_render(
                    self.render_dir, self.shot_name, self.uuid, self.logger
                ):
                    break
                if self.render:
                    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                        # Reshape the mask to match image format
                        # mask_img_format = out_mask.reshape(self.H, self.W,1)
                        mask_img_format = out_mask.astype(int) * 255
                        frame_n = (
                            out_frame_idx + self.first_frame_sequence
                            if not self.is_limit
                            else out_frame_idx
                            + self.limit_range[0]
                            + self.first_frame_sequence
                        )

                        filename = f"{self.render_name}_{self.name_idx}_{out_obj_id}_"
                        name = f"{filename}_{str(frame_n)}.png"

                        cv2.imwrite(
                            os.path.join(self.render_dir, name).replace("\\", "/"),
                            mask_img_format,
                        )

                        # Compute rendering progress
                        if out_frame_idx % 10 == 0 or (
                            out_frame_idx + 1 == total_steps
                        ):
                            filenames.append(filename)
                            filenames = list(set(filenames))
                            render_progress = calc_progress(
                                total_boxes,
                                idx + sam3_det_idx,
                                out_frame_idx + 1,
                                total_steps,
                            )
                            if out_frame_idx + 1 == total_steps:
                                render_progress = "100%"
                            write_stats_file(
                                self.render_dir,
                                filenames,
                                self.uuid,
                                render_progress,
                                "100%",
                                False,
                            )
                        sam3_det_idx += 1
                        #  We check for cancel file but not delete file
                        if check_for_abort_render(
                            self.render_dir,
                            self.shot_name,
                            self.uuid,
                            self.logger,
                            is_tracking=True,
                        ):
                            break
                    # And we delete the file here
                    if check_for_abort_render(
                        self.render_dir, self.shot_name, self.uuid, self.logger
                    ):
                        break

        if len(self.boxes_filt) == 0:
            write_stats_file(self.render_dir, [""], self.uuid, "100%", "100%", False)
            self.logger.info("No bbox found. Please try another word")
        self.predictor.model._exit_context()
        return self.inference_state
