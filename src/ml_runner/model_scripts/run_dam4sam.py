######################################################################
# Copyright (c) 2025 Luca Prestini
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
######################################################################

import os
import numpy as np
import torch
import cv2
from tqdm import tqdm

from edited_dam4sam.dam4sam_tracker import DAM4SAMTracker
from ml_runner.utils.logs import write_stats_file, calc_progress, check_for_abort_render


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")

print(f"using device: {device}")
# print(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'third_party_models', 'edited_dam4sam'))


class runDAM4SAM(object):
    def __init__(
        self,
        checkpoint_path,
        first_frame_sequence,
        numpy_img_list,
        render_dir,
        render_name,
        boxes_filt,
        ann_frame_idx,
        pred_phrases,
        shot_name,
        logger,
        uuid,
        H=None,
        W=None,
        use_gdino=True,
        limit_range=None,
    ):

        self.predictor = DAM4SAMTracker("sam21pp-L", ch=checkpoint_path)
        self.first_frame_sequence = first_frame_sequence
        self.numpy_img_list = numpy_img_list
        self.render_dir = render_dir
        self.render_name = render_name
        self.boxes_filt = boxes_filt
        self.ann_frame_idx = ann_frame_idx
        self.inference_state = None
        self.pred_phrases = pred_phrases
        self.use_gdino = use_gdino
        self.shot_name = shot_name
        self.logger = logger
        self.uuid = uuid
        self.H = H
        self.W = W
        self.pred_phrases = (
            self.pred_phrases if self.use_gdino else ["" for i in self.pred_phrases]
        )
        self.track_progress = 0
        self.is_pil = not shot_name.endswith(".exr")
        self.is_abort = False
        self.filenames = []

        ##Debug paramters
        self.render = True  ## This is for debug only
        self.plot_results = False

    def prep_box_for_dam(self, box, H, W, from_nuke=False):
        """When using DAM4SAM the order of the bbox is as follow:
        left,top,abs(left-right),abs(top-bottom)"""
        if from_nuke:
            box = [
                min(box[0], box[2]),
                min(box[1], box[3]),
                max(box[0], box[2]),
                max(box[1], box[3]),
            ]
            # Flip boxes
            box = [
                box[0],
                H - box[3] - 1,
                box[2],
                H - box[3] - 1 + (box[3] - box[1]),
            ]

        box = [box[0], box[1], abs(box[0] - box[2]), abs(box[1] - box[-1])]
        return box

    def track_mask_and_render(
        self,
        sorted_frames,
        ann_fram_idx,
        idx,
        direction="forward",
        prev_track_progress=0,
    ):

        is_forward = direction == "forward"
        frames = (
            sorted_frames[ann_fram_idx:] if is_forward else sorted_frames[:ann_fram_idx]
        )
        indexed = (
            enumerate(frames, start=ann_fram_idx)
            if is_forward
            else reversed(list(enumerate(frames, start=0)))
        )
        total_steps = len(sorted_frames)
        total_boxes = len(self.boxes_filt)

        for iidx, img in tqdm(indexed, total=len(frames), desc=f"Render {direction}"):
            # Check for interrupt file without deleting file
            if check_for_abort_render(
                self.render_dir,
                self.shot_name,
                self.uuid,
                self.logger,
                is_tracking=True,
            ):
                break
            if iidx - ann_fram_idx == 0:
                output = self.predictor.initialize(
                    img, None, bbox=np.array(self.boxes_filt[idx]).astype(np.float32)
                )
            else:
                output = self.predictor.track(img)

            # Check again
            if check_for_abort_render(
                self.render_dir,
                self.shot_name,
                self.uuid,
                self.logger,
                is_tracking=True,
            ):
                break

            pred_mask = output["pred_mask"]
            frame_n = iidx + self.first_frame_sequence

            name_no_frame = f"{self.render_name}_{self.pred_phrases[idx]}_{idx}"
            name = f"{name_no_frame}_{frame_n}.png"
            cv2.imwrite(
                os.path.join(self.render_dir, name).replace("\\", "/"), pred_mask * 255
            )

            # Compute progress tracking forward
            # TODO this needs testing
            step = (
                ((iidx + 1) - ann_fram_idx) + prev_track_progress
                if direction == "forward"
                else ((ann_fram_idx - (iidx + 1)) + 1) + prev_track_progress
            )

            if iidx % 10 == 0:
                self.filenames.append(name_no_frame)
                self.filenames = list(set(self.filenames))
                track_progress = calc_progress(total_boxes, idx, step, total_steps)
                write_stats_file(
                    self.render_dir,
                    self.filenames,
                    self.uuid,
                    track_progress,
                    track_progress,
                    False,
                )

        # And here we delete
        if not check_for_abort_render(
            self.render_dir, self.shot_name, self.uuid, self.logger
        ):
            self.filenames.append(name_no_frame)
            self.filenames = list(set(self.filenames))
            step = (
                ((iidx + 1) - ann_fram_idx) + prev_track_progress
                if direction == "forward"
                else ((ann_fram_idx - (iidx + 1)) + 1) + prev_track_progress
            )
            track_progress = calc_progress(total_boxes, idx, step, total_steps)
            write_stats_file(
                self.render_dir,
                self.filenames,
                self.uuid,
                track_progress,
                track_progress,
                False,
            )
            self.track_progress = step
        else:
            self.is_abort = True

    def run(self):
        self.render_name = (
            self.render_name if not self.render_name == "" else self.shot_name
        )
        sorted_frames = self.numpy_img_list

        # Prep boxes
        for idx, b in enumerate(self.boxes_filt):
            self.boxes_filt[idx] = self.prep_box_for_dam(
                b, self.H, self.W, from_nuke=not self.use_gdino
            )

        for idx, b in enumerate(self.boxes_filt):
            self.logger.info(
                f"Segmenting object ID {idx} out of {len(self.boxes_filt)}"
            )
            # Cant specify to start from frame 0 so we have to do this in 2 passes if ann frame idx != 0
            # First pass
            self.logger.info("Track mask forward")

            self.track_mask_and_render(
                sorted_frames,
                self.ann_frame_idx,
                idx,
            )

            if self.ann_frame_idx != 0 and not self.is_abort:
                self.logger.info("Track mask backward")
                self.track_mask_and_render(
                    sorted_frames,
                    self.ann_frame_idx,
                    idx,
                    direction="backward",
                    prev_track_progress=self.track_progress,
                )
