import os
import torch
import cv2
from mlrunner_utils.logs import write_stats_file, calc_progress, check_for_abort_render
from rgb2x.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
from diffusers import DDIMScheduler
import torchvision.transforms.functional as tvf
import torch.nn.functional as F
from tqdm import tqdm


class run_rgb2x(object):
    def __init__(
        self,
        pretrain_path,
        numpy_img_list,
        render_dir,
        render_name,
        first_frame_sequence,
        shot_name,
        logger,
        uuid,
        passes,
        H=None,
        W=None,
        name_idx=0,
        delimiter=".",
    ):

        self.pretrain_path = pretrain_path
        self.numpy_img_list = numpy_img_list
        self.render_dir = render_dir
        self.render_name = render_name
        self.first_frame_sequence = first_frame_sequence
        self.shot_name = shot_name
        self.logger = logger
        self.uuid = uuid
        self.passes = passes
        self.H = H
        self.W = W
        self.name_idx = name_idx
        self.delimiter = delimiter

        self.pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(
            self.pretrain_path, torch_dtype=torch.float16
        ).to("cuda")
        # cache_dir=os.path.join(current_directory, "model_cache"), )

        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config,
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to("cuda")

    def convert_to_tensor_and_ensure_legal_size(self, array, max_res=1024):
        """Ensure image is multiple of 8 and resize it to max_res"""
        self.pad_H = round(self.H / 64) * 64
        self.pad_W = round(self.W / 64) * 64
        tensor = torch.cat(
            [
                torch.unsqueeze(torch.from_numpy(i).permute(2, 1, 0), dim=0)
                for i in array
            ],
            dim=0,
        )
        tensor = tvf.pad(
            tensor,
            (0, 0, self.pad_W - self.W, self.pad_H - self.H),
            padding_mode="reflect",
        )
        self.downsize_to_max_size_ratio = max_res / max(self.pad_W, self.pad_H)
        height = round(self.pad_H * self.downsize_to_max_size_ratio / 64) * 64
        width = round(self.pad_W * self.downsize_to_max_size_ratio / 64) * 64
        tensor = F.interpolate(
            tensor, size=(int(width), int(height)), mode="bilinear", align_corners=True
        ).permute(0, 1, 3, 2)
        return tensor

    def restore_og_size(self, array):
        return cv2.resize(array, (self.W, self.H), interpolation=cv2.INTER_CUBIC)

    def run(self):
        img_tensor = self.convert_to_tensor_and_ensure_legal_size(self.numpy_img_list)
        generator = torch.Generator(device="cuda").manual_seed(30)

        prompts = {
            "albedo": "Albedo (diffuse basecolor)",
            "normal": "Camera-space Normal",
            "roughness": "Roughness",
            "metallic": "Metallicness",
            "irradiance": "Irradiance (diffuse lighting)",
        }

        inference_step = 50
        B, C, height, width = img_tensor.shape
        for idx, frame in enumerate(img_tensor):
            # Check for abort render file and delete file
            if check_for_abort_render(
                self.render_dir, self.shot_name, self.uuid, self.logger
            ):
                break
            filenames = []

            for p, aov_name in tqdm(
                enumerate(self.passes),
                total=(len(self.passes) * B),
                desc=f"Computing passes for frame {idx + self.first_frame_sequence}",
            ):
                prompt = prompts[aov_name]
                generated_image = self.pipe(
                    prompt=prompt,
                    photo=frame,
                    num_inference_steps=inference_step,
                    height=height,
                    width=width,
                    generator=generator,
                    required_aovs=[aov_name],
                    output_type="np",
                ).images[0][0]

                # Check for abort but dont delete file
                if check_for_abort_render(
                    self.render_dir,
                    self.shot_name,
                    self.uuid,
                    self.logger,
                    is_tracking=True,
                ):
                    break

                filename = f"{self.render_name}_{self.name_idx}_{aov_name}"
                frame_n = idx + self.first_frame_sequence
                name = f"{filename}_{str(frame_n)}.exr"
                generated_image = self.restore_og_size(generated_image)
                cv2.imwrite(
                    os.path.join(self.render_dir, name).replace("\\", "/"),
                    generated_image,
                )
                filenames.append(filename)

            if idx % 10:
                render_progress = calc_progress(1, 0, idx + 1, B)
                write_stats_file(
                    self.render_dir,
                    filenames,
                    self.uuid,
                    render_progress,
                    "100%",
                    False,
                )
            
            # Check for abort render file and delete file
            if check_for_abort_render(
                self.render_dir, self.shot_name, self.uuid, self.logger
            ):
                break

        # Render is finished - lets set progress to 100%
        if idx +1 == B:
            write_stats_file(
                self.render_dir,
                filenames,
                self.uuid,
                "100%",
                "100%",
                False,
            )