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
from PIL import Image
import cv2
import sys
import torch.nn.functional as F

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")

print(f"using device: {device}")
from sharp.models import (
    PredictorParams,
    create_predictor,
)
from sharp.utils.gaussians import (
    save_ply,
    unproject_gaussians,
)

from mlrunner_utils.logs import write_stats_file, calc_progress, check_for_abort_render

class runMLSharp(object):
    def __init__(self, 
                 mlsharp_ckpt_path,
                 numpy_img_list,
                 focal_lenght, 
                 film_back_size, 
                 render_dir,
                 render_name,
                 ann_frame_idx,
                 first_frame_sequence,
                 shot_name, 
                 logger,
                 uuid,
                 H = None,
                 W = None,
                 limit_range = False,
                 name_idx = 0,
                 delimiter = "."):

        self.gaussian_predictor = create_predictor(PredictorParams())
        self.gaussian_predictor.load_state_dict(torch.load(mlsharp_ckpt_path, weights_only=True))
        self.gaussian_predictor.eval()
        self.gaussian_predictor.to(device)

        self.numpy_img_list = numpy_img_list
        self.render_dir = render_dir
        self.focal_lenght = focal_lenght
        self.film_back_w, self.film_back_h = film_back_size
        self.render_name = render_name
        self.ann_frame_idx = ann_frame_idx
        self.first_frame_sequence = first_frame_sequence
        self.shot_name = shot_name
        self.logger = logger
        self.uuid = uuid
        self.H = H 
        self.W = W
        self.limit_range = limit_range
        self.name_idx = name_idx
        self.delimiter = delimiter
        self.is_limit = self.limit_range != False

        ##Debug paramters
        self.render = True ## This is for debug only
        self.plot_results = False 
    
    def predit_gs(self, image_tensor, focal_lenght_px, device):
        """ images are expected as 0-1 range torch arrays"""
        # TODO - figure out if internal shape should be exposed
        internal_shape = (1536, 1536)
        _, height, width = image_tensor.shape
        disparity_factor = torch.tensor([focal_lenght_px / width]).float().to(device)

        image_resized_pt = F.interpolate(
            image_tensor[None],
            size=(internal_shape[1], internal_shape[0]),
            mode="bilinear",
            align_corners=True,
        )
        gaussians_ndc = self.gaussian_predictor(image_resized_pt, disparity_factor)
        intrinsics = (
            torch.tensor(
                [
                    [focal_lenght_px, 0, width / 2, 0],
                    [0, focal_lenght_px, height / 2, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            .float()
            .to(device)
        )
        intrinsics_resized = intrinsics.clone()
        intrinsics_resized[0] *= internal_shape[0] / width
        intrinsics_resized[1] *= internal_shape[1] / height

        # Convert Gaussians to metrics space.
        gaussians = unproject_gaussians(
            gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
        )

        return gaussians
    
    def focal_lenght_to_fpx(self, im_width, im_height, f = 30, film_back_w = 36, film_back_h = 24):
        """Go from mm to px
        args: 
        im_widht = iamge width
        im_height = image_height
        f = focal lenght in millimiters
        film_back_w = size of sensor width
        film_back_h = size of sensor height
        returns float of focal lenght in pixels
        """
        return f * np.sqrt((im_width ** 2.0) + (im_height ** 2.0)) / np.sqrt((film_back_w ** 2) + (film_back_h ** 2))
        
    def run(self):
        shot_name, ext = os.path.splitext(self.shot_name)
        self.render_name = self.render_name if not self.render_name == "" else shot_name
        filenames = []

        # If we limited the range, the annotation frame will have a different index
        # This is only used if we are not reading the preloaded frames
        if self.is_limit and len(self.numpy_img_list) == 0:
            self.ann_frame_idx -= self.limit_range[0]

        B = len(self.numpy_img_list)
        
        if not check_for_abort_render(self.render_dir, self.shot_name, self.uuid, self.logger):
            self.logger.info(f"Starting to convert GS")
            for out_frame_idx, image in enumerate(self.numpy_img_list):
                # Check for abort render file at major break points
                if check_for_abort_render(self.render_dir, self.shot_name, self.uuid, self.logger, is_tracking = True):
                    break

                width , height = (self.W, self.H)
                image = torch.from_numpy(image).permute(-1,0,1).float().to(device)
                f_px = self.focal_lenght_to_fpx(width, height, f = self.focal_lenght, film_back_w = self.film_back_w, film_back_h = self.film_back_h)
                gaussians = self.predit_gs(image, f_px, device)

                if self.render:
                    if not os.path.isdir(self.render_dir):
                        os.mkdir(self.render_dir)

                # Compute progress tracking forward
                out_frame_idx = out_frame_idx + self.first_frame_sequence if not self.is_limit else out_frame_idx + self.limit_range[0] + self.first_frame_sequence
                filename = f"{self.render_name}_{self.name_idx}"
                name = f"{filename}_{str(out_frame_idx)}.ply"
                save_ply(gaussians, f_px, (height, width), os.path.join(self.render_dir, name))

                if out_frame_idx % 10 == 0:
                    filenames.append(filename)
                    filenames = list(set(filenames))
                    track_progress = calc_progress(1, 0, (out_frame_idx - self.ann_frame_idx) + 1, B - self.ann_frame_idx)
                    write_stats_file(self.render_dir, filenames, self.uuid, track_progress, "100%", False)

            if not check_for_abort_render(self.render_dir, self.shot_name, self.uuid, self.logger):
                if len(filenames) == 0:
                    filenames.append(filename)
                write_stats_file(self.render_dir, filenames, self.uuid, "100%", "100%", False)
                self.logger.info("All done!")