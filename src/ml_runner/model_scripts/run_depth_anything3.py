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

from depth_anything_3.api import DepthAnything3
from depth_anything_3.dp_utils.visualize import visualize_depth
import cv2
from ml_runner.utils.logs import write_stats_file, calc_progress, check_for_abort_render


class run_depth_anything3(object):
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
        self.H = H
        self.W = W
        self.name_idx = name_idx
        self.delimiter = delimiter

        self.model = DepthAnything3.from_pretrained(pretrain_path).to("cuda")

    def restore_og_size(self, array):
        return cv2.resize(array, (self.W, self.H), interpolation=cv2.INTER_CUBIC)

    def run(self):

        # Stack frames
        ## TODO put assert for img having different H-W betweenf rames
        stop = False
        total_steps = len(self.numpy_img_list)

        if check_for_abort_render(
            self.render_dir, self.shot_name, self.uuid, self.logger
        ):
            stop = True
        if not stop:
            result = self.model.inference(
                image=self.numpy_img_list, infer_gs=False
            ).depth

            for idx, frame in enumerate(result):
                if check_for_abort_render(
                    self.render_dir, self.shot_name, self.uuid, self.logger
                ):
                    break
                frame = visualize_depth(frame)
                filename = f"{self.render_name}_{self.name_idx}"
                frame_n = idx + self.first_frame_sequence
                name = f"{filename}_{str(frame_n)}.exr"
                frame = self.restore_og_size(frame).astype(np.float32)
                cv2.imwrite(
                    os.path.join(self.render_dir, name).replace("\\", "/"),
                    frame / 255.0,
                )
                if idx % 10:
                    render_progress = calc_progress(1, 0, idx + 1, total_steps)
                    write_stats_file(
                        self.render_dir,
                        [filename],
                        self.uuid,
                        render_progress,
                        "100%",
                        False,
                    )
