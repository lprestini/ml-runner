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

# ruff: noqa: E402

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from pathlib import Path
import os
import json
import time
import logging
import sys
import torch
import traceback
import argparse
import threading
from simple_website import MLRunnerServer
from datetime import datetime
import queue

from mlrunner_utils.imgutils import (
    get_im_width_height,
    load_imgs_to_numpy,
    get_frame_list,
    load_mov_to_numpy,
)
from mlrunner_utils.logs import write_stats_file


base_path = os.path.join(
    os.path.dirname(os.path.dirname((os.path.abspath(__file__)))), "third_party_models"
).replace("\\", "/")
for i in os.listdir(base_path):
    sys.path.append(os.path.join(base_path, i).replace("\\", "/"))
sys.path.append(base_path.replace("\\", "/"))
sys.path.append(os.path.join(base_path, "edited_sam2/sam2/configs/").replace("\\", "/"))
sys.path.append(os.path.join(base_path, "edited_sam2/sam2/").replace("\\", "/"))
sys.path.append(os.path.join(base_path, "edited_sam3/sam3/").replace("\\", "/"))
sys.path.append(os.path.join(base_path, "edited_dam4sam/dam4sam_2").replace("\\", "/"))
sys.path.append(os.path.join(base_path, "edited_dam4sam/dam4sam_2").replace("\\", "/"))
sys.path.append(os.path.join(base_path, "depth_crafter/").replace("\\", "/"))
sys.path.append(os.path.join(base_path, "rgb2x/").replace("\\", "/"))
sys.path.append(os.path.join(base_path, "co-tracker/").replace("\\", "/"))
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs/damsam2").replace(
        "\\", "/"
    )
)
sys.path.append(os.path.join(base_path, "depth_anything3/").replace("\\", "/"))

# TODO: Evaluate available models at runtime
from model_scripts.run_sam import runSAM2
from model_scripts.run_sam3 import runSAM3
from model_scripts.run_florence import run_florence
from model_scripts.run_dam4sam import runDAM4SAM
from model_scripts.run_gdino import run_gdino
from model_scripts.run_depth_crafter import run_depth_crafter
from model_scripts.run_rgb2x import run_rgb2x
from model_scripts.run_cotracker import run_cotracker
from model_scripts.run_depth_anything3 import run_depth_anything3


PROJECT_ROOT = Path(__file__).parent.parent
CONFIGS_PATH = Path(__file__).parent / "configs"
MODEL_CONFIG_FP = Path(__file__).parent / "model_config.json"

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


class MLRunner(object):
    def __init__(self, listen_dir, use_florence=False):
        self.listen_dir = (
            listen_dir if not listen_dir.endswith("/") else listen_dir[:-1]
        )

        self.job_config = None
        self.model_config_paths = "./model_configs"
        self.ml_logger = logging.getLogger("ML_Runner")
        self.ml_logger.setLevel("INFO")
        self.model_config = self.read_model_config()
        self.use_florence = use_florence
        self.prev_path = None
        self.prev_idx = None
        self.prev_limit = None
        self.prev_extension = None
        self.prev_colourspace = None
        self.prev_sam = None
        self.is_same_path = None
        self.is_same_name = None
        self.is_same_colourspace = None
        self.is_same_extension = None
        self.is_same_sam = None
        self.inference_state = None
        self.name_idx = 0
        self.is_pil = True
        self.loaded_frames = None

        self.path_remap_base_server_path = None
        self.path_remap_replace_path = None

        # Set OCIO config path as env variable cause its easier to pass around
        os.environ["MLRUNNER_OCIOCONFIG_PATH"] = str(
            CONFIGS_PATH / "aces1.2" / "config.ocio"
        )

        self.supported_models = {
            "sam": runSAM2,
            "sam3": runSAM3,
            "florence": run_florence,
            "dam": runDAM4SAM,
            "gdino": run_gdino,
            "depth_crafter": run_depth_crafter,
            "rgb2x": run_rgb2x,
            "cotracker": run_cotracker,
            "depth_anything3": run_depth_anything3,
        }

        self.inform_server_running()

    def inform_server_running(self, closing=False):
        if not closing:
            with open(
                os.path.join(self.listen_dir, ".server_is_running.tmp").replace(
                    "\\", "/"
                ),
                "w",
            ) as f:
                f.write("server is running!")
        else:
            inform_file = os.path.join(
                self.listen_dir, ".server_is_running.tmp"
            ).replace("\\", "/")
            if os.path.isfile(inform_file):
                os.remove(inform_file)

    def fetch_base_and_replace_path(self, config_server_path):
        """Function to detect what parts of config server path needs replacing to ensure smooth operations between OSs."""
        config_list = config_server_path.split("/")
        listen_dir_list = self.listen_dir.split("/")

        # Find what parts of config path are in listen dir
        matched = [i for i in config_list if i in listen_dir_list]

        # Get whatever parts of listen dir arent in config path
        missing = [i for i in listen_dir_list if i not in matched]

        # Get what parts need to be replaced in config path
        replace = [i for i in config_list if i not in listen_dir_list]

        # Create paths
        as_path = os.path.join(*missing)
        replace_path = os.path.join(*replace)

        # Ensure they are correct
        if self.listen_dir.startswith("/") and not as_path.startswith("/"):
            as_path = "/" + as_path
        if config_server_path.startswith("/") and not replace_path.startswith("/"):
            replace_path = "/" + replace_path

        self.path_remap_base_server_path = as_path
        self.path_remap_replace_path = replace_path

    def remap_path_to_server_os(self, path):
        """Function to fix the paths received in server.
        This is needed because users may be sending the config from a different OS, and we need to adjust the paths to be matching on the machine that is running the server"""
        edited_path = path.replace(
            self.path_remap_replace_path, self.path_remap_base_server_path
        )
        return edited_path

    def read_job_config(self, filename):
        if filename.endswith("json"):
            with open(filename, "r") as f:
                try:
                    self.job_config = json.load(f)
                except json.decoder.JSONDecodeError:
                    time.sleep(1)
                    self.job_config = json.load(f)
        return self.job_config

    def read_model_config(self):
        with open(MODEL_CONFIG_FP) as f:
            return json.load(f)

    def resolve_model_config_path(self, path):
        if Path(path).parts[0] != "third_party_models":
            return path
        return str(PROJECT_ROOT / path)

    def sam_definition(
        self,
        video_dir,
        numpy_img_list,
        render_dir,
        render_name,
        boxes_filt,
        ann_frame_idx,
        first_frame_sequence,
        pred_phrases,
        shot_name,
        logger,
        uuid,
        H=None,
        W=None,
        use_gdino=True,
        limit_range=None,
        delimiter=".",
    ):
        sam = self.supported_models["sam"](
            self.resolve_model_config_path(self.model_config["sam"]["checkpoint_path"]),
            self.resolve_model_config_path(self.model_config["sam"]["config_path"]),
            video_dir,
            numpy_img_list,
            render_dir,
            render_name,
            boxes_filt,
            ann_frame_idx,
            first_frame_sequence,
            pred_phrases,
            shot_name,
            logger,
            uuid,
            H,
            W,
            use_gdino,
            limit_range,
            self.inference_state,
            self.name_idx,
            delimiter,
        )
        return sam

    def sam3_definition(
        self,
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
        limit_range=None,
        delimiter=".",
    ):
        sam = self.supported_models["sam3"](
            self.resolve_model_config_path(
                self.model_config["sam3"]["checkpoint_path"]
            ),
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
            H,
            W,
            use_gdino,
            limit_range,
            self.inference_state,
            self.name_idx,
            delimiter,
        )
        return sam

    def dam_definition(
        self,
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
    ):
        dam = self.supported_models["dam"](
            self.resolve_model_config_path(self.model_config["dam"]["checkpoint_path"]),
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
            H,
            W,
            use_gdino,
        )
        return dam

    def florence_definition(
        self,
        image_arr,
        caption,
        box_threshold,
        text_threshold=None,
        with_logits=True,
        H=None,
        W=None,
    ):
        florence = self.supported_models["florence"](
            self.resolve_model_config_path(self.model_config["florence"]["model_path"]),
            image_arr,
            caption,
            box_threshold,
            text_threshold,
            with_logits,
            H,
            W,
        )
        return florence

    def gdino_definition(
        self,
        image_arr,
        caption,
        box_threshold,
        text_threshold=None,
        with_logits=True,
        H=None,
        W=None,
    ):
        gdino_path = [i for i in sys.path if "dino" in i.lower()][0]
        gdino_configs = os.path.join(
            gdino_path, self.model_config["gdino"]["config_path"]
        ).replace("\\", "/")
        gdino_checkpoint = os.path.join(
            gdino_path, self.model_config["gdino"]["checkpoint_path"]
        ).replace("\\", "/")
        gdino = self.supported_models["gdino"](
            image_arr,
            gdino_configs,
            gdino_checkpoint,
            caption,
            box_threshold,
            text_threshold,
            with_logits,
            H,
            W,
        )
        return gdino

    def depth_crafter_definitition(
        self,
        image_arr,
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
        unet_path = self.resolve_model_config_path(
            self.model_config["depth_crafter"]["unet_path"]
        )
        pretrain_path = self.resolve_model_config_path(
            self.model_config["depth_crafter"]["pretrain_path"]
        )
        depth_crafter = self.supported_models["depth_crafter"](
            pretrain_path,
            unet_path,
            image_arr,
            render_dir,
            render_name,
            first_frame_sequence,
            shot_name,
            logger,
            uuid,
            H=H,
            W=W,
            name_idx=name_idx,
            delimiter=delimiter,
        )
        return depth_crafter

    def depth_anything3_definitition(
        self,
        image_arr,
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
        pretrain_path = self.resolve_model_config_path(
            self.model_config["depth_anything3"]["pretrain_path"]
        )
        depth_anything = self.supported_models["depth_anything3"](
            pretrain_path,
            image_arr,
            render_dir,
            render_name,
            first_frame_sequence,
            shot_name,
            logger,
            uuid,
            H=H,
            W=W,
            name_idx=name_idx,
            delimiter=delimiter,
        )
        return depth_anything

    def rgbx2_definitition(
        self,
        image_arr,
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
        pretrain_path = self.resolve_model_config_path(
            self.model_config["rgb2x"]["pretrain_path"]
        )
        rgb2x = self.supported_models["rgb2x"](
            pretrain_path,
            image_arr,
            render_dir,
            render_name,
            first_frame_sequence,
            shot_name,
            logger,
            uuid,
            passes,
            H=H,
            W=W,
            name_idx=name_idx,
            delimiter=delimiter,
        )
        return rgb2x

    def cotracker_definition(
        self,
        first_frame_sequence,
        numpy_img_list,
        render_dir,
        render_name,
        boxes_filt,
        grid_size,
        use_grid,
        ann_frame_idx,
        shot_name,
        logger,
        uuid,
        H=None,
        W=None,
        use_gdino=True,
    ):
        cotracker = self.supported_models["cotracker"](
            self.resolve_model_config_path(
                self.model_config["cotracker"]["checkpoint_path"]
            ),
            numpy_img_list,
            render_dir,
            render_name,
            boxes_filt,
            grid_size,
            use_grid,
            ann_frame_idx,
            first_frame_sequence,
            shot_name,
            logger,
            uuid,
            H,
            W,
            self.name_idx,
        )
        return cotracker

    def clear_sam_memory(self):
        if not self.is_same_path:
            try:
                del self.model
            except AttributeError:
                self.ml_logger.error("tried to release model from memory but failed")
            self.model = None

    def keep_track(
        self, frame_idx, path, limit, render_name, colourspace, ext, model_to_run
    ):
        self.prev_idx = frame_idx
        self.prev_path = path
        self.prev_limit = limit
        self.prev_render_name = render_name
        self.prev_colourspace = colourspace
        self.prev_extension = ext
        self.prev_sam = model_to_run

    def run_model_based_on_cfg(self, filename):
        """Function to run a model based on a config file"""

        # Read config
        config = self.read_job_config(filename)
        model_to_run = config["model_to_run"]
        id_class = config["id_class"]
        text_prompt = config["id_class"] if not "" else None
        bbox = config["crop_position"] if config["crop_position"] else [None]
        path_to_sequence = config["path_to_sequence"]
        shot_name = config["shot_name"]
        ext = os.path.splitext(shot_name)[-1].lower()
        render_to = config["render_to"]
        render_name = config["render_name"]
        frame_idx = int(config["frame_idx"])
        first_frame_sequence = int(config["first_frame_sequence"])
        use_gdino = config["use_gdino"]
        is_mov = config["is_mov"]
        mov_last_frame = config["mov_last_frame"]
        limit_range = config["limit_range"]
        limit_first = config["limit_first"]
        limit_last = config["limit_last"]
        delimiter = config["delimiter"]
        use_grid = config["use_grid"]
        grid_size = config["grid_size"]
        colourspace = config["colourspace"]
        is_srgb = colourspace == "srgb"
        uuid = config["uuid"]
        limit_range = (
            (int(limit_first), int(limit_last)) if limit_range else limit_range
        )
        self.is_pil = False if ".exr" in shot_name.lower() else True
        errors = []
        return_code = 0

        # Ensure paths are correct
        config_server_dir = config["listen_dir"]
        config_server_dir = (
            config_server_dir
            if not config_server_dir.endswith("/")
            else config_server_dir[:-1]
        )
        if config_server_dir != self.listen_dir:
            # Check path differences and sets them as attributes
            self.fetch_base_and_replace_path(config_server_dir)

            # Edit all paths
            path_to_sequence = self.remap_path_to_server_os(path_to_sequence)
            render_to = self.remap_path_to_server_os(render_to)

        # Check if we are processing same footage
        self.is_same_path = path_to_sequence == self.prev_path
        self.is_same_name = (
            frame_idx == self.prev_idx and render_name == self.prev_render_name
        )
        self.is_same_limit = limit_range == self.prev_limit
        self.is_same_colourspace = colourspace == self.prev_colourspace
        self.is_same_extension = ext == self.prev_extension
        self.is_same_sam = model_to_run == self.prev_sam

        # If path or limit is not the same, clear inference state
        if (
            not self.is_same_path
            or not self.is_same_limit
            or not self.is_same_colourspace
            or not self.is_same_extension
        ):
            self.inference_state = None
            self.loaded_frames = None

        if not self.is_same_sam:
            self.inference_state = None

        # If same name, increase index by one to avoid ovewrwriting
        # TODO implement maybe a folder scanning to avoid rewriting with new instances of the runner
        if self.is_same_name:
            self.name_idx += 1
        else:
            self.name_idx = 0

        # Check there are no renders with same name
        if os.path.isdir(render_to):
            # remove duplicate names
            files = list(
                set(
                    [
                        os.path.splitext(os.path.splitext(f)[0])[0]
                        for f in os.listdir(render_to)
                        if render_name in f
                    ]
                )
            )
            if len(files) > 1:
                self.name_idx = len(files) + 1

        # Set prev frame idx and prev path, so that we can keep track of what we are doing
        self.keep_track(
            frame_idx,
            path_to_sequence,
            limit_range,
            render_name,
            colourspace,
            ext,
            model_to_run,
        )

        H, W = None, None
        self.ml_logger.info(f"Shot {shot_name} received!")

        ## Execute model
        try:
            if os.path.isdir(path_to_sequence):
                # If we're not using movs we fetch all the image frames as list and limit those
                if not is_mov:
                    self.frame_names = get_frame_list(
                        path_to_sequence, shot_name, delimiter
                    )
                    self.ml_logger.info("Frames list fetched!")
                else:
                    self.frame_names = list(range(mov_last_frame))
            else:
                raise ValueError(f"The path {path_to_sequence} doesnt seem to exists!")

            # If we limit the range we need to change frame index
            if limit_range:
                self.frame_names = self.frame_names[limit_range[0] : limit_range[1]]
                frame_idx -= limit_range[0]
            try:
                if not self.loaded_frames and not is_mov:
                    self.loaded_frames = load_imgs_to_numpy(
                        self.frame_names, to_srgb=is_srgb
                    )
                    self.ml_logger.info("Frames loaded!")
                elif not self.loaded_frames and is_mov:
                    mov_first_frame = 0
                    if limit_range:
                        mov_first_frame = limit_range[0]
                        mov_last_frame = limit_range[1]
                    self.loaded_frames, self.frame_names = load_mov_to_numpy(
                        path_to_sequence, shot_name, mov_last_frame, mov_first_frame
                    )
                    if not all(i for i in self.frame_names):
                        failed_frames = [
                            (i, idx) for idx, i in enumerate(self.frame_names) if not i
                        ]
                        frames_messages = "\n" + "\n".join(
                            str(i[1]) for i in failed_frames
                        )
                        raise ValueError(
                            f"There was an error loading {len(failed_frames)} frames. These are the failed frames:\n{frames_messages}"
                        )
                    else:
                        self.ml_logger.info("Mov loaded!")
                else:
                    self.ml_logger.info("Using cached video!")
            except Exception:
                raise ValueError(
                    "Something went wrong while loading the images to numpy."
                )

            self.ml_logger.info("Fetching image info..")
            try:
                image_arr = self.loaded_frames[frame_idx]
                H, W = get_im_width_height(image_arr)
                self.ml_logger.info("Fetched image info")
            except IndexError:
                raise IndexError(
                    f"Failed to fetch image information. Here is the frame list info: list lenght: {len(self.frame_names)} frame index: {frame_idx}"
                )

            # Get bboxes with GDINO or Florence if we want to
            if use_gdino or self.use_florence:
                box_threshold = 0.35
                text_threshold = 0.25

                if self.use_florence:
                    florence = self.florence_definition(
                        image_arr, id_class, box_threshold, text_threshold, H, W
                    )
                    bbox, pred_phrases = florence.run()
                elif use_gdino:
                    gdino = self.gdino_definition(
                        image_arr, id_class, box_threshold, text_threshold, H, W
                    )
                    bbox, pred_phrases = gdino.run()

                self.ml_logger.info("BBox fetched!")
            else:
                # If we're not using a model like gdino/florence we need to ensure that pred phrases is set to a list == to the bbox indices
                if not text_prompt:
                    pred_phrases = range(len(bbox))
                else:
                    pred_phrases = [0]

            # Run the requested model
            self.ml_logger.info(f"Running model {model_to_run} on shot {shot_name}")
            if model_to_run == "sam":
                self.model = self.sam_definition(
                    path_to_sequence,
                    self.loaded_frames,
                    render_to,
                    render_name,
                    bbox,
                    frame_idx,
                    first_frame_sequence,
                    pred_phrases,
                    shot_name,
                    self.ml_logger,
                    uuid,
                    H,
                    W,
                    use_gdino,
                    limit_range,
                    delimiter,
                )
                self.inference_state = self.model.run()

            elif model_to_run == "sam3":
                self.model = self.sam3_definition(
                    path_to_sequence,
                    self.loaded_frames,
                    render_to,
                    render_name,
                    bbox,
                    frame_idx,
                    first_frame_sequence,
                    pred_phrases,
                    shot_name,
                    text_prompt,
                    self.ml_logger,
                    uuid,
                    H,
                    W,
                    use_gdino,
                    limit_range,
                    delimiter,
                )
                self.inference_state = self.model.run()

            elif model_to_run == "dam":
                self.model = self.dam_definition(
                    first_frame_sequence,
                    self.loaded_frames,
                    render_to,
                    render_name,
                    bbox,
                    frame_idx,
                    pred_phrases,
                    shot_name,
                    self.ml_logger,
                    uuid,
                    H,
                    W,
                    use_gdino,
                )
                self.model.run()

            elif model_to_run == "depth_crafter":
                # SAM and DAM manage the limit_range interanlly
                if limit_range:
                    first_frame_sequence += limit_range[0]
                self.model = self.depth_crafter_definitition(
                    self.loaded_frames,
                    render_to,
                    render_name,
                    first_frame_sequence,
                    shot_name,
                    self.ml_logger,
                    uuid,
                    H,
                    W,
                    self.name_idx,
                    delimiter,
                )
                self.model.run()

            elif model_to_run == "rgb2x":
                # SAM and DAM manage the limit_range interanlly
                if limit_range:
                    first_frame_sequence += limit_range[0]
                self.model = self.rgbx2_definitition(
                    self.loaded_frames,
                    render_to,
                    render_name,
                    first_frame_sequence,
                    shot_name,
                    self.ml_logger,
                    uuid,
                    config["passes"],
                    H,
                    W,
                    self.name_idx,
                    delimiter,
                )
                self.model.run()

            elif model_to_run == "cotracker":
                # SAM and DAM manage the limit_range interanlly
                if limit_range:
                    first_frame_sequence += limit_range[0]
                self.model = self.cotracker_definition(
                    first_frame_sequence,
                    self.loaded_frames,
                    render_to,
                    render_name,
                    bbox,
                    grid_size,
                    use_grid,
                    frame_idx,
                    shot_name,
                    self.ml_logger,
                    uuid,
                    H,
                    W,
                    use_gdino,
                )
                self.model.run()

            elif model_to_run == "depth_anything3":
                # SAM and DAM manage the limit_range interanlly
                if limit_range:
                    first_frame_sequence += limit_range[0]
                self.model = self.depth_anything3_definitition(
                    self.loaded_frames,
                    render_to,
                    render_name,
                    first_frame_sequence,
                    shot_name,
                    self.ml_logger,
                    uuid,
                    H,
                    W,
                    self.name_idx,
                    delimiter,
                )
                self.model.run()

            self.ml_logger.info(f"shot {shot_name} saved at {render_to}")

        except (IndexError, ValueError, TypeError, KeyError, Exception) as e:
            return_code = 1
            error = str(e)
            if isinstance(e, KeyError):
                self.ml_logger.error(
                    "Florence model seems to have failed. try a different frame or pass a boundibox yourself"
                )
            if "cuda out of memory" in str(e).lower():
                self.ml_logger.error("Looks like you run out of memory!")
                error = "Out of memory"
            write_stats_file(
                render_to, [render_name], uuid, "0%", "0%", True, error_msg=error
            )
            self.ml_logger.error(
                f"An error was caught processing the config, here is the printout \n{e}"
            )
            traceback_error = traceback.format_exc()
            self.ml_logger.error(traceback_error)
            errors.append(traceback_error)
        try:
            del self.model
            del bbox
            del pred_phrases
            if self.use_florence:
                del florence
            elif use_gdino:
                del gdino
            torch.cuda.empty_cache()
        except Exception:
            self.ml_logger.error("tried to release some memory but failed")
            self.ml_logger.error(f"ERROR:{traceback.format_exc()}")

        return return_code, errors


class MLRunnerHandler(FileSystemEventHandler):
    def __init__(self, runner):
        super().__init__()
        # Threading queue stuff
        # We use this so that we can display real queue
        self.lock = threading.Lock()
        self.runner_started = False

        self.mlrunner = runner
        self.listen_dir = self.mlrunner.listen_dir
        self.queue_file = os.path.join(self.listen_dir, "queue.json")
        self.queue = queue.Queue()
        self.pending = set()
        self.timestamp = []
        self.processing_timestamp = []
        self.job_id = []
        self.status = []
        self.error_log = []
        self.model_2_run = []
        self.cached_queue = None
        self.initialize_cache()

        # Load existing cache if it exists
        self.load_queue()

    def start_runner(self):
        """Start runner thread"""

        if self.runner_started:
            return None
        self.runner_started = True
        thread = threading.Thread(target=self.runner_loop, daemon=True)
        thread.start()

    def initialize_cache(self):
        """Create empty queue file if it doesnt exist"""
        if not os.path.isfile(self.queue_file):
            self.write_queue()

    def runner_loop(self):
        """Fetch item from queue and run it - then write to cache"""
        while True:
            current = self.queue.get()
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            job_id_idx = len(self.job_id) - 1
            self.status[job_id_idx] = "In progress"
            self.processing_timestamp[job_id_idx] = timestamp
            self.write_queue()
            status, errors = self.mlrunner.run_model_based_on_cfg(current)

            if status == 0:
                self.status[job_id_idx] = "Completed"
            else:
                self.status[job_id_idx] = "Error"
                self.error_log[job_id_idx] = "\n".join(i for i in errors)

            self.write_queue()
            with self.lock:
                self.pending.discard(current)

            self.queue.task_done()

    def get_model_and_shot_name(self, path_to_json):
        config = self.mlrunner.read_job_config(path_to_json)
        return config["model_to_run"], config["shot_name"]

    def add_queue(self, path):
        """Add configs to the queue and write cache to file to update web server"""
        # Create timestamp and get model and shot name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_to_run, shot_name = self.get_model_and_shot_name(path)
        shot_name, _ = os.path.splitext(shot_name)

        # Add to queue
        self.queue.put(path)
        self.status.append("In queue..")
        self.timestamp.append(timestamp)
        self.processing_timestamp.append(None)
        self.error_log.append(None)
        self.model_2_run.append(model_to_run)
        self.job_id.append(shot_name)

        # Write cache
        self.write_queue()
        self.mlrunner.ml_logger.info(f"{os.path.basename(path)} added to queue")
        self.mlrunner.ml_logger.info(
            f"Total number of jobs in queue is: {self.queue.qsize()}"
        )

    def on_created(self, event: FileSystemEvent) -> None:
        """This functionts takes care of the events.
        If file is directory or not a json returns so we dont run the rest
        if its in the pending set, return so we dont run the rest
        otherwise add to queue and start runner
        Note runner will start only if its not already started

        Using this system allow the UI to display proper queue as the runner happens on a different thread

        """
        if event.is_directory:
            return
        if not event.src_path.endswith(".json"):
            return

        with self.lock:
            if event.src_path in self.pending:
                return
            self.pending.add(event.src_path)
        self.add_queue(event.src_path)
        self.start_runner()

    def write_queue(self):
        """Write data to file"""
        data = {
            "queue": list(self.queue.queue),
            "job_id": self.job_id,
            "status": self.status,
            "timestamp": self.timestamp,
            "started_at": self.processing_timestamp,
            "error_log": self.error_log,
            "model2run": self.model_2_run,
        }
        with self.lock:
            with open(self.queue_file, "w") as f:
                json.dump(data, f, indent=4)

    def load_queue(self):
        if os.path.isfile(self.queue_file):
            self.mlrunner.ml_logger.info("Found existing queue cache, loading it now..")
            with open(self.queue_file, "r") as f:
                self.cached_queue = json.load(f)

        # Load cached queue
        if self.cached_queue:
            for i in self.cached_queue["queue"]:
                self.queue.put(i)
                self.pending.add(i)
            self.job_id = self.cached_queue["job_id"]
            self.status = self.cached_queue["status"]
            self.timestamp = self.cached_queue["timestamp"]
            self.processing_timestamp = self.cached_queue["started_at"]
            self.model_2_run = self.cached_queue["model2run"]
            self.error_log = self.cached_queue["error_log"]
            if self.queue.qsize() > 0:
                self.start_runner()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s][%(levelname)s]: %(message)s", level=logging.INFO
    )
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(prog="MLRunner")
    parser.add_argument(
        "-l",
        "--listen_dir",
        required=True,
        help="Directory to listen for config delivery on.",
    )
    parser.add_argument(
        "-f",
        "--use_florence",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to use florence or grounding dino",
    )
    parser.add_argument(
        "-wb",
        "--web_server",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to use web server or not",
    )
    parser.add_argument(
        "--ip_allow_list",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to use an IP allow least to reach the webserver",
    )
    parser.add_argument(
        "-p", "--port", default=8001, help="What port to use for the web server"
    )
    args = parser.parse_args()
    listen_path = args.listen_dir
    use_florence = args.use_florence
    use_web_server = args.web_server
    use_allow_list = args.ip_allow_list
    port = args.port

    assert os.path.isdir(listen_path), (
        "Hey the path you passed doesnt exists. Pleas insert a valid path"
    )
    if use_web_server:
        allow_list = []  # EDIT ME TO ADD SUBNETS YOU WANT TO ALLOW -E.G ['100.123.123.0/24',]
        if use_allow_list and len(allow_list) == 0:
            raise AssertionError("Please set up the IP allow list to continue")
        web_server = MLRunnerServer(
            data_path=os.path.join(listen_path, "queue.json"),
            web_dir=os.path.dirname(os.path.abspath(__file__)),
            ip_allow_list=allow_list,
            use_ip_allow_list=use_allow_list,
        )
        web_server.start_threaded(host="0.0.0.0", port=port)

    runner = MLRunner(listen_path, use_florence=use_florence)

    # Gotta use poll as the standard observer doesnt work on nfs
    is_poll = not listen_path.startswith("/mnt/")
    is_poll = not listen_path.startswith("/Volumes/") if is_poll else is_poll

    observer = Observer() if is_poll else PollingObserver(timeout=1.0)
    event_handler = MLRunnerHandler(runner)
    observer.schedule(event_handler, runner.listen_dir)
    runner.ml_logger.info(f"Starting listening at directory {runner.listen_dir}")

    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        runner.ml_logger.error("User interrupted the server, shutting down")
        runner.inform_server_running(closing=True)
        try:
            sys.exit(130)
        except SystemExit:
            os._exit(130)
    finally:
        runner.ml_logger.info("Shutting down server")
        runner.inform_server_running(closing=True)
        observer.stop()
        sys.exit()
