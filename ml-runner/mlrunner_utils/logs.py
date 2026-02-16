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
import json
import math


def write_stats_file(
    path, filename, name, render_p, track_p, error, cancelled=False, error_msg=""
):
    progress = {
        "name": name,
        "tracking_progress": track_p,
        "render_progress": render_p,
        "filename": filename,
        "is_cancelled": cancelled,
        "error": error,
        "error_msg": error_msg,
    }
    with open(
        os.path.join(path, f"{name}_render_progress.json").replace("\\", "/"), "w"
    ) as f:
        json.dump(progress, f, indent=4)


def calc_progress(boxes_filt, id, step, total_step):
    total_s = total_step * boxes_filt
    current = id * total_step + step
    value = int((current / total_s) * 100)
    return f"{value}%"


def check_for_abort_render(render_dir, shot_name, uuid, logger, is_tracking=False):
    is_abort = os.path.isfile(
        os.path.join(render_dir, "cancel_render").replace("\\", "/")
    )
    if is_abort and not is_tracking:
        os.remove(os.path.join(render_dir, "cancel_render").replace("\\", "/"))
    if is_abort:
        logger.info("Interrupting render as we detected an abort request from the user")
        write_stats_file(render_dir, shot_name, uuid, "0%", "0%", "False", "True")
    return is_abort
