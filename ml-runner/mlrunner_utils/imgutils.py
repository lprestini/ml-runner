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

import PyOpenColorIO as OCIO
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import os

def convert_linear_to_srgb(_img):
    config = OCIO.Config.CreateFromFile(os.environ['MLRUNNER_OCIOCONFIG_PATH'])
    OCIO.SetCurrentConfig(config)
    config = OCIO.GetCurrentConfig()
    ocioprocessor = config.getProcessor('linear', 'srgb')
    cpu = ocioprocessor.getDefaultCPUProcessor()
    cpu.applyRGB(_img)
    return _img

def convert_srgb_to_linear(_img):
    config = OCIO.Config.CreateFromFile(os.environ['MLRUNNER_OCIOCONFIG_PATH'])
    OCIO.SetCurrentConfig(config)
    config = OCIO.GetCurrentConfig()
    ocioprocessor = config.getProcessor('srgb', 'linear')
    cpu = ocioprocessor.getDefaultCPUProcessor()
    cpu.applyRGB(_img)
    return _img

def load_imgs_to_numpy(frame_to_load, clip = True, rgb_only = True, to_srgb = True):
    images = []
    for idx,i in tqdm(enumerate(frame_to_load), total=len(frame_to_load), desc = 'Loading frames'):
        frame = i.replace('\\','/')
        if '.png' in i:
            image = np.array(Image.open(frame).convert('RGB'))
            if image.dtype == np.uint8:
                image = image.astype(np.float64) / 255.0

            if not to_srgb:
                image = image.astype(np.float32)
                image = convert_srgb_to_linear(image)
        else:
            image = cv2.imread(frame, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if to_srgb:
                image = convert_linear_to_srgb(image)
            if clip:
                image = np.clip(image, 0,1)

        # Limit to read only first 3 channels
        if image.shape[-1] > 3 and rgb_only:
            image = image[:3,:,:]

        images.append(image)
    return images

def load_mov_to_numpy(path_to_sequence, shot_name, last_frame, first_frame):
    """Function to load a mov into a numpy array - it'll retrun a list of np.arrays and list bools saying if frame was laoded"""
    images = []
    frames = []
    video = cv2.VideoCapture(os.path.join(path_to_sequence, shot_name))
    video.set(cv2.CAP_PROP_POS_FRAMES, int(first_frame))
    for f in tqdm(enumerate(list(range(first_frame, last_frame))), total = len(list(range(first_frame, last_frame)))):
        succ, image = video.read()
        if succ:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.dtype == np.uint8:
                image = np.clip(image.astype(np.float64) / 255.0,0,1)
            images.append(image)
        frames.append(succ)
    return images, frames

def get_im_width_height(img, is_pil = False):
    if is_pil:
        H,W = (img.height, img.width)
    else:
        H,W,C = img.shape 
    return H,W

def get_frame_list(path_to_sequence, shot_name, delimiter):
    # TODO right now this works only if there is 1 sequence per folder. 
    frame_names = [
        p for p in os.listdir(path_to_sequence)
        if os.path.splitext(shot_name)[0].split(delimiter)[0] in p and p.endswith('.png') or p.endswith('.exr')   
        ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split(delimiter)[-1]))
    frame_names = [os.path.join(path_to_sequence, i) for i in frame_names]
    return frame_names