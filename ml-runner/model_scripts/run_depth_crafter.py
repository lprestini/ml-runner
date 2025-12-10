import os
import numpy as np
import torch

from diffusers.training_utils import set_seed
from depth_crafter_ppl import DepthCrafterPipeline
from unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
import cv2
import torchvision.transforms.functional as tvf
import torch.nn.functional as F
from mlrunner_utils.logs import write_stats_file, calc_progress, check_for_abort_render

class run_depth_crafter(object):
    def __init__(self, 
                pretrain_path,
                unet_path,
                numpy_img_list,
                render_dir,
                render_name,
                first_frame_sequence,
                shot_name, 
                logger,
                uuid,
                H = None,
                W = None,
                name_idx = 0,
                delimiter = '.'):
        
        self.pretrain_path = pretrain_path
        self.unet_path = unet_path
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
        
        self.unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            self.unet_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        self.pipe = DepthCrafterPipeline.from_pretrained(
            self.pretrain_path,
            unet=self.unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        self.pipe.to('cuda')
        self.pipe.enable_attention_slicing()

    def convert_to_tensor_and_ensure_legal_size(self, array, max_res = 1024): 
        self.pad_H = round(self.H / 64) * 64
        self.pad_W = round(self.W / 64) * 64
        tensor = torch.cat([torch.unsqueeze(torch.from_numpy(i).permute(2,1,0), dim = 0) for i in array], dim = 0)
        tensor = tvf.pad(tensor, (0, 0, self.pad_W-self.W, self.pad_H-self.H), padding_mode='reflect')
        self.downsize_to_max_size_ratio = max_res/ max(self.pad_W, self.pad_H)
        height = round(self.pad_H * self.downsize_to_max_size_ratio  / 64) * 64
        width =  round(self.pad_W * self.downsize_to_max_size_ratio  / 64) * 64
        tensor = F.interpolate(tensor, size = (int(width), int(height)), mode="bilinear", align_corners=True).permute(0,1,3,2)
        return tensor
    
    def restore_og_size(self, array):
        return cv2.resize(array, (self.W, self.H), interpolation= cv2.INTER_CUBIC)

    def run(self):
        set_seed(30)

        # Stack frames
        ## TODO put assert for img having different H-W betweenf rames
        img_tensors = self.convert_to_tensor_and_ensure_legal_size(self.numpy_img_list)
        stop = False
        total_steps = img_tensors.shape[0]

        if check_for_abort_render(self.render_dir, self.shot_name, self.uuid, self.logger):
            stop = True
        if not stop:
            with torch.inference_mode():
                result = self.pipe(img_tensors,
                    height= img_tensors.shape[2],
                    width= img_tensors.shape[3],
                    output_type="np",
                    guidance_scale = 1.0,
                    num_inference_steps = 5,
                    window_size = 110,
                    overlap = 25,
                    track_time = False).frames[0]
                
            result = result.sum(-1) / result.shape[-1]
            result = (result - result.min()) / (result.max() - result.min())

            for idx, frame in enumerate(result):
                if check_for_abort_render(self.render_dir, self.shot_name, self.uuid, self.logger):
                    break
                filename = f'{self.render_name}_{self.name_idx}'
                frame_n = idx + self.first_frame_sequence
                name = f'{filename}_{str(frame_n)}.exr'
                frame = self.restore_og_size(frame)
                cv2.imwrite(os.path.join(self.render_dir, name).replace('\\','/'), frame)
                if idx % 10:
                    render_progress = calc_progress(1, 0, idx + 1, total_steps)
                    write_stats_file(self.render_dir, [filename], self.uuid, render_progress, '100%', False)
                    print(render_progress)

        