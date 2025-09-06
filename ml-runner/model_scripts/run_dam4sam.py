import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import sys
import glob 
from tqdm import tqdm

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")

print(f"using device: {device}")
# print(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'third_party_models', 'edited_dam4sam'))
from edited_dam4sam.dam4sam_tracker import DAM4SAMTracker
from mlrunner_utils.logs import write_stats_file, calc_progress
import PyOpenColorIO as OCIO

class runDAM4SAM(object):
    def __init__(self, checkpoint_path, first_frame_sequence, video_dir, frame_names, render_dir, render_name, boxes_filt, ann_frame_idx, pred_phrases, shot_name,logger, uuid, H = None,W = None, use_gdino = True, limit_range = None):

        self.predictor = DAM4SAMTracker('sam21pp-L', ch = checkpoint_path)
        self.first_frame_sequence = first_frame_sequence
        self.frame_names = frame_names
        self.video_dir = video_dir
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
        self.limit_range = limit_range
        self.pred_phrases = self.pred_phrases if self.use_gdino else ['' for i in self.pred_phrases]
        self.track_progress = 0
        self.is_pil = not shot_name.endswith('.exr')


        ##Debug paramters
        self.render = True ## This is for debug only
        self.plot_results = False 

    def prep_box_for_dam(self, box, H,W, from_nuke = False):
        """When using DAM4SAM the order of the bbox is as follow:
        left,top,abs(left-right),abs(top-bottom)"""
        if from_nuke:
            box = [min(box[0], box[2]), min(box[1], box[3]), max(box[0],box[2]), max(box[1],box[3])]
            # Flip boxes
            box = [box[0],  H-box[3]-1, box[2],  H-box[3]-1 + (box[3]-box[1]),]
            
        box = [box[0], box[1], abs(box[0]- box[2]), abs(box[1]- box[-1])]
        return box

    def convert_linear_to_srgb(self, _img):
        config = OCIO.Config.CreateFromFile(os.environ['MLRUNNER_OCIOCONFIG_PATH'])
        OCIO.SetCurrentConfig(config)
        config = OCIO.GetCurrentConfig()
        ocioprocessor = config.getProcessor('linear', 'srgb')
        cpu = ocioprocessor.getDefaultCPUProcessor()
        cpu.applyRGB(_img)
        return _img

    def load_exr(self, img):
        image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.convert_linear_to_srgb(image)
        return image
    
    def load_frames(self):
        frames_dir = sorted(glob.glob(os.path.join(self.video_dir, '*.%s' % os.path.splitext(self.shot_name)[-1].replace('.','')).replace('\\','/')))
        if self.limit_range:
            frames_dir = frames_dir[self.limit_range[0]: self.limit_range[1]]
        if self.is_pil:
            frames = [Image.open(frame) for frame in frames_dir]
        else:
            frames = [self.load_exr(frame) for frame in frames_dir]
        return frames

    def track_mask_and_render(self, sorted_frames, ann_fram_idx, idx, direction = 'forward', prev_track_progress = 0):

        is_forward = direction == 'forward'
        frames = sorted_frames[ann_fram_idx:] if is_forward else sorted_frames[:ann_fram_idx]
        indexed = enumerate(frames, start = ann_fram_idx) if is_forward else reversed(list(enumerate(frames, start = 0))) 
        total_steps = len(sorted_frames)
        total_boxes = len(self.boxes_filt)
        for iidx, img in tqdm(indexed, total = len(frames), desc = f'Render {direction}'):
            if iidx - ann_fram_idx == 0:
                output = self.predictor.initialize(img, None, bbox = np.array(self.boxes_filt[idx]).astype(np.float32))
            else:
                output = self.predictor.track(img)
            pred_mask = output['pred_mask']
            frame_n = iidx + self.first_frame_sequence 
            if self.limit_range:
                frame_n += self.limit_range[0] + 1
            name_no_frame = f'{self.render_name}_{self.pred_phrases[idx]}_{idx}'
            name = f'{name_no_frame}_{frame_n}.png'
            cv2.imwrite(os.path.join(self.render_dir, name).replace('\\','/'), pred_mask * 255)

            # Compute progress tracking forward
            # TODO this needs testing 
            step = ((iidx + 1 ) - ann_fram_idx) + prev_track_progress if direction == 'forward' else ((ann_fram_idx - (iidx + 1)) + 1 ) + prev_track_progress

            if iidx % 10 == 0:
                track_progress = calc_progress(total_boxes, idx + 1, step, total_steps)
                write_stats_file(self.render_dir, name_no_frame, self.uuid, track_progress, track_progress, False)

        step = ((iidx + 1 ) - ann_fram_idx) + prev_track_progress if direction == 'forward' else ((ann_fram_idx - (iidx + 1)) + 1 ) + prev_track_progress
        track_progress = calc_progress(total_boxes, idx + 1, step, total_steps)
        write_stats_file(self.render_dir, name_no_frame, self.uuid, track_progress, track_progress, False)
        self.track_progress = step

    def run(self):
        self.render_name = self.render_name if not self.render_name == '' else self.shot_name
        sorted_frames = self.load_frames()
        
        if self.limit_range:
            self.ann_frame_idx -= self.limit_range[0]

        # Prep boxes
        for idx,b in enumerate(self.boxes_filt):
            self.boxes_filt[idx] = self.prep_box_for_dam(b, self.H, self.W, from_nuke = not self.use_gdino)

        for idx, b in enumerate(self.boxes_filt):
            self.logger.info(f'Segmenting object ID {idx} out of {len(self.boxes_filt)}')
            ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

            # Cant specify to start from frame 0 so we have to do this in 2 passes if ann frame idx != 0
            # First pass
            self.logger.info('Track mask forward')
            
            
            self.track_mask_and_render(sorted_frames, self.ann_frame_idx, idx,)
            
            if self.ann_frame_idx != 0:
                self.logger.info('Track mask backward')
                self.track_mask_and_render(sorted_frames, self.ann_frame_idx, idx, direction='backward', prev_track_progress= self.track_progress)
