import os
import numpy as np
import torch

from cotracker.predictor import CoTrackerOnlinePredictor
import cv2
import torchvision.transforms.functional as tvf
import torch.nn.functional as F
from mlrunner_utils.logs import write_stats_file, calc_progress, check_for_abort_render

class run_cotracker(object):
    def __init__(self, 
                pretrain_path,
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
                H = None,
                W = None,
                name_idx = 0,
                delimiter = '.'):
        
        self.pretrain_path = pretrain_path
        self.numpy_img_list = numpy_img_list
        self.render_dir = render_dir
        self.render_name = render_name
        self.boxes_filt = boxes_filt
        self.grid_size = int(grid_size)
        self.use_grid = use_grid
        self.ann_frame_idx = ann_frame_idx
        self.first_frame_sequence = first_frame_sequence
        self.shot_name = shot_name
        self.logger = logger
        self.uuid = uuid
        self.H = H
        self.W = W
        self.name_idx = name_idx
        self.delimiter = delimiter
        
        self.predictor = CoTrackerOnlinePredictor(checkpoint=self.pretrain_path, window_len=16, v2=False).to('cuda')

    def pad_tensor_to_multiple16(self, x, direction = 'both'):
        ann_idx = self.ann_frame_idx if direction == 'tail' else self.ann_frame_idx + 1
        B,C,H,W = x.shape
        tot_pad_amount = 16 - ((B - ann_idx) % 16)
        pad_side_1 = tot_pad_amount // 2
        pad_side_2 = tot_pad_amount - pad_side_1
        chunk_front = torch.flip(x[:pad_side_1], dims=[0])
        chunk_end = torch.flip(x[-pad_side_2:], dims=[0])
        padded_tensor = torch.cat([chunk_front, x, chunk_end], dim = 0)
        if direction == 'head':
            chunk_front = torch.flip(x[:tot_pad_amount], dims=[0])
            padded_tensor = torch.cat([chunk_front, x[:ann_idx,]], dim = 0)
            pad_side_1 = tot_pad_amount
            pad_side_2 = 0
        elif direction == 'tail':
            chunk_end = torch.flip(x[-tot_pad_amount:], dims=[0])
            padded_tensor = torch.cat([x[ann_idx:, ], chunk_end], dim = 0)
            pad_side_2 = tot_pad_amount
            pad_side_1 = 0

        return padded_tensor, pad_side_1, pad_side_2

    def prep_images(self, x):
        return x.permute(0,3,1,2)[None].to('cuda').float() * 255
    
    def tracks_2_tracker(self, tracks):
        # Tracks with no Batch BxTxNx2 -> TxNx2 
        # T being frames
        np_tracks = tracks.detach().cpu().numpy()[0]
        T,N,XY = np_tracks.shape
        tracks = {}
        for track_n  in range(N):
            all_points = np_tracks[:,track_n,:]
            tracks[f'track_{track_n}']= {'coord_x':[], 'coord_y':[]}
            for x,y in all_points:
                tracks[f'track_{track_n}']['coord_x'].append(float(x))
                tracks[f'track_{track_n}']['coord_y'].append((self.H - 1) - float(y))
        
        nuke_tracker = self.build_tracker(tracks)

        return nuke_tracker

    def make_grid_given_crop(self, box):
        """Make a grid of points in a given area."""
        grid_width =  box[-2] 
        grid_height = box[-1] 
        if self.grid_size == 1:
            query = torch.tensor([ grid_height / 2,grid_width / 2], device='cuda')[None, None]
        else:
            y_range = torch.linspace(int(box[1]), int(box[-1]), int(self.grid_size))
            x_range = torch.linspace(int(box[0]) - 4, int(box[-2]) + 4, int(self.grid_size))
            grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
            query = torch.stack([grid_x, grid_y], dim=-1).reshape(1,-1,2).to('cuda')
        return query


    def build_tracker(self, tracks,):
        TRACKER = """Tracker4 {
        tracks { { 1 31 <NUMTRACKS> } 
        { { 5 1 20 enable e 1 } 
        { 3 1 75 name name 1 } 
        { 2 1 58 track_x track_x 1 } 
        { 2 1 58 track_y track_y 1 } 
        { 2 1 63 offset_x offset_x 1 } 
        { 2 1 63 offset_y offset_y 1 } 
        { 4 1 27 T T 1 } 
        { 4 1 27 R R 1 } 
        { 4 1 27 S S 1 } 
        { 2 0 45 error error 1 } 
        { 1 1 0 error_min error_min 1 } 
        { 1 1 0 error_max error_max 1 } 
        { 1 1 0 pattern_x pattern_x 1 } 
        { 1 1 0 pattern_y pattern_y 1 } 
        { 1 1 0 pattern_r pattern_r 1 } 
        { 1 1 0 pattern_t pattern_t 1 } 
        { 1 1 0 search_x search_x 1 } 
        { 1 1 0 search_y search_y 1 } 
        { 1 1 0 search_r search_r 1 } 
        { 1 1 0 search_t search_t 1 } 
        { 2 1 0 key_track key_track 1 } 
        { 2 1 0 key_search_x key_search_x 1 } 
        { 2 1 0 key_search_y key_search_y 1 } 
        { 2 1 0 key_search_r key_search_r 1 } 
        { 2 1 0 key_search_t key_search_t 1 } 
        { 2 1 0 key_track_x key_track_x 1 } 
        { 2 1 0 key_track_y key_track_y 1 } 
        { 2 1 0 key_track_r key_track_r 1 } 
        { 2 1 0 key_track_t key_track_t 1 } 
        { 2 1 0 key_centre_offset_x key_centre_offset_x 1 } 
        { 2 1 0 key_centre_offset_y key_centre_offset_y 1 } 
        } 
        { 
        <TRACKS>
        } 
        }
        reference_frame 20
        translate {{curve x28 0 9.465942383 18.76318359} {curve x28 0 -0.7134399414 -1.827941895}}
        center {{curve x28 1218 1218 1218} {curve x28 685 685 685}}
        selected_tracks 1
        name Tracker1
        selected true
        xpos -1122
        ypos -461
        }"""

        NUM_TRACKS = len(tracks)
        track_template = '{ {curve K xREF_FRAME 1} "track TRACK_NUM" {curve xREF_FRAME COORDS_X} {curve xREF_FRAME COORDS_Y} {curve K xREF_FRAME 0} {curve K xREF_FRAME 0} 1 1 1 {curve xREF_FRAME 0} 0 0 0 0 0 0 0 0 0 0 {} {curve xREF_FRAME 0} {curve xREF_FRAME 0} {curve xREF_FRAME 0} {curve xREF_FRAME 0} {curve xREF_FRAME 0} {curve xREF_FRAME 0} {curve xREF_FRAME 0} {curve xREF_FRAME 0} {curve xREF_FRAME 0} {curve xREF_FRAME 0}  }' 
        formatted_tracks = []

        for idx, track_idx in enumerate(tracks):
            base_format = track_template
            coord_x = ' '.join(str(i) for i in tracks[track_idx]['coord_x'])
            coord_y = ' '.join(str(i) for i in tracks[track_idx]['coord_y'])

            base_format = base_format.replace('REF_FRAME', str(self.first_frame_sequence)).replace('TRACK_NUM', f'{idx + 1}').replace('COORDS_X', coord_x).replace('COORDS_Y', coord_y)
            formatted_tracks.append(base_format)
        formatted_tracks = '\n'.join(i for i in formatted_tracks)
        TRACKER = TRACKER.replace('<TRACKS>', formatted_tracks)
        TRACKER = TRACKER.replace('<NUMTRACKS>', str(NUM_TRACKS))
        return TRACKER
    
    def flip_boxes(self,box, H, W):
        if box[0] > box[2] or box[1]<box[3]:
            box = [min(box[0],box[2]), min(box[1],box[3]), max(box[0],box[2]), max(box[1],box[3]),]
        box = [box[0],  H-box[3]-1, box[2],  H-box[3]-1 + (box[3]-box[1]),]
        return box        

    def track(self, original_img_list, queries = None):
        full_tracks = None
        stop = False

        # Track forward
        fwd_frames_padded, pad_fwd_front, pad_fwd_back = self.pad_tensor_to_multiple16(original_img_list, direction='tail')
        fwd_frames_padded =self.prep_images(fwd_frames_padded)
        self.predictor(video_chunk=fwd_frames_padded, is_first_step=True, grid_size=self.grid_size, add_support_grid = True, queries = queries)
        fwd_tracks = []
        fwd_vis = []
        self.logger.info('Track shot forward')
        for ind in range(0 , fwd_frames_padded.shape[1]-self.predictor.step, self.predictor.step):
            chunked = fwd_frames_padded[:, ind  : ind + self.predictor.step * 2]
            pred_tracks, pred_visibility = self.predictor(video_chunk = chunked)
            fwd_tracks.append(pred_tracks)
            fwd_vis.append(pred_visibility)
            # If we find file while tracking we break loop but not delete the file 
            if check_for_abort_render(self.render_dir, self.shot_name, self.uuid, self.logger, is_tracking=True):
                break
        # And we delete the file here
        if check_for_abort_render(self.render_dir, self.shot_name, self.uuid, self.logger):
            stop = True

        if not stop:
            full_tracks = fwd_tracks[-1]

            if self.ann_frame_idx != 0:
                # We need to add 1 to the reference frame so that the tracking starts on the same frame
                # And remember to remove it later from the padding as well so that we don't have slides/overlapping frames.
                bwd_frames_padded, pad_bwd_front, pad_bwd_back = self.pad_tensor_to_multiple16(original_img_list, direction='head')
                bwd_frames_padded = torch.flip(self.prep_images(bwd_frames_padded), dims =[0,1])
                self.predictor(video_chunk=bwd_frames_padded, is_first_step=True, grid_size=self.grid_size, add_support_grid = True, queries = queries)
                bwd_tracks = []
                bwd_vis = []
                self.logger.info('Track shot backward')
                for ind in range(0, bwd_frames_padded.shape[1]-self.predictor.step, self.predictor.step):
                    chunked = bwd_frames_padded[:, ind : ind + self.predictor.step * 2 , :, :]
                    bwd_pred_tracks, bwd_pred_visibility = self.predictor(video_chunk = chunked)
                    # If we find file while tracking we break loop but not delete the file 
                    if check_for_abort_render(self.render_dir, self.shot_name, self.uuid, self.logger, is_tracking=True):
                        break
                # And we delete the file here
                if check_for_abort_render(self.render_dir, self.shot_name, self.uuid, self.logger):
                    stop = True

                if not stop:
                    # Remove extra frame from pad_bwd_front and from the tracks
                    bwd_tracks.append(torch.flip(bwd_pred_tracks,dims=[0,1])[:,1:-1])
                    pad_bwd_front -= 1
                    
                    # Concat backward and forwrad and remove pads
                    full_tracks = torch.cat([bwd_tracks[0], fwd_tracks[-1]], dim = 1)
                    full_tracks = full_tracks[:,pad_bwd_front:-pad_fwd_back]
            else:
                full_tracks = full_tracks[:,:-pad_fwd_back]
            if isinstance(queries, torch.Tensor):
                full_tracks = full_tracks[:,:,:int((self.grid_size*self.grid_size)),:]
        return full_tracks

    def run(self):
        stop = False
        # Stack frames
        if check_for_abort_render(self.render_dir, self.shot_name, self.uuid, self.logger):
            stop = True

        if not self.boxes_filt:
            self.boxes_filt = [0]
        if not stop:
            filenames = []
            total_boxes = len(self.boxes_filt)

            # Convert to tensor
            original_img_list = torch.concat([torch.unsqueeze(torch.from_numpy(i), dim = 0) for i in self.numpy_img_list],dim =0)
            
            # Resize to HD 
            ratio = 1
            if self.H > 1080 or self.W > 1920:
                ratio = 1920 / self.W
                new_h = int(self.H * ratio)
                new_w = int(self.W * ratio)
                original_img_list = F.interpolate(original_img_list.permute(0,-1, 1,2), size=(new_h, new_w), mode="bilinear", align_corners=True).permute(0,2,3,1)
            
            for idx, box in enumerate(self.boxes_filt):
                queries = None
                # If we arent using a grid - we are tracking a given crop
                # Lets build the traks and pass it to the tracker
                if not self.use_grid:
                    box = self.flip_boxes([int(float(b)) for b in box], self.H, self.W)
                    box = [int(b*ratio) for b in box]
                    queries = self.make_grid_given_crop(box)
                    query_frame = torch.zeros(queries[:,:,1:].shape).to('cuda')
                    queries = torch.cat([query_frame,queries], dim = -1)


                if check_for_abort_render(self.render_dir, self.shot_name, self.uuid, self.logger, is_tracking=True):
                    stop = True

                if not stop:
                    # BxTxNx2
                    full_tracks = self.track(original_img_list,queries=queries)

                    # Rescale tracks 
                    if self.H > 1080 or self.W > 1920:
                        full_tracks = full_tracks * (1/ratio)
                    
                    # Export tracks to nuke
                    nuke_tracker = self.tracks_2_tracker(full_tracks)
                    filename = f'{self.render_name}_crop_{idx}_{self.name_idx}.nk'

                    with open(os.path.join(self.render_dir, filename), 'w') as f:
                        f.write(nuke_tracker)
                    filenames.append(filename)
                    render_progress = calc_progress(total_boxes, idx + 1, 1, 1)
                    write_stats_file(self.render_dir, filenames, self.uuid, render_progress, '100%', False)
                    self.logger.info(f'Track crop{idx} completed!')

            if check_for_abort_render(self.render_dir, self.shot_name, self.uuid, self.logger):
                stop = True
            