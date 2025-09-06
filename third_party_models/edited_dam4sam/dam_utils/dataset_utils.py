import os
import glob
import yaml
import numpy as np
import cv2
from PIL import Image

def load_mask(mask_path):
    mask = np.asarray(Image.open(mask_path)).astype(np.float32)
    mask = (mask > 0).astype(np.uint8)
    return mask

def pil2array(img):
    return np.asarray(img)

class MyDataset():
    def __init__(self, base_dir, init_masks=None):
        self.sequence_list = []
        self.base_dir = base_dir
        self.sequences = {}
        self.init_masks = init_masks

        if self.init_masks is not None:
            if not isinstance(self.init_masks, str):
                print('Error: init_masks parameter must be string.')
                exit(-1)
            with open("./dam4sam_config.yaml") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            base_dir_ = config['box_datasets_gt_masks_path']
            self.init_masks_dir = os.path.join(base_dir_, self.init_masks)

    def get_seq_name(self, seq_index):
        return self.sequence_list[seq_index]
    
    def size(self):
        return len(self.sequence_list)
    
    def get_seq_len(self, sequence_name):
        if sequence_name not in self.sequences:
            self._load_sequence(sequence_name)
        return len(self.sequences[sequence_name]['frames'])
    
    def get_frame(self, sequence_name, frame_index):
        if sequence_name not in self.sequences:
            self._load_sequence(sequence_name)
        frame_path = self.sequences[sequence_name]['frames'][frame_index]
        img = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
        return img

    def get_pil_frame(self, sequence_name, frame_index):
        if sequence_name not in self.sequences:
            self._load_sequence(sequence_name)
        frame_path = self.sequences[sequence_name]['frames'][frame_index]
        image = Image.open(frame_path)
        return image

    def get_groundtruth(self, sequence_name, frame_index):
        if sequence_name not in self.sequences:
            self._load_sequence(sequence_name)
        return self.sequences[sequence_name]['gt'][frame_index]

    def get_init_mask(self, sequence_name):
        if self.init_masks is None:
            print('Error: Loader is not set to load init masks.')
            exit(-1)
        if sequence_name not in self.sequences:
            self._load_sequence(sequence_name)
        
        init_mask_path = self.sequences[sequence_name]['init_mask_path']
        init_mask = load_mask(init_mask_path)
        return init_mask

    def _load_sequence(self, sequence_name):
        pass


class GOTDataset(MyDataset):
    def __init__(self, base_dir, init_masks=None):
        super().__init__(base_dir, init_masks=init_masks)

        with open(os.path.join(self.base_dir, 'list.txt'), 'r') as f:
            self.sequence_list = [line_.strip() for line_ in f.readlines()]
        
        if self.init_masks is not None:
            self.init_masks_dir = os.path.join(self.init_masks_dir, 'got')
        
    def _load_sequence(self, sequence_name):
        if sequence_name in self.sequences:
            return
        
        seq_dict = {}
        
        # load groundtruth
        with open(os.path.join(self.base_dir, sequence_name, 'groundtruth.txt'), 'r') as f:
            gt = []
            for line_ in f.readlines():
                gt.append([float(el) for el in line_.strip().split(',')])
            seq_dict['gt'] = gt
        
        # load path to all frames
        seq_dict['frames'] = sorted(glob.glob(os.path.join(self.base_dir, sequence_name, '*.jpg')))

        # create path to the initialization mask
        if self.init_masks is not None:
            init_mask_path = os.path.join(self.init_masks_dir, '%s.png' % sequence_name)
            if not os.path.exists(init_mask_path):
                print('Error: Init mask on this path does not exist.')
                print(init_mask_path)
                exit(-1)
            seq_dict['init_mask_path'] = init_mask_path

        self.sequences[sequence_name] = seq_dict


class LasotDataset(MyDataset):
    def __init__(self, base_dir, init_masks=None):
        super().__init__(base_dir, init_masks=init_masks)

        with open(os.path.join('./lasot_testing_set.txt'), 'r') as f:
            self.sequence_list = [line_.strip() for line_ in f.readlines()]
        
        if self.init_masks is not None:
            self.init_masks_dir = os.path.join(self.init_masks_dir, 'lasot')
        
    def _load_sequence(self, sequence_name):
        if sequence_name in self.sequences:
            return
        
        seq_dict = {}
        category_name = sequence_name[:sequence_name.find('-')]
        
        # load groundtruth
        with open(os.path.join(self.base_dir, category_name, sequence_name, 'groundtruth.txt'), 'r') as f:
            gt = []
            for line_ in f.readlines():
                gt.append([float(el) for el in line_.strip().split(',')])
            seq_dict['gt'] = gt
        
        # load path to all frames
        seq_dict['frames'] = sorted(glob.glob(os.path.join(self.base_dir, category_name, sequence_name, 'img', '*.jpg')))

        # create path to the initialization mask
        if self.init_masks is not None:
            init_mask_path = os.path.join(self.init_masks_dir, '%s.png' % sequence_name)
            if not os.path.exists(init_mask_path):
                print('Error: Init mask on this path does not exist.')
                print(init_mask_path)
                exit(-1)
            seq_dict['init_mask_path'] = init_mask_path

        self.sequences[sequence_name] = seq_dict

class LasotExtDataset(LasotDataset):
    def __init__(self, base_dir, init_masks=None):
        super().__init__(base_dir, init_masks=init_masks)

        with open(os.path.join('./lasot_ext_testing_set.txt'), 'r') as f:
            self.sequence_list = [line_.strip() for line_ in f.readlines()]
        
        if self.init_masks is not None:
            self.init_masks_dir = os.path.join(self.init_masks_dir, 'lasot_ext')

def get_dataset(dataset_name, init_masks=None):
    with open("./dam4sam_config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if dataset_name == 'got':
        return GOTDataset(config['got_10k_dataset_path'], init_masks=init_masks)
    elif dataset_name == 'lasot':
        return LasotDataset(config['lasot_dataset_path'], init_masks=init_masks)
    elif dataset_name == 'lasot_ext':
        return LasotExtDataset(config['lasot_ext_dataset_path'], init_masks=init_masks)
    else:
        print('Error: Unknown dataset -', dataset_name)
        exit(-1)
