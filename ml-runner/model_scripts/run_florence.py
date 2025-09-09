import sys
# dino_path = '/workspace/lucap/GroundingDINO'
# sys.path.append(dino_path)
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import sys
import requests

from transformers import AutoProcessor, AutoModelForCausalLM 



class run_florence(object):
    def __init__(self,image, model, prcoessor, caption, box_threshold, text_threshold = None, with_logits = True, H = None, W = None):
        self.image = image

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        # Check model config and checkpoints exists
        # assert os.path.isfile(model_config_path), 'Model config path does not exists, please check your folder structures!'
        # assert os.path.isfile(model_checkpoint_path), 'Model checkpoint path does not exists, please check your folder structures!'

        self.model = model
        self.processor = prcoessor
        self.caption = caption
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.with_logits = with_logits
        self.H = H
        self.W = W

        ## Returned parameters
        self.boxes = None
        self.labels = None
        self.transformed_image = None
        self.boxes_filt = None
        self.pred_phrases = None
        return None

    def run_example(self, image, task_prompt, text_input=''):

        prompt = task_prompt + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"].cuda(),
            pixel_values=inputs["pixel_values"].cuda(),
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
            early_stopping=False,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        if not self.H or not self.W:
            self.H, self.W, C = self.image.shape
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(self.W, self.H)
        )
        return parsed_answer
    
    def run(self):
        result = self.run_example(self.image, '<CAPTION_TO_PHRASE_GROUNDING>', self.caption)
        return result['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes'], result['<CAPTION_TO_PHRASE_GROUNDING>']['labels']