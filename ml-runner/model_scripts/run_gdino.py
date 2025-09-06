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
from PIL import Image
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
import os


class run_gdino(object):
    def __init__(self,image_pil, model_config_path, model_checkpoint_path, caption, box_threshold, text_threshold = None, with_logits = True, is_pil = True, H = None, W = None):
        self.image_pil = image_pil
        self.model_config_path = model_config_path
        self.model_checkpoint_path = model_checkpoint_path

        # Check model config and checkpoints exists
        assert os.path.isfile(model_config_path), 'Model config path does not exists, please check your folder structures!'
        assert os.path.isfile(model_checkpoint_path), 'Model checkpoint path does not exists, please check your folder structures!'

        self.model = self.load_model()
        self.caption = caption
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.with_logits = with_logits
        self.is_pil = is_pil
        self.H = H
        self.W = W

        ## Returned parameters
        self.boxes = None
        self.labels = None
        self.transformed_image = None
        self.boxes_filt = None
        self.pred_phrases = None
        return None

    def load_image(self):
        # load image

        # If image isnt PIL we need to convert it to PIL as its simpler than going edit every file in gdino.
        if not self.is_pil:
            self.image_pil = Image.fromarray(np.uint8(self.image_pil * 255))
            self.is_pil = True

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        image, _ = transform(self.image_pil, None)  # 3, h, w
        return image

    def load_model(self, cpu_only=False):
        args = SLConfig.fromfile(self.model_config_path)
        args.device = "cuda" if not cpu_only else "cpu"
        model = build_model(args)
        checkpoint = torch.load(self.model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        _ = model.eval()
        return model


    def get_grounding_output(self, cpu_only=False, token_spans=None):
        assert self.text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
        caption = self.caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        device = "cuda" if not cpu_only else "cpu"
        model = self.model.to(device)
        image = self.transformed_image.to(device)
        shape_size = image.shape

        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)

        # filter output
        if token_spans is None:
            logits_filt = logits.cpu().clone()
            boxes_filt = boxes.cpu().clone()
            filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
            logits_filt = logits_filt[filt_mask]  # num_filt, 256
            boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

            # get phrase
            tokenlizer = model.tokenizer
            tokenized = tokenlizer(caption)
            # build pred
            pred_phrases = []
            for logit, box in zip(logits_filt, boxes_filt):
                pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenlizer)
                if self.with_logits:
                    pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
                else:
                    pred_phrases.append(pred_phrase)
        else:
            # given-phrase mode
            positive_maps = create_positive_map_from_span(
                model.tokenizer(caption),
                token_span=token_spans
            ).to(image.device) # n_phrase, 256

            logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
            all_logits = []
            all_phrases = []
            all_boxes = []
            for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
                # get phrase
                phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
                # get mask
                filt_mask = logit_phr > self.box_threshold
                # filt box
                all_boxes.append(boxes[filt_mask])
                # filt logits
                all_logits.append(logit_phr[filt_mask])
                if self.with_logits:
                    logit_phr_num = logit_phr[filt_mask]
                    all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
                else:
                    all_phrases.extend([phrase for _ in range(len(filt_mask))])
            self.boxes_filt = torch.cat(all_boxes, dim=0).cpu()
            self.pred_phrases = all_phrases

        return boxes_filt, pred_phrases

    def flip(self, boxes, labels):
        
        for box, label in zip(boxes, labels):
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        return box
    
    def get_bbox(self, boxes, _W,_H):
        processed_box = []
        for box in boxes:
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([_W, _H, _W, _H])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            processed_box.append(box.cpu().numpy())
        return processed_box


    def run(self,):
        self.transformed_image = self.load_image()

        H,W = (self.H, self.W)
        if self.H == None or self.W == None:
            if self.is_pil:
                H,W = (self.image_pil.height, self.image_pil.width)
            else:
                H,W,C = self.image_pil.shape

        # run model
        self.boxes_filt, self.pred_phrases = self.get_grounding_output(False, None)
        self.boxes_filt = self.get_bbox(self.boxes_filt, W,H)

        return self.boxes_filt, self.pred_phrases