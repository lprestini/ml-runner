import numpy as np
import cv2


# box 1 pixel 

def overlay_rectangle(img, rect, color=(0, 255, 0), line_width=2):
    if rect is not None:
        tl_ = (int(round(rect[0])), int(round(rect[1])))
        br_ = (int(round(rect[0] + rect[2])), int(round(rect[1] + rect[3])))
        cv2.rectangle(img, tl_, br_, color, line_width)

def overlay_mask(img, mask, color=(0, 255, 0), line_width=2, alpha=0.6):
    if mask is not None:
        m = mask.astype(np.float32)
        m_bin = m > 0.5
        img_r = img[:, :, 0]
        img_g = img[:, :, 1]
        img_b = img[:, :, 2]
        img_r[m_bin] = alpha * img_r[m_bin] + (1 - alpha) * color[0]
        img_g[m_bin] = alpha * img_g[m_bin] + (1 - alpha) * color[1]
        img_b[m_bin] = alpha * img_b[m_bin] + (1 - alpha) * color[2]

        # draw contour around mask
        M = m_bin.astype(np.uint8)
        if cv2.__version__[0] == '4':
            contours, _ = cv2.findContours(M, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(M, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, color, line_width)


class VisualizerSimple():
    def __init__(self, window_name='Window'):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
    
    def show(self, img, mask=None, box=None):
        # suppose image is in RGB format
        img_vis = img.copy()
        
        if mask is not None:
            overlay_mask(img_vis, mask, color=(255, 255, 0), line_width=1, alpha=0.7)
        
        if (img_vis.shape[0] * img_vis.shape[1]) > 1000000:
            img_vis = cv2.resize(img_vis, (0, 0), fx=0.5, fy=0.5)
        
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.window_name, img_vis)
        key_ = cv2.waitKey(0)

        if key_ == 27:
            exit(0)


class VisualizerTracking():
    def __init__(self, window_name='Window'):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        self.wait_time = 0
    
    def show(self, img, mask=None, box=None):
        # suppose image is in RGB format
        img_vis = img.copy()
        
        if mask is not None:
            overlay_mask(img_vis, mask, color=(255, 255, 0), line_width=1, alpha=0.7)
        
        if box is not None:
            overlay_rectangle(img_vis, box, color=(255, 255, 0))

        if (img_vis.shape[0] * img_vis.shape[1]) > 1000000:
            img_vis = cv2.resize(img_vis, (0, 0), fx=0.5, fy=0.5)
        
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.window_name, img_vis)
        key_ = cv2.waitKey(self.wait_time)
        
        if key_ == 27:
            exit(0)
        elif key_ == 32:
            if self.wait_time == 0:
                self.wait_time = 1
            else:
                self.wait_time = 1
