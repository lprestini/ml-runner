import cv2
import time

class BoxSelector():
    # A click-and-hold bounding box selector
    # Bounding box output format: [x, y, w, h]
    # (x, y): coordinates of the top-left corner
    # (w, h): bounding box width and height
    def __init__(self):
        self.start_point = None
        self.end_point = None
        self.temp_img = None
        self.changed = False
        self.finished = False
        self.win_name = 'bboxer'

    def mouse_handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.end_point = None
            self.changed = True
            time.sleep(0.02)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.start_point is not None:
                self.end_point = (x, y)
                self.changed = True
                time.sleep(0.02)
        elif event == cv2.EVENT_LBUTTONUP:
            self.end_point = (x, y)
            self.finished = True
            time.sleep(0.02)

    def select_box(self, img_in):
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win_name, self.mouse_handler)
        
        vis_img = img_in.copy()

        while True:
            if self.start_point and self.end_point and self.changed:
                vis_img = cv2.rectangle(img_in.copy(), self.start_point, self.end_point, (0, 255, 255), 2)
            
            cv2.imshow(self.win_name, vis_img)
            key_ = cv2.waitKey(20)

            if key_ == 27 or self.finished:
                break

        cv2.destroyAllWindows()

        if not self.start_point or not self.start_point:
            print('Error: Bounding box has not been drawn.')
            exit(-1)

        x0 = min([self.start_point[0], self.end_point[0]])
        x1 = max([self.start_point[0], self.end_point[0]])
        y0 = min([self.start_point[1], self.end_point[1]])
        y1 = max([self.start_point[1], self.end_point[1]])

        bbox_out = [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]
        
        return bbox_out
