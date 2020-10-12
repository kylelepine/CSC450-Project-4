from cv2 import cv2
import numpy as np

class template_generator:
    def __init__(self):
        self.refPt = []
        self.cropping = False
        self.current_frame = None

    def click_and_crop(self, event, x, y, flags, param):
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt = [(x, y)]
            self.cropping = True
        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            self.refPt.append((x, y))
            self.cropping = False
            # draw a rectangle around the region of interest
            # TODO: Display rectangle in current frame without it affecting the 'original_frame'.
            cv2.rectangle(self.current_frame, self.refPt[0], self.refPt[1], (0, 255, 0),2)
            # cv2.imshow("current_frame", current_frame)

    def crop_template(self):
        template_types = ['upright', 'falling', 'sitting', 'lying']
        template_type = None
        file_count = 0
        roi = np.array([])

        cv2.namedWindow("current_frame")
        cv2.setMouseCallback("current_frame", self.click_and_crop)
        
        image_path = f'./templates/layered_frames/test_template{file_count}.png'
        self.current_frame = cv2.imread(image_path)
        original_frame = np.copy(self.current_frame)

        print("""
                Click the key to save the 'ROI' as a template
                Option:    Template Type:
                1----------Upright
                2----------Falling
                3----------Sitting Down
                4----------Lying Down      
                """)

        while True:
            if self.cropping == True:
                self.current_frame = original_frame

            cv2.imshow("current_frame", self.current_frame)    
            # cv2.imshow('original_frame', original_frame)    
            
            if len(self.refPt) == 2:
                # self.refPt = [[x1, y1], [x2, y2]]
                x1 = self.refPt[0][0]
                x2 = self.refPt[1][0]
                y1 = self.refPt[0][1]
                y2 = self.refPt[1][1]
                if ((y1 != y2) & (x1 != x2)):

                    if ((y1 < y2) &(x1 < x2)):
                        roi = np.copy(original_frame[y1:y2, x1:x2])
                    if ((y1 > y2) &(x1 < x2)):
                        roi = np.copy(original_frame[y2:y1, x1:x2])
                    if ((y1 < y2) &(x1 > x2)):
                        roi = np.copy(original_frame[y1:y2, x2:x1])
                    if ((y1 > y2) &(x1 > x2)):
                        roi = np.copy(original_frame[y2:y1, x2:x1])
                    # TODO: resize ROI so that it doesn't display previous ROI image data.
                    cv2.imshow("ROI", roi)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('['):
                if file_count != 0:
                    file_count -= 1
                    image_path = f'./templates/layered_frames/test_template{file_count}.png'
                    self.current_frame = cv2.imread(image_path)
                    original_frame = self.current_frame.copy()

            elif key == ord(']'):
                # TODO: fix to where it doesn't access files out of range
                file_count += 1
                image_path = f'./templates/layered_frames/test_template{file_count}.png'
                self.current_frame = cv2.imread(image_path)
                original_frame = self.current_frame.copy()
            
            elif key == ord('q'):
                break
        
            # keys to save a template
            if roi.size != 0:
                if key == ord('1'):
                    template_type = template_types[0]
                    cv2.imwrite(f'./templates/cropped_templates/{template_type}{file_count}.png', roi)
                elif key == ord('2'):
                    template_type = template_types[1]
                    cv2.imwrite(f'./templates/cropped_templates/{template_type}{file_count}.png', roi)
                elif key == ord('3'):
                    template_type = template_types[2]
                    cv2.imwrite(f'./templates/cropped_templates/{template_type}{file_count}.png', roi)
                elif key == ord('4'):
                    template_type = template_types[3]
                    cv2.imwrite(f'./templates/cropped_templates/{template_type}{file_count}.png', roi)
            
        cv2.destroyAllWindows()

def main():
    print('main()')
    tg = template_generator()
    tg.crop_template()
    
if __name__ == '__main__':
    main()