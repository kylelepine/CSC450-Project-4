from cv2 import cv2
import numpy as np

class template_modifier:
    template_type = 'upright'
    template_characteristic = 'edge'

    def __init__(self, templates):
        self.templates = templates
        
        self.refPt = []
        self.cropping = False
        self.current_image = None
        
        self.images = np.array([])
        self.images_length = len(self.images)
        self.current_image_index = 0
        
    # def click_and_crop(self, event, x, y, flags, param):
    #     # if the left mouse button was clicked, record the starting
    #     # (x, y) coordinates and indicate that cropping is being
    #     # performed
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         self.refPt = [(x, y)]
    #         self.cropping = True
    #     # check to see if the left mouse button was released
    #     elif event == cv2.EVENT_LBUTTONUP:
    #         # record the ending (x, y) coordinates and indicate that
    #         # the cropping operation is finished
    #         self.refPt.append((x, y))
    #         self.cropping = False
    #         # draw a rectangle around the region of interest
    #         # TODO: Display rectangle in current frame without it affecting the 'original_frame'.
    #         cv2.rectangle(self.current_image, self.refPt[0], self.refPt[1], (0, 255, 0),2)
    #         # cv2.imshow("current_image", current_image)

    def crop_template(self):
        # template_types = ['upright', 'falling', 'sitting', 'lying']
        # images = np.array(self.templates[template_characteristic][template_type])
        # images = images[:,1]

        # current_image_index = 0
        # self.current_image = images[current_image_index]
        self.update_image_list(self.template_characteristic, self.template_type)
        original_frame = np.copy(self.current_image)
        # roi = np.array([])

        cv2.namedWindow("current_image")
        # cv2.setMouseCallback("current_image", self.click_and_crop)

        print("""
                Use following commands to sort viewing template.
                Command:   Template Type:
                1----------Upright
                2----------Falling
                3----------Sitting Down
                4----------Lying Down

                           Template Characteristic:
                f----------foreground
                e----------edge  
                """)

        while True:
            if self.cropping == True:
                self.current_image = original_frame

            cv2.imshow("current_image", self.current_image)    
            # cv2.imshow('original_frame', original_frame)
            
            # if len(self.refPt) == 2:
            #     # self.refPt = [[x1, y1], [x2, y2]]
            #     x1 = self.refPt[0][0]
            #     x2 = self.refPt[1][0]
            #     y1 = self.refPt[0][1]
            #     y2 = self.refPt[1][1]
            #     if ((y1 != y2) & (x1 != x2)):

            #         if ((y1 < y2) &(x1 < x2)):
            #             roi = np.copy(original_frame[y1:y2, x1:x2])
            #         if ((y1 > y2) &(x1 < x2)):
            #             roi = np.copy(original_frame[y2:y1, x1:x2])
            #         if ((y1 < y2) &(x1 > x2)):
            #             roi = np.copy(original_frame[y1:y2, x2:x1])
            #         if ((y1 > y2) &(x1 > x2)):
            #             roi = np.copy(original_frame[y2:y1, x2:x1])
            #         # TODO: resize ROI so that it doesn't display previous ROI image data.
            #         cv2.imshow("ROI", roi)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            elif key == ord('['):
                if 0 < self.current_image_index:
                    self.current_image_index -= 1
                print(self.current_image_index)
                self.current_image = self.images[self.current_image_index]
                original_frame = self.current_image.copy()

            elif key == ord(']'):
                if self.images_length - 1 > self.current_image_index:
                    self.current_image_index += 1
                print(self.current_image_index)
                self.current_image = self.images[self.current_image_index]
                original_frame = self.current_image.copy()
            
            elif key == ord('e'):
                self.current_image_index = 0
                self.update_image_list('edge', self.template_type)
                original_frame = self.current_image.copy()

            elif key == ord('f'):
                self.current_image_index = 0
                self.update_image_list('foreground', self.template_type)
                original_frame = self.current_image.copy()
            
            elif key == ord('1'):
                self.current_image_index = 0
                self.current_image_index = 0
                self.update_image_list(self.template_characteristic, 'upright')
                original_frame = self.current_image.copy()

            elif key == ord('2'):
                self.current_image_index = 0
                self.current_image_index = 0
                self.update_image_list(self.template_characteristic, 'falling')
                original_frame = self.current_image.copy()
                
            elif key == ord('3'):
                self.current_image_index = 0
                self.current_image_index = 0
                self.update_image_list(self.template_characteristic, 'sitting')
                original_frame = self.current_image.copy()

            elif key == ord('4'):
                self.current_image_index = 0
                self.current_image_index = 0
                self.update_image_list(self.template_characteristic, 'lying')
                original_frame = self.current_image.copy()
            
            # keys to save a template
            # if roi.size != 0:
            #     if key == ord('1'):
            #         template_type = template_types[0]
            #         cv2.imwrite(f'./templates/cropped_templates/{template_type}{file_count}.png', roi)
            #     elif key == ord('2'):
            #         template_type = template_types[1]
            #         cv2.imwrite(f'./templates/cropped_templates/{template_type}{file_count}.png', roi)
            #     elif key == ord('3'):
            #         template_type = template_types[2]
            #         cv2.imwrite(f'./templates/cropped_templates/{template_type}{file_count}.png', roi)
            #     elif key == ord('4'):
            #         template_type = template_types[3]
            #         cv2.imwrite(f'./templates/cropped_templates/{template_type}{file_count}.png', roi)
            
        cv2.destroyAllWindows()

    def update_image_list(self, templateCharacteristic, templateType):
        print(f"templateCharacteristic: {templateCharacteristic}\nType: {templateType}")
        self.template_characteristic = templateCharacteristic
        self.template_type = templateType

        self.images = np.array(self.templates[self.template_characteristic][self.template_type], dtype=object)
        self.images = self.images[:,1]
        self.images_length = len(self.images)
        self.current_image = self.images[self.current_image_index]