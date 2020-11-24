from cv2 import cv2
import numpy as np

class TemplateViewer:
    template_type = 'upright'
    template_characteristic = 'edge'

    def __init__(self, templates):
        self.templates = templates
        self.current_image = None
        self.images = np.array([])
        self.images_length = len(self.images)
        self.current_image_index = 0

    def view_templates(self):

        self.update_image_list(self.template_characteristic, self.template_type)

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

            cv2.imshow("current_image", self.current_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            elif key == ord('['):
                if 0 < self.current_image_index:
                    self.current_image_index -= 1
                print(self.current_image_index)
                self.current_image = self.images[self.current_image_index]

            elif key == ord(']'):
                if self.images_length - 1 > self.current_image_index:
                    self.current_image_index += 1
                print(self.current_image_index)
                self.current_image = self.images[self.current_image_index]
            
            elif key == ord('e'):
                self.current_image_index = 0
                self.update_image_list('edge', self.template_type)

            elif key == ord('f'):
                self.current_image_index = 0
                self.update_image_list('foreground', self.template_type)
            
            elif key == ord('1'):
                self.current_image_index = 0
                self.current_image_index = 0
                self.update_image_list(self.template_characteristic, 'upright')

            elif key == ord('2'):
                self.current_image_index = 0
                self.current_image_index = 0
                self.update_image_list(self.template_characteristic, 'falling')
                
            elif key == ord('3'):
                self.current_image_index = 0
                self.current_image_index = 0
                self.update_image_list(self.template_characteristic, 'sitting')

            elif key == ord('4'):
                self.current_image_index = 0
                self.current_image_index = 0
                self.update_image_list(self.template_characteristic, 'lying')
                
        cv2.destroyAllWindows()

    def update_image_list(self, templateCharacteristic, templateType):
        print(f"templateCharacteristic: {templateCharacteristic}\nType: {templateType}")
        self.template_characteristic = templateCharacteristic
        self.template_type = templateType
        self.images = np.array(self.templates[self.template_characteristic][self.template_type])
        self.images_length = len(self.images)
        self.current_image = self.images[self.current_image_index]