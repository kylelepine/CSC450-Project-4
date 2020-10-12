import numpy as np
from cv2 import cv2

# Our moduels
import GenerateTemplates
import DatabaseFunctionality

templates = {}

def readimage(path):
    print(f'readimage({path})')
    with open(path, 'rb') as f:
        return f.read()

def byteStr_to_image(source_str):
    print('byteStr_to_image')
    decoded = cv2.imdecode(np.frombuffer(source_str, np.uint8), -1)
    return decoded

def show_image(source):
    print('show_image()')
    print(type(source))
    cv2.imshow('Image', source)
    while True:
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    return cv2.destroyAllWindows()

# displays image from file
def image_display_test(image_path):
    print('image_display_test()')
    image_bytes = readimage(image_path)
    img = byteStr_to_image(image_bytes)
    print('OpenCV:\n', img)
    show_image(img)

def display(video_path = None, save_template = False):
    file_count = 0
    if save_template:
        print("Saving frames as template")

    if video_path is not None:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(cv2.CAP_DSHOW)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=10,
        varThreshold=2,
        detectShadows=False)
    print("Click '1' to start saving templates. They will be autmatically cropped to the contour detection bounding box.")
    # Read the video
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Converting the image to grayscale.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Smoothing without removing edges.
            gray_filtered = cv2.bilateralFilter(gray, 7, 75, 75)
            # Extract the foreground
            foreground = fgbg.apply(gray_filtered) 
            # Smooth out to get the moving area
            kernel_close = np.ones((10,10),np.uint8)
            kernel_open = np.ones((10,10),np.uint8)
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))

            foreground_morph_close = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel_close)
            foreground_morph_open = cv2.morphologyEx(foreground_morph_close, cv2.MORPH_OPEN, kernel_open)
            foreground_morph_close = cv2.morphologyEx(foreground_morph_open, cv2.MORPH_CLOSE, kernel_close)
            foreground_morph_dilate = cv2.dilate(foreground_morph_close,kernel_dilate,iterations = 1)

            edges_filtered = cv2.Canny(gray_filtered, 60, 120)
            # Crop off the edges out of the moving area
            cropped_edges = (foreground_morph_dilate // 255) * edges_filtered


            #EXPERIMENTAl
            layered_frames = np.add(cropped_edges, foreground_morph_dilate)
            # image splice by contour detection
            r, thresh = cv2.threshold(layered_frames, 91, 255, cv2.THRESH_BINARY)
            contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
            contour_frame = frame.copy()
            spliced_frame = contour_frame.copy()
            if len(contours) != 0:
                contour = max(contours, key = cv2.contourArea)
                x_pos, y_pos, width, height = cv2.boundingRect(contour)
                buffer_space = 40
                box_area = (width * x_pos) * (height * y_pos)
                min_area = 10000
                if (abs(box_area) > min_area):
                    if(width - x_pos + buffer_space > height - y_pos):
                        spliced_frame = np.copy(layered_frames[y_pos:(y_pos + height), x_pos:(x_pos + width)])
                        cv2.rectangle(contour_frame, (x_pos, y_pos), (x_pos + width, y_pos + height), (0, 0, 255), 2)
                        
                    else:
                        spliced_frame = np.copy(layered_frames[y_pos:(y_pos + height), x_pos:(x_pos + width)])
                        cv2.rectangle(contour_frame, (x_pos, y_pos), (x_pos + width, y_pos + height), (0, 255, 0), 2)
            
            # Stacking the images to print them together
            # For comparison
            gray_frames = np.hstack(( gray,  gray_filtered))
            edge_detection_frames = np.hstack((edges_filtered,  cropped_edges))
            foreground_morphs = np.hstack((foreground_morph_close, foreground_morph_open))
            
            # Display the resulting frame
            cv2.imshow('gray_frames', gray_frames)
            cv2.imshow('edge_detection_frames', edge_detection_frames)
            cv2.imshow('Foreground Detection', foreground)
            cv2.imshow('foreground_morphs', foreground_morphs)
            cv2.imshow('layered_frames', layered_frames)
            cv2.imshow('contour frame', contour_frame)
            cv2.imshow('spliced_frame', spliced_frame)

            if save_template:
                save_path_frame = f"./templates/layered_frames/ {'webcam' if video_path is None else video_path[15:-4]}_{str(file_count)}.png"
                save_path_template = f"./templates/cropped_templates/ {'webcam' if video_path is None else video_path[15:-4]}_{str(file_count)}.png"
                # print(save_path)
                cv2.imwrite(save_path_frame, layered_frames)
                cv2.imwrite(save_path_template, spliced_frame)
                file_count += 1

            # controls
            key = cv2.waitKey(25) & 0xFF
            # Press Q on keyboard to exit
            if key == ord('q'):
                break
            # save frame as template
            elif key == ord('1'):
                save_template = not save_template
                print('save_template: {save_template}')
        # Break the loop
        else: 
            break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()

def compare_template_to_frame(template_path, frame_path):
    print('compare_template_to_frame()')
    template = cv2.imread(template_path)
    frame = cv2.imread(frame_path)
    print(template.shape)
    row_difference = frame.shape[0] - template.shape[0] 
    column_difference = frame.shape[1] - template.shape[1]  
    n = 0
    highest_similarity = 0.0
    for starting_ypoint in range(0, row_difference + 1 ,template.shape[1]//4):
        for starting_xpoint in range(0, column_difference + 1,template.shape[1]//4):
            n += 1
            temp = image_compare(frame, template, (starting_ypoint, starting_xpoint))
            if temp > highest_similarity:
                highest_similarity = temp
    print(f'n: {n} comparisons')
    print(f'highest_similarity_percent: {highest_similarity}')
    
def image_compare(source, comparison, starting_point):
    common_pixels = 0
    total_pixels_compared = comparison.shape[0] * comparison.shape[1]
    similarity_percent = 0.0
    
    for y in range(starting_point[0], starting_point[0] + comparison.shape[0], 1):
        for x in range(starting_point[1], starting_point[1]+ comparison.shape[1], 1):
            
            comp_x = x - starting_point[1]
            comp_y = y - starting_point[0]
            if (source[y][x] == comparison[comp_y][comp_x]).all():
                common_pixels += 1
            
    similarity_percent = common_pixels/total_pixels_compared * 100
    return similarity_percent

def User_interface():
    
    while True:
        print("""
        Command:(button)              Description:
        view_video:(1)                Displays available videos in 'fall_samples' with computer vision.
        view_webcam:(2)               Displays connected webcam with computer vision
        crop_templates:(3)            Allows user to crop templates that exist in 'templates/layered_frames'.
        compare_template:(4)          Demonstrates comparing a template to a frame.
        database:(5)                  Access Database UI.
        quit:(q)
        """)
        command = input("Command: ")
        if command == '1':
            available_videos = ['./fall_samples/fall-01-cam0.mp4', './fall_samples/fall-27-cam0.mp4']
            print("Please choose from: ")
            i = 0
            for video in available_videos:
                print(str(i) + video)
                i+=1
            selection = int(input("Enter: "))
            print("Would you like to save the frames as templates?(y/n):")
            save_templates = input()
            save_templates = True if save_templates == 'y' else False
            display(available_videos[selection], save_templates)
        elif command == '2':
            display()
        elif command == '3':
            template_generator = GenerateTemplates.template_generator()
            template_generator.crop_template()
        elif command == '4': 
            compare_template_to_frame('./templates/cropped_templates/falling69.png', './templates/layered_frames/test_template69.png')
        elif command == '5':
            DatabaseFunctionality.user_interface()
        elif command == 'q':
            break
        else:
            print("incorrect command.")

def load_templates():
    global templates
    dbname = 'CSC-450_FDS'
    pword = 'Apcid28;6jdn'
    database = DatabaseFunctionality.FDSDatabase(dbname, pword)
    database.connect()
    templates = database.load_template_dictionary()

def main():
    print('Starting FDSystem')
    load_templates()
    # test_byte_str = templates['upright'][0]
    # img = byteStr_to_image(test_byte_str)
    # show_image(img)
    User_interface()

if __name__ == '__main__':
    main()
    