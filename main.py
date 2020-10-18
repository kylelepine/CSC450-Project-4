import numpy as np
from cv2 import cv2
from timeit import default_timer as timer

# Our moduels
import GenerateTemplates
import DatabaseFunctionality

templates = {}

def read_img_path_as_byte_str(path):
    print(f'readimage({path})')
    with open(path, 'rb') as f:
        return f.read()

def show_image(source):
    print('show_image()')
    print(type(source))
    cv2.imshow('Image', source)
    while True:
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    return cv2.destroyAllWindows()

def byte_str_to_image_array(source_str):
    print('byteStr_to_image')
    decoded = cv2.imdecode(np.frombuffer(source_str, np.uint8), -1)
    return decoded

# displays image from file
def image_display_test(image_path):
    print('image_display_test()')
    image_bytes = read_img_path_as_byte_str(image_path)
    img = byte_str_to_image_array(image_bytes)
    print('OpenCV:\n', img)
    show_image(img)

def display(video_path = None, save_template = False, check_template = True):
    frame_count = 0
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

            # image splice by contour detection for foreground
            ret_fg, thresh_fg = cv2.threshold(layered_frames, 91, 255, cv2.THRESH_BINARY)
            contours_foreground = cv2.findContours(thresh_fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
            contour_frame = frame.copy()
            spliced_foreground_frame = contour_frame.copy()
            spliced_edge_frame = contour_frame.copy()
            if len(contours_foreground) != 0:
                contour = max(contours_foreground, key = cv2.contourArea)
                x_pos, y_pos, width, height = cv2.boundingRect(contour)
                buffer_space = 40
                box_area = (width * x_pos) * (height * y_pos)
                min_area = 10000
                if (abs(box_area) > min_area):
                    if(width - x_pos + buffer_space > height - y_pos):
                        spliced_edge_frame = np.copy(cropped_edges[y_pos:(y_pos + height), x_pos:(x_pos + width)])
                        spliced_foreground_frame = np.copy(foreground_morph_dilate[y_pos:(y_pos + height), x_pos:(x_pos + width)])
                        cv2.rectangle(contour_frame, (x_pos, y_pos), (x_pos + width, y_pos + height), (0, 0, 255), 2)
                        
                    else:
                        spliced_edge_frame = np.copy(cropped_edges[y_pos:(y_pos + height), x_pos:(x_pos + width)])
                        spliced_foreground_frame = np.copy(foreground_morph_dilate[y_pos:(y_pos + height), x_pos:(x_pos + width)])
                        cv2.rectangle(contour_frame, (x_pos, y_pos), (x_pos + width, y_pos + height), (0, 255, 0), 2)
            
            if check_template:
                if spliced_foreground_frame.shape != frame.shape:
                    comp_start = timer()
                    highest_similarity = compare_template_to_frame(templates['edge']['upright'][0], spliced_foreground_frame)
                    comp_end = timer()
                    if highest_similarity > 50:
                        print(f'highest_similarity_percent: {highest_similarity}')
                        print(f"Compared in {comp_end-comp_start} seconds.")
            
            # Stacking the images to print them together
            # For comparison
            gray_frames = np.hstack(( gray,  gray_filtered))
            edge_detection_frames = np.hstack((edges_filtered,  cropped_edges))
            foreground_morphs = np.hstack((foreground_morph_close, foreground_morph_open))
            
            # # Display the resulting frame
            # cv2.imshow('gray_frames', gray_frames)
            # cv2.imshow('edge_detection_frames', edge_detection_frames)
            # cv2.imshow('Foreground Detection', foreground)
            # cv2.imshow('foreground_morphs', foreground_morphs)
            # cv2.imshow('layered_frames', layered_frames)
            cv2.imshow('contour frame', contour_frame)
            # cv2.imshow('spliced_foreground_frame', spliced_foreground_frame)
            # cv2.imshow('spliced_edge_frame', spliced_edge_frame)

            if save_template:
                save_path_frame = f"./templates/layered_frames/{'webcam' if video_path is None else video_path[15:-4]}_{str(frame_count)}.png"
                save_path_foreground_template = f"./templates/cropped_templates/foreground/{'webcam' if video_path is None else video_path[15:-4]}_{str(frame_count)}.png"
                save_path_edge_template = f"./templates/cropped_templates/edges/{'webcam' if video_path is None else video_path[15:-4]}_{str(frame_count)}.png"

                # print(save_path)
                cv2.imwrite(save_path_frame, foreground_morph_dilate)
                if spliced_foreground_frame.shape != frame.shape:
                    cv2.imwrite(save_path_foreground_template, spliced_foreground_frame)
                    cv2.imwrite(save_path_edge_template, spliced_edge_frame)
                frame_count += 1

            # controls
            key = cv2.waitKey(25) & 0xFF
            # Press Q on keyboard to exit
            if key == ord('q'):
                break
            # save frame as template
            elif key == ord('1'):
                save_template = not save_template
                print(f'save_template: {save_template}')
        # Break the loop
        else: 
            break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()

def compare_template_to_frame(template, frame):
    highest_similarity = 0.0
    return highest_similarity

# def compare_template_to_frame(template, frame):
#     highest_similarity = 0.0
#     # print('compare_template_to_frame()')
#     # template = cv2.imread(template_path)
#     # frame = cv2.imread(frame_path)
#     # template_size = template.shape[0] * template.shape[1]
#     # frame_size = frame.shape[0] * frame.shape[1]
#     if template.shape[0] <= frame.shape[0]:
#         if template.shape[1] <= frame.shape[1]:
#             row_difference = frame.shape[0] - template.shape[0] 
#             column_difference = frame.shape[1] - template.shape[1]  
#             n = 0
#             print(row_difference)
#             print(column_difference)
#             for starting_ypoint in range(0, row_difference + 1 ,template.shape[1]//2):
#                 for starting_xpoint in range(0, column_difference + 1,template.shape[1]//2):
#                     n += 1
#                     temp = image_compare(frame, template, (starting_ypoint, starting_xpoint))
#                     if temp > highest_similarity:
#                         highest_similarity = temp
#             print(f'comparisons: {n}')
#     return highest_similarity
    
# def image_compare(source, comparison, starting_point):
#     common_pixels = 0
#     total_pixels_compared = comparison.shape[0] * comparison.shape[1]
#     similarity_percent = 0.0
    
#     for y in range(starting_point[0], starting_point[0] + comparison.shape[0], 1):
#         for x in range(starting_point[1], starting_point[1]+ comparison.shape[1], 1):
            
#             comp_x = x - starting_point[1]
#             comp_y = y - starting_point[0]
#             if (source[y][x] == comparison[comp_y][comp_x]).all():
#                 common_pixels += 1
            
#     similarity_percent = common_pixels/total_pixels_compared * 100
#     return similarity_percent

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
            try:
                selection = int(input("Enter: "))
                if selection < i:
                    print("Would you like to save the frames as templates?(y/n):")
                    save_templates = input()
                    save_templates = True if save_templates == 'y' else False
                    print("Would you like to compare templates to video frame?(y/n):")
                    check_templates = input()
                    check_templates = True if check_templates == 'y' else False
                    display(available_videos[selection], save_templates, check_templates)
                else:
                    print("Incorrect selection.")
            except ValueError as error:
                print(error)
        elif command == '2':
            display(None, False, False)
        elif command == '3':
            template_generator = GenerateTemplates.template_generator()
            template_generator.crop_template()
        elif command == '4': 
            comparison_template = templates['edge']['upright'][0]
            # comparison_frame_str = read_img_path_as_byte_str('./templates/layered_frames/fall-01-cam0_75.png')
            # comparison_frame = byte_str_to_image_array(comparison_frame_str)
            comparison_frame = templates['edge']['upright'][0]
            start = timer()
            highest_similarity = compare_template_to_frame(comparison_template, comparison_frame)
            end = timer()
            print(f"Compared in {end-start} seconds.")
            print(f'highest_similarity_percent: {highest_similarity}')
        elif command == '5':
            DatabaseFunctionality.user_interface()
            load_templates()
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
    print(templates)
    # test_byte_str = templates['upright'][0]
    # img = byteStr_to_image(test_byte_str)
    # show_image(img)
    User_interface()

if __name__ == '__main__':
    main()
    