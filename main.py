import numpy as np
from cv2 import cv2
import pandas as pd
from timeit import default_timer as timer
from os import walk

# Modules for our system
import TemplateModifier
import DatabaseFunctionality
import CompareTemplates


templates = {}

# Database credentials 
LOCAL_DATABASE_NAME = 'CSC-450_FDS'
LOCAL_DATABASE_PASSWORD = 'Apcid28;6jdn'

def imagePathToByteString(path):
    with open(path, 'rb') as f:
        return f.read()

def byteStringToImage(byteString):
    decoded = cv2.imdecode(np.frombuffer(byteString, np.uint8), -1)
    return decoded

def showImage(source):
    cv2.imshow('Image', source)
    while True:
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    return cv2.destroyAllWindows()

def display(videoPath = None, saveTemplate = False, checkTemplate = False):
    frame_count = 0
    if saveTemplate:
        print("Saving frames as template")

    if videoPath is not None:
        cap = cv2.VideoCapture(videoPath)
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

            layered_frames = np.add(cropped_edges, foreground_morph_dilate)

            # image splice by contour detection for foreground
            _, thresh_fg = cv2.threshold(layered_frames, 91, 255, cv2.THRESH_BINARY)
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
            
            edge_template = cv2.resize(spliced_edge_frame, dsize = (50,75), interpolation=cv2.INTER_CUBIC)
            foreground_template = cv2.resize(spliced_foreground_frame, dsize = (50,75), interpolation=cv2.INTER_CUBIC)

            if checkTemplate:
                if spliced_foreground_frame.shape != frame.shape:
                    k = None

                    total_comparison_time_start = timer()

                    # edge_comparison_start = timer()
                    edge_classification = CompareTemplates.classifyKnn(edge_template, templates['edge'], k)
                    # edge_comparison_end = timer()
                    
                    # print(f"Edge compared in {edge_comparison_end - edge_comparison_start} seconds.")
                    # print(f'edge_classification: {edge_classification}')

                    # foreground_comparison_start = timer()
                    foreground_classification = CompareTemplates.classifyKnn(foreground_template, templates['foreground'], k)
                    # foreground_comparison_end = timer()

                    # print(f"Foreground compared in {foreground_comparison_end - foreground_comparison_start} seconds.")
                    # print(f'foreground_classification: {foreground_classification}')
                    
                    total_comparison_time_end = timer()
                    print(f"Total comparison time {total_comparison_time_end - total_comparison_time_start} seconds.")
                    
                    # if (edge_classification == 'falling') & (foreground_classification == 'falling'):
                        # print("fall")
                    # if edge_classification == 'falling':
                    #         print("Potential fall detected.")
                    # if foreground_classification == 'falling':
                    #         print("Potential fall detected.")
                    
                    
            
            # Stacking the images to print them together
            # For comparison
            gray_frames = np.hstack(( gray,  gray_filtered))
            edge_detection_frames = np.hstack((edges_filtered,  cropped_edges))
            foreground_morphs = np.hstack((foreground_morph_close, foreground_morph_open))
            
            # Display the resulting frame
            # cv2.imshow('gray_frames', gray_frames)
            # cv2.imshow('edge_detection_frames', edge_detection_frames)
            # cv2.imshow('Foreground Detection', foreground)
            # cv2.imshow('foreground_morphs', foreground_morphs)
            # cv2.imshow('layered_frames', layered_frames)
            cv2.imshow('contour frame', contour_frame)
            cv2.imshow('edge_template', edge_template)
            cv2.imshow('foreground_template', foreground_template)

            if saveTemplate:
                save_path_frame = f"./templates/layered_frames/{'webcam' if videoPath is None else videoPath[15:-4]}_{str(frame_count)}.png"
                save_path_foreground_template = f"./templates/cropped_templates/foreground/{'webcam' if videoPath is None else videoPath[15:-4]}_{str(frame_count)}.png"
                save_path_edge_template = f"./templates/cropped_templates/edge/{'webcam' if videoPath is None else videoPath[15:-4]}_{str(frame_count)}.png"

                # print(save_path)
                cv2.imwrite(save_path_frame, foreground_morph_dilate)
                if spliced_foreground_frame.shape != frame.shape:
                    cv2.imwrite(save_path_foreground_template, foreground_template)
                    cv2.imwrite(save_path_edge_template, edge_template)
                frame_count += 1

            # controls
            key = cv2.waitKey(25) & 0xFF
            # Press Q on keyboard to exit
            if key == ord('q'):
                break
            # save frame as template
            elif key == ord('1'):
                saveTemplate = not saveTemplate
                print(f'saveTemplate: {saveTemplate}')
        # Break the loop
        else: 
            break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()

def templateDictionaryToDataframe(template_dictionary):
    template_tuple_list = []
    template_types = ['upright', 'falling', 'sitting', 'lying']
    for template_type in template_types:
        for entry in template_dictionary[template_type]:
            template_tuple_list.append(entry)
    tempalte_dataframe = pd.DataFrame(template_tuple_list, columns = ['class', 'image'])
    return tempalte_dataframe

def userInterface(database):
    global templates
    while True:
        print("""
        Command:(button)              Description:
        view_video:(1)                Displays available videos in 'fall_samples' with computer vision.
        view_webcam:(2)               Displays connected webcam with computer vision
        modify templates:(3)          Allows user to modify templates that exist in the database.
        compare_template:(4)          Demonstrates comparing a template to a frame.
        database:(5)                  Access Database UI.
        load templates from file:(6)  Loads templates from files manually instead of the database. 
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
                    display(videoPath = available_videos[selection], saveTemplate = save_templates, checkTemplate = check_templates)
                else:
                    print("Incorrect selection.")
            except ValueError as error:
                print(error)
        elif command == '2':
            display()
        elif command == '3':
            template_modifier = TemplateModifier.template_modifier(templates)
            template_modifier.crop_template()
        elif command == '4': 
            k = None
            comparison_frame = database.access_image_by_id(12)
            # showImage(comparison_frame)
            start = timer()
            classification = CompareTemplates.classifyKnn(comparison_frame, templates['edge'], k)
            end = timer()
            print(f"Compared in {end-start} seconds.")
            print(f'classification: {classification}')

            start = timer()
            classification = CompareTemplates.classifyKnn(comparison_frame, templates['foreground'], k)
            end = timer()
            print(f"Compared in {end-start} seconds.")
            print(f'classification: {classification}')
        elif command == '5':
            DatabaseFunctionality.userInterface(database)
            loadTemplates(database)
        elif command == '6':
            templates = loadLocalTemplates()
        elif command == 'q':
            break
        else:
            print("incorrect command.")
# loads templates from files saved on local machine. (NOTE: Folders must be premade and organized to use)
def loadLocalTemplates():
    local_templates = {"edge": {}, "foreground": {}}

    template_characteristics = ["edge", "foreground"]
    template_types = ["upright", "falling", "sitting", "lying"]
    
    for template_characteristic in template_characteristics:
        for template_type in template_types:
            path = f"./templates/cropped_templates/{template_characteristic}/{template_type}/"
            for (_, _, filenames) in walk(path):
                images = []
                for filename in filenames:
                    file_path = f"{path}{filename}"
                    byte_str = imagePathToByteString(file_path)
                    image = byteStringToImage(byte_str)
                    images.append(image)
                local_templates[template_characteristic][template_type] = images
    return local_templates

# loads templates from database
def loadTemplates(database):
    global templates

    templates = database.load_template_dictionary()
    if templates is None:
        print("Loading files locally")
        templates = loadLocalTemplates()

def main():
    print('Starting FDSystem')
    database = DatabaseFunctionality.FDSDatabase(LOCAL_DATABASE_NAME, LOCAL_DATABASE_PASSWORD)
    database.connect()
    loadTemplates(database)
    userInterface(database)

if __name__ == '__main__':
    main()
    