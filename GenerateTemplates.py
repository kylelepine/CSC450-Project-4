import numpy as np
from cv2 import cv2


refPt = []
cropping = False
current_frame = None

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

def display(video_path = None):
    file_count = 0
    
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
            kernel_close = np.ones((9,9),np.uint8)
            kernel_open = np.ones((10,10),np.uint8)

            foreground_morph = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel_close)
            foreground_morph = cv2.morphologyEx(foreground_morph, cv2.MORPH_OPEN, kernel_open)

            edges_filtered = cv2.Canny(gray_filtered, 60, 120)

            # Crop off the edges out of the moving area
            cropped_edges = (foreground_morph // 255) * edges_filtered

            #EXPERIMENTAl
            layered_frames = np.add(cropped_edges, foreground_morph)

            # Stacking the images to print them together
            # For comparison
            frames_normal = np.hstack(( gray,  gray_filtered))
            frames_edges = np.hstack((edges_filtered,  cropped_edges))

            # Display the resulting frame
            cv2.imshow('Normal Frames', frames_normal)
            cv2.imshow('Frames Edges', frames_edges)
            cv2.imshow('Foreground Frames', foreground)
            cv2.imshow('layered Frames', layered_frames)
            cv2.imshow('MorphologyEx', foreground_morph)

            cv2.imwrite('./templates/layered_frames/test_template' + str(file_count) + '.png', layered_frames)
            file_count += 1

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        # Break the loop
        else: 
            break
    
    # When everything done, release the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()

def click_and_crop(event, x, y, flags, param):
	global refPt, cropping, current_frame
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
		# draw a rectangle around the region of interest
        # TODO: Display rectangle in current frame without it affecting the 'original_frame'.
		cv2.rectangle(current_frame, refPt[0], refPt[1], (0, 255, 0),2)
		# cv2.imshow("current_frame", current_frame)

def crop_template():
    global refPt, cropping, current_frame
    template_types = ['upright', 'falling', 'sitting', 'lying']
    template_type = None
    file_count = 0
    roi = np.array([])

    cv2.namedWindow("current_frame")
    cv2.setMouseCallback("current_frame", click_and_crop)
    
    image_path = f'./templates/layered_frames/test_template{file_count}.png'
    current_frame = cv2.imread(image_path)
    original_frame = np.copy(current_frame)

    print("""
            Click the key to save the 'ROI' as a template
            Option:    Template Type:
            1----------Upright
            2----------Falling
            3----------Sitting Down
            4----------Lying Down      
            """)

    while True:
        if cropping == True:
            current_frame = original_frame

        cv2.imshow("current_frame", current_frame)    
        cv2.imshow('original_frame', original_frame)    
        
        if len(refPt) == 2:
            # refPt = [[x1, y1], [x2, y2]]
            x1 = refPt[0][0]
            x2 = refPt[1][0]
            y1 = refPt[0][1]
            y2 = refPt[1][1]
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
                current_frame = cv2.imread(image_path)
                original_frame = current_frame.copy()

        elif key == ord(']'):
            # TODO: fix to where it doesn't access files out of range
            file_count += 1
            image_path = f'./templates/layered_frames/test_template{file_count}.png'
            current_frame = cv2.imread(image_path)
            original_frame = current_frame.copy()
        
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
    # image_display_test('test_image.jpeg')
    display('./fall_samples/fall-01-cam0.mp4')
    # display()
    crop_template()
    
if __name__ == '__main__':
    main()
    