import numpy as np
from cv2 import cv2

 # initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
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

def frame_difference(frame, previous_frame):
    frame = frame.astype('int16')
    previous_frame = previous_frame.astype('int16')
    diff_frame = np.zeros_like(frame)

    if frame.shape == previous_frame.shape:
        diff_frame = np.absolute(np.subtract(frame, previous_frame))

    diff_frame = diff_frame.astype('uint8')
    return diff_frame

def transform_binary(frame):
    im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return im_bw

# displays image from file
def image_display_test(image_path):
    print('test0()')
    image_bytes = readimage(image_path)
    img = byteStr_to_image(image_bytes)
    print('OpenCV:\n', img)
    show_image(img)

# Demonstrates calculating frame difference, displays binary image
def original_frame_difference_calculator():
    cap = cv2.VideoCapture(cv2.CAP_DSHOW)
    previous_frame = None

    if not cap.isOpened():
        raise  IOError("Cannot open webcam")

    while(True):
        ret, frame = cap.read()
        if ret == True:
            edges = cv2.Canny(frame, 480, 640)
            display_img = frame
            if previous_frame is not None:
                display_img = frame_difference(frame, previous_frame)
            
            display_img = transform_binary(display_img)
            
            cv2.imshow('edges', edges)
            cv2.imshow('display', display_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        previous_frame = frame
    cap.release()
    cv2.destroyAllWindows()

def video_display(video_path):
    previous_frame = None
    cap = cv2.VideoCapture(video_path)

    target = 0
    counter = 0
    file_count = 0
    while(True):
        if counter == target:
            ret, frame = cap.read()
            if ret == True:
                binary_subtraction_img = frame
                if previous_frame is not None:
                    binary_subtraction_img = frame_difference(frame, previous_frame)
                
                binary_subtraction_img = transform_binary(binary_subtraction_img)

                edges = cv2.Canny(frame, 640, 240)

                cv2.imshow('edges', edges)
                cv2.imshow('mask', binary_subtraction_img)
                cv2.imshow('original', frame)
                cv2.imwrite('./templates/background_subtraction/test_template' + str(file_count) + '.png', binary_subtraction_img)
                cv2.imwrite('./templates/edge_detection/test_template' + str(file_count) + '.png', edges)
                
                previous_frame = frame
                file_count +=1
                counter = 0
        else:
            ret = cap.grab()
            counter += 1
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

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
            kernel = np.ones((10,10),np.uint8)

            foreground_morph = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

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

            # cv2.imwrite('./templates/layered_frames/test_template' + str(file_count) + '.png', layered_frames)
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
	# grab references to the global variables
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
		cv2.rectangle(current_frame, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("current_frame", current_frame)

def crop_template():
    global refPt, cropping
    image_path = 'test_image.jpeg'
    image = cv2.imread(image_path)
    # load the image, clone it, and setup the mouse callback function
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
    # if there are two reference points, then crop the region of interest
    # from teh image and display it
    if len(refPt) == 2:
        roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)
    # close all open windows
    cv2.destroyAllWindows()

def main():
    print('main()')
    # image_display_test('test_image.jpeg')
    
    display('./fall_samples/fall-01-cam0.mp4')
    
    # display()
    # crop_template()
    
if __name__ == '__main__':
    main()
    