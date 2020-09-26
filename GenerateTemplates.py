import numpy as np
from cv2 import cv2

from array import array


def readimage(path):
    print(f'readimage({path})')
    with open(path, 'rb') as f:
        return f.read()

def byteStr_to_image(img_str):
    print('byteStr_to_image')
    decoded = cv2.imdecode(np.frombuffer(img_str, np.uint8), -1)
    return decoded

def show_image(img):
    print('show_image()')
    print(type(img))
    cv2.imshow('Image', img)
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == 27: break
    return cv2.destroyAllWindows()

def img_diff(frame, previous_frame):
    frame = frame.astype('int16')
    previous_frame = previous_frame.astype('int16')
    diff_frame = np.zeros_like(frame)

    if frame.shape == previous_frame.shape:
        diff_frame = np.subtract(frame, previous_frame)
        diff_frame = np.absolute(diff_frame)

        # Manual way -- NOTE: Very slow 
    #     for x in range(len(frame)):
    #         for y in range(len(frame[x])):
    #             for BGR in range(3):
    #                 diff_frame[x,y,BGR] = np.absolute(int(frame[x,y,BGR]) - int(previous_frame[x,y,BGR]))

    # print("Frame: " + str(frame[0,0,:]))
    # print("Prevoius Frame: " + str(previous_frame[0,0,:]))
    # print("Diff Frame: " + str(diff_frame[0,0,:]))
    diff_frame = diff_frame.astype('uint8')
    return diff_frame

def transform_binary(frame):
    # binary_img = np.where((frame < 127), 0, 255)
    # binary_img.astype('uint8')
    # print(binary_img)

    # frame[frame < 127] = 0
    # frame[frame >= 127] = 255

    im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return im_bw

# displays image from file
def test0():
    print('test0()')
    image_bytes = readimage('test_image.jpeg')
    img = byteStr_to_image(image_bytes)
    print('OpenCV:\n', img)
    show_image(img)

# Shows differences of opencv's cvtColor()
def test1():
    print('test1()')
    img = np.zeros([200, 200, 3])
    # print (img)
    img[:,:,0] = np.ones([200,200]) * 255
    img[:,:,1] = np.ones([200,200]) * 255
    img[:,:,2] = np.ones([200,200]) * 0

    for row in img:
        for pixel in row:
            # print(pixel)
            pixel[0] = 255 # B
            pixel[1] = 0   # G
            pixel[2] = 255 # R
            # print(pixel)

    # cv2.imwrite('color_img.jpg', img)

    c = cv2.imread('color_img.jpg', 1)
    c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)

    d = cv2.imread('color_img.jpg', 1)
    d = cv2.cvtColor(c, cv2.COLOR_RGB2BGR)

    e = cv2.imread('color_img.jpg', -1)
    e = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)

    f = cv2.imread('color_img.jpg', -1)
    f = cv2.cvtColor(c, cv2.COLOR_RGB2BGR)

    images = [c, d, e, f]

    for img in images:
        show_image(img)

    print(c)
    print(c.shape)

# Demonstrates calculating frame difference, displays binary image
def test2():
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
                display_img = img_diff(frame, previous_frame)
            
            display_img = transform_binary(display_img)

            
            cv2.imshow('edges', edges)
            cv2.imshow('display', display_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        previous_frame = frame
    cap.release()
    cv2.destroyAllWindows()

# Uses OpenCv's background subtractions
def test3():
    cap = cv2.VideoCapture(cv2.CAP_DSHOW)
    fgbg = cv2.createBackgroundSubtractorMOG2()

    while(True):
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)

        cv2.imshow('frame', fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

def test4():
    cap = cv2.VideoCapture('./fall_samples/fall-01-cam0.mp4')
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows= False)

    target = 10
    counter = 0

    while(True):
        if counter == target:
            ret, frame = cap.read()
            fgmask = fgbg.apply(frame)
            cv2.imshow('mask', fgmask)
            cv2.imshow('original', frame)
            counter = 0
        else:
            ret = cap.grab()
            counter += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def test5():
    previous_frame = None
    cap = cv2.VideoCapture('./fall_samples/fall-01-cam0.mp4')

    target = 0
    counter = 0
    file_count = 0
    while(True):
        if counter == target:
            ret, frame = cap.read()
            if ret == True:
                binary_subtraction_img = frame
                if previous_frame is not None:
                    binary_subtraction_img = img_diff(frame, previous_frame)
                
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

def test6():
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
            gray_filtered = cv2.bilateralFilter(gray, 7, 50, 50)
            
            # Extract the foreground
            foreground = fgbg.apply(gray_filtered)
            
            # Smooth out to get the moving area
            kernel = np.ones((50,50),np.uint8)
            foreground_morph = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
            
            # Using the Canny filter to get contours
            edges = cv2.Canny(gray, 20, 30)

            # Using the Canny filter with different parameters
            edges_high_thresh = cv2.Canny(gray, 60, 120)

            edges_filtered = cv2.Canny(gray_filtered, 60, 120)

            # Crop off the edges out of the moving area
            cropped = (foreground // 255) * edges_filtered

            # Stacking the images to print them together
            # For comparison
            images = np.hstack((gray, edges,  edges_filtered))

            # Display the resulting frame
            cv2.imshow('Frame', images)
            cv2.imshow('original', frame)
            cv2.imshow('cropped', cropped)
            cv2.imshow('morphologyEx', foreground_morph)
            cv2.imshow('background_subtractor', foreground)


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

def main():
    print('main()')

    # test0()
    # test1()
    # test2()
    # test3()
    # test4()
    # test5()
    test6()

if __name__ == '__main__':
    main()
    