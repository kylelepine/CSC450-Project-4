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

def test0():
    print('test0()')
    image_bytes = readimage('test_image.jpeg')
    img = byteStr_to_image(image_bytes)
    print('OpenCV:\n', img)
    show_image(img)

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

def test2():
    cap = cv2.VideoCapture(0)
    previous_frame = None

    if not cap.isOpened():
        raise  IOError("Cannot open webcam")

    while(True):
        ret, frame = cap.read()
        if ret == True:
            display_img = frame
            if previous_frame is not None:
                display_img = img_diff(frame, previous_frame)
            
            cv2.imshow('display', display_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        previous_frame = frame
    cap.release()
    cv2.destroyAllWindows()

def main():
    print('main()')

    # test0()
    # test1()
    test2()

if __name__ == '__main__':
    main()
    