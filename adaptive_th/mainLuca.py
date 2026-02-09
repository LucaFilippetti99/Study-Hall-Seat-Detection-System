import cv2
import pickle
import numpy as np


def grab_frame(cap):
    """
    Method to grab a frame from the camera
    :param cap: the VideoCapture object
    :return: the captured image
    """
    ret, frame = cap.read()
    return frame

def main():
    #carico il video
    d_rect_x = 170
    d_rect_y = 170
    video = cv2.VideoCapture('CVvideo_ver.mp4')
    counter = 0
    side = 150

    with open("VideoPickle", "rb") as f:        
        posList = pickle.load(f)  

    while(True):
        if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):       # se il frame corrente e l'ultimo frame sono uguali
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)                                         # setto il frame corrente a 0 (ricomincio da capo)

        frame = grab_frame(video)

        frame = cv2.resize(frame,(1280, 720), fx=0 ,fy=0)           # resizing perché la finestra di python taglia il video
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        imgBlur = cv2.GaussianBlur(frameGray, (7, 7), 1)    

        # se = cv2.getStructuringElement(cv2.MORPH_RECT , (3,3))
        # bg = cv2.morphologyEx(imgBlur, cv2.MORPH_DILATE, se)
        # divsion_result =cv2.divide(frameGray, bg, scale=255)   

        imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 45, 3) 
        
        #cv2.rectangle(imgThreshold, (150, 350), (320, 520), (255, 0, 255), 2)
        #cv2.rectangle(imgThreshold, (510, 530), (510 + d_rect_x, 530 + d_rect_y), (255, 0, 255), 2)
        #cv2.rectangle(imgThreshold, (827, 287), (827 + d_rect_x, 287 + d_rect_y), (255, 0, 255), 2)
        #cv2.rectangle(imgThreshold, (460, 100), (460 + d_rect_x, 100 + d_rect_y), (255, 0, 255), 2)
        #area1 = imgThreshold[530:530 + d_rect_y , 510: 510 + d_rect_x]
        #count1 = cv2.countNonZero(area1)
        #print("count1",count1)
        #cv2.imshow("video", area1)

        for pos in posList:
            x1, y1 = pos
#provare a dividere il quadrato in celle e verificare poi il count sulle singole celle e vedere la maggioranza
        
            imgCrop = imgThreshold[(y1 - int(side/ 2)): (y1 + int(side/ 2)),
                      (x1 - int(side/ 2)): (x1 + int(side/ 2))]

            se_erode = cv2.getStructuringElement(cv2.MORPH_RECT , (5, 5))
            imgCrop = cv2.morphologyEx(imgCrop, cv2.MORPH_ERODE, se_erode)
            se_dilate = cv2.getStructuringElement(cv2.MORPH_RECT , (2, 2))
            imgCrop = cv2.morphologyEx(imgCrop, cv2.MORPH_DILATE, se_dilate)


            cv2.imshow(str(x1),imgCrop)
            count = cv2.countNonZero(imgCrop)

            print(count)

            if count < 4000:
                color = (0, 255, 0)
                thickness = 4
                counter += 1
            else:
                color = (0, 0, 255)
                thickness = 2

            cv2.rectangle(frame, (pos[0] - int(side/ 2), pos[1] - int(side/ 2)),
                          (pos[0] + int(side/ 2), pos[1] + int(side/ 2)), color, thickness)

            cv2.imshow("Result",frame)

        if(cv2.waitKey(24) & 0xFF == ord('q')):                     # ?
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)