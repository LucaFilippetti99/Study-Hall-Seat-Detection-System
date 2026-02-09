import cv2
import pickle

try:
    with open("VideoPickle", "rb") as f:        
        posList = pickle.load(f)
except:
        posList = []

# center = [550,300]

def grab_frame(cap):
    """
    Method to grab a frame from the camera
    :param cap: the VideoCapture object
    :return: the captured image
    """
    ret, frame = cap.read()
    return frame


def mouse_click(events, x, y, flags, params):
    side = 80

    if events == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if (x1 - int(side/2)) < x < (x1 + int(side/2)) and (y1 - int(side/2)) < y < (y1 + int(side/2)):
                posList.pop(i)


    with open("VideoPickle", "wb") as f:
        pickle.dump(posList, f)

def main():
    video = cv2.VideoCapture('CVvideo_aula.MOV')
    side = 80

    while video.isOpened():
        frame = grab_frame(video)
        frame = cv2.resize(frame, (1280, 720), fx=0, fy=0)

        for pos in posList:
            cv2.rectangle(frame, (pos[0] - int(side/2), pos[1] - int(side/2)),
                          (pos[0] + int(side/2), pos[1] + int(side/2)), (255, 0, 255), 2)
        cv2.imshow("Image", frame)
        cv2.setMouseCallback("Image", mouse_click)
        if (cv2.waitKey(1) & 0xFF == ord('q')):  
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)