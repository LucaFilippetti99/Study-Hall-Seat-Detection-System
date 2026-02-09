import cv2
import pickle
import json
from copy import deepcopy
import numpy

DRAW_BOX = True
FRAME_ROBUSTNESS = 60
FRAME_THRESHOLD = 20
SIDE = 85

def grab_frame(cap: object) -> numpy.ndarray:
    _, frame = cap.read()
    return frame

def list2dict(lst: list) -> dict:
    res_dct = {lst[i]: 0 for i in range(len(lst))}
    return res_dct

def relative2pixel(x_center: float, y_center: float, w: float, h: float,  image_w=1280, image_h=720):
    w = w * image_w
    h = h * image_h
    x1 = ((2 * x_center * image_w) - w)/2
    y1 = ((2 * y_center * image_h) - h)/2
    return [round(x) for x in [x1, y1, w, h]]

def occupied(center: tuple, pos: tuple, side=SIDE) -> None:
    return center[0] > pos[0] - int(side/2) and center[1] > pos[1] - int(side/2) and center[0] < pos[0] + int(side/2) and center[1] < pos[1] + int(side/2)

def update_dict(posList: list, centers: list, posDict: dict) -> None:
    posList_tmp = deepcopy(posList)

    for center in centers:
        for pos in posList_tmp:
            if occupied(center, pos):
                posList_tmp.remove(pos)
                if posDict[pos] < FRAME_ROBUSTNESS:
                    posDict[pos] += 1
                break
        
    for pos in posList_tmp:
        if posDict[pos] > 0:
            posDict[pos] -= 1


def main() -> None:
    f = open('result.json')        
    result = json.load(f)       
    with open("VideoPickle", "rb") as f:        
        posList = pickle.load(f) 
    total_seats = len(posList)
    
    video = cv2.VideoCapture('CVvideo_aula.MOV')
    posDict = list2dict(posList)
    pre_occ_seats = 0

    frame_counter = 0
    while(True):
        if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):       # se il frame corrente e l'ultimo frame sono uguali
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)  
            frame_counter = 0

        occupied_seats = 0
        if video.get(cv2.CAP_PROP_POS_FRAMES) % 3 == 0:
            frame = grab_frame(video)
            continue

        frame = grab_frame(video)
        frame = cv2.resize(frame,(1280, 720), fx=0 ,fy=0)

        objects = result[frame_counter]['objects']
        person = list(filter(lambda obj : obj['name'] == 'person', objects))
        centers = []
        for obj in person:
            if obj['name'] == 'person':
                box_rel = list(obj['relative_coordinates'].values())
                box = relative2pixel(*box_rel)

                center = (box[0] + box[2]//2), (box[1] + box[3]//2)
                centers.append(center)
                if DRAW_BOX:
                    cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color=(139, 0, 139), thickness=1)   
                    text = '%s: %.2f' % (obj['name'], obj['confidence'])
                    cv2.putText(frame, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(139, 0, 139), thickness=1)
                    cv2.circle(frame, center, radius=2, color=(0, 0, 255), thickness=1)

        update_dict(posList, centers, posDict)

        for pos in posDict:
            if video.get(cv2.CAP_PROP_POS_FRAMES) < FRAME_ROBUSTNESS:
                color=(128, 128, 128)         
                thickness = 1
            else:
                if posDict[pos] < FRAME_THRESHOLD:
                    color = (0, 255, 0)         
                    thickness = 2
                else:
                    occupied_seats += 1
                    color = (0, 0, 255)         
                    thickness = 3
            cv2.rectangle(frame, (pos[0] - int(SIDE/ 2), pos[1] - int(SIDE/ 2)),
                (pos[0] + int(SIDE/ 2), pos[1] + int(SIDE/ 2)), color, thickness)

        frame_counter += 1
        if occupied_seats != pre_occ_seats:
            print(f"Seats available : {total_seats - occupied_seats}/{total_seats}")

        cv2.imshow("Result",frame)

        pre_occ_seats = occupied_seats
        if cv2.waitKey(35) & 0xFF == ord('q'):                    
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
