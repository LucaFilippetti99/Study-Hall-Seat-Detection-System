import cv2
import pickle


def grab_frame(cap):
    """
    Method to grab a frame from the camera
    :param cap: the VideoCapture object
    :return: the captured image
    """
    ret, frame = cap.read()
    return frame

def main():
    with open('object_detection_classes_coco.txt', 'r') as f:
        classes = f.read().splitlines()

    net7 = cv2.dnn.readNetFromDarknet('yolov7-tiny.cfg', 'yolov7-tiny.weights')
    net = cv2.dnn.readNetFromDarknet('googlecolab/yolov4.cfg', 'googlecolab/yolov4.weights')

    model = cv2.dnn_DetectionModel(net)
    model7 = cv2.dnn_DetectionModel(net7)
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
    model7.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

    video = cv2.VideoCapture('CVvideo_aula.MOV')
    # counter = 0
    # side = 150

    # with open("VideoPickle", "rb") as f:        
    #     posList = pickle.load(f)  

    i = 1
    while(True):
        if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):       # se il frame corrente e l'ultimo frame sono uguali
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)                                         # setto il frame corrente a 0 (ricomincio da capo)

        if i % 2 == 0:
            frame = grab_frame(video)
            i += 1
            continue
        
        frame = grab_frame(video)

        frame = cv2.resize(frame,(1280, 720), fx=0 ,fy=0)           # resizing perché la finestra di python taglia il video

        classIds, scores, boxes = model.detect(frame, confThreshold=0.35, nmsThreshold=0.7)
        classIds7, scores7, boxes7 = model7.detect(frame, confThreshold=0.2, nmsThreshold=0.7)

        for (classId, score, box) in zip(classIds, scores, boxes):
            if classes[classId] == 'person':
                cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                            color=(0, 255, 0), thickness=2)

        for (classId, score, box) in zip(classIds7, scores7, boxes7):
            if classes[classId] == 'person':
                cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                            color=(0, 255, 0), thickness=2)

                # text = '%s: %.2f' % (classes[classId], score)
                # cv2.putText(frame, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                #             color=(0, 255, 0), thickness=1)

            cv2.imshow("Result",frame)

   
        i += 1

        if(cv2.waitKey(24) & 0xFF == ord('q')):                  
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)