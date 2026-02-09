import cv2

frame = cv2.imread('frame0.png')

with open('object_detection_classes_coco.txt', 'r') as f:
    classes = f.read().splitlines()

net = cv2.dnn.readNetFromDarknet('googlecolab/yolov4.cfg', 'googlecolab/yolov4.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True, crop=False)

img = cv2.resize(frame,(1280, 720), fx=0 ,fy=0)

classIds, scores, boxes = model.detect(img, confThreshold=0.5, nmsThreshold=0.6)

for (classId, score, box) in zip(classIds, scores, boxes):
    if classes[classId] == 'person':
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                    color=(0, 255, 0), thickness=2)

        text = '%s: %.2f' % (classes[classId], score)
        cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color=(0, 255, 0), thickness=1)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()