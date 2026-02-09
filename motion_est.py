import cv2
import pickle
import numpy as np

# Video feed
cap = cv2.VideoCapture('yolo_dnn/CVvideo_aula.MOV')

# with open('CarParkPos', 'rb') as f:                         # apre il file del pickle
#     posList = pickle.load(f)                                # carica il file del pickle

width, height = 107, 48


# def checkParkingSpace(imgPro):                              # funzione che verifica se c'è spazio oppure no
#     spaceCounter = 0

#     for pos in posList:                                     # per ogni posizione della lista
#         x, y = pos

#         imgCrop = imgPro[y:y + height, x:x + width]         # prende il quadrato a partire dalla posizione
#         # cv2.imshow(str(x * y), imgCrop)
#         count = cv2.countNonZero(imgCrop)                   # Ritorna il numero di pixel che non sono zero


#         if count < 900:                                     # non occupata
#             color = (0, 255, 0) 
#             thickness = 5
#             spaceCounter += 1
#         else:                                               # occupata
#             color = (0, 0, 255)
#             thickness = 2

#         cv2.rectangle(frame, pos, (pos[0] + width, pos[1] + height), color, thickness)

pre_frame_gray = np.zeros((1280, 720))     
while True:
    # CAP_PROP_FRAME_COUNT -> Number of frames in the video file.
    # CAP_PROP_POS_FRAMES  -> 0-based index of the frame to be decoded/captured next.

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):       # se il frame corrente e l'ultimo frame sono uguali
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)                                         # setto il frame corrente a 0 (ricomincio da capo)

    success, next_frame = cap.read() 
    next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    next_frame_gray = cv2.fastNlMeansDenoising(next_frame_gray, None, 20, 7, 21)

    # Dividing the image by its blurred version is a background removal method. It whitens the background. !!!
    # ho visto divisioni che fanno anche uso del risultato dell'operatore morfologico come denominatore

    # frame_gray_blurred = cv2.GaussianBlur(next_frame_gray, (35, 35), 2)

    if  not pre_frame_gray.any():
        pre_frame_gray = next_frame_gray
        continue
    
    hsv = np.zeros_like(next_frame)
    hsv[..., 1] = 255                   # all values in the first column are set to 255 (che cambia da :)
    flow = cv2.calcOpticalFlowFarneback(pre_frame_gray, next_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    

    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    bgr = cv2.resize(bgr,(1280, 720), fx=0 ,fy=0) 
    cv2.imshow('frame2', bgr)

    # frame_diff_th = cv2.threshold(frame_diff, 100, 255 ,cv2.THRESH_BINARY)[1]


    # zero = np.abs(pre_frame - frame).max()
    # if not zero:
    #     print(True)

    # frame_diff = pre_frame - frame

    # print(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
    # print(cap.get(cv2.CAP_PROP_POS_FRAMES))
    # succ, pre_frame = cap.retrieve(np.int((cap.get(cv2.CAP_PROP_POS_FRAMES)- 1)))

    # frame_diff = pre_frame - frame
    # zero = np.abs(pre_frame - frame).max()
    # if not zero:
    #     print(True)

    # frameGray = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)                                 
    # imgBlur = cv2.GaussianBlur(frameGray, (3, 3), 1)                                  
    # imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                      cv2.THRESH_BINARY_INV, 25, 16)

    # imgMedian = cv2.medianBlur(imgThreshold, 5)
    # kernel = np.ones((3, 3), np.uint8)
    # imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    # checkParkingSpace(imgDilate)
    # frameGrayRes = cv2.resize(frame_gray_blurred,(1280, 720), fx=0 ,fy=0)           # resizing perché la finestra di python taglia il video
    # cv2.imshow("Image", frameGrayRes)
    # cv2.imshow("ImageBlur", imgBlur)
    # cv2.imshow("ImageThres", imgMedian)
    # cv2.waitKey(10)
    #  cv2.waitKey(10)
    pre_frame_gray = next_frame_gray
    if(cv2.waitKey(1) & 0xFF == ord('q')):                     # ?
        cap.release()
        cv2.destroyAllWindows()
        break



# adaptiveThreshold è un metodo che, a differenza della sogliatura classica
# e quella du Otsu, permette di effettuare un thresholding locale senza specificare
# manualmente un valore di threshold T (che richiederebbero molti esperimenti manuali; 
# Otsu risolve questo, ma la threshold rimane comunque globale cosa che può portare a problemi
# nel caso in cui l'illuminazione non sia costante)

# adaptive thresholding, sometimes called local thresholding -> considers small neighbors of 
# pixels and then finds an optimal threshold value T for each neighbor.

# The general assumption that underlies all adaptive and local thresholding methods is that smaller 
# regions of an image are more likely to have approximately uniform illumination. This implies that local
# regions of an image will have similar lighting, as opposed to the image as a whole, which may have dramatically 
# different lighting for each region.

# However, choosing the size of the pixel neighborhood for local thresholding is absolutely crucial.