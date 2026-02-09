import cv2
import numpy as np

def main():
    #carico il video
    d_rect_x = 170
    d_rect_y = 170
    video = cv2.VideoCapture('yolo_dnn/CVvideo_aula.MOV')


    while(True):
        if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):       # se il frame corrente e l'ultimo frame sono uguali
            break        



        success, frame = video.read()
        
        frame = cv2.resize(frame,(1280, 720), fx=0 ,fy=0)          


        cv2.imshow("window", frame)      

        if(cv2.waitKey(24) & 0xFF == ord('q')):                    
            break

    video.release()
    cv2.destroyAllWindows()
        

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)