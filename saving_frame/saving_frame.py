import cv2
import pickle


def grab_frame(cap):
    ret, frame = cap.read()
    return frame



def main():
    #carico il video
    video = cv2.VideoCapture('CVvideo_prosp.MP4')

    i = 0
    frame_counter = 0
    while(True):
        if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):       # se il frame corrente e l'ultimo frame sono uguali
            break        
        # if video.get(cv2.CAP_PROP_POS_FRAMES) == 24:       # se il frame corrente e l'ultimo frame sono uguali
        #     break                                 
        
    
        if i % 3 == 0:
            frame = grab_frame(video)
            i += 1
            continue

        print(video.get(cv2.CAP_PROP_POS_FRAMES))

        frame = grab_frame(video)

        frame = cv2.resize(frame,(1280, 720), fx=0 ,fy=0) 
         
        filename = "frame"+ str(frame_counter) +".png"  

        cv2.imwrite(filename, frame)
        i += 1
        frame_counter += 1

        # cv2.imshow("Result",frame)

        if cv2.waitKey(60) & 0xFF == ord('q'):                    
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
