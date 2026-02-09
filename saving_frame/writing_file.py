file1 = open("frame.txt", "w")
frame_counter = 0

for frame_counter in range(4775):
    filename = "frame/frame"+ str(frame_counter) +".png\n"  
    file1.write(filename)
    frame_counter += 1

file1.close() 