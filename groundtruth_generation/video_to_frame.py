import cv2
       
       
vidcap = cv2.VideoCapture('')

framespersecond= int(vidcap.get(cv2.CAP_PROP_FPS))
#print(framespersecond) ---> 30 frame per second

def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        if count < 10:
            cv2.imwrite("data/Knot_tying/B001/frame000"+str(count)+".jpg", image)
        if count >= 10 and count < 100 :
            cv2.imwrite("data/Knot_tying/B001//frame00"+str(count)+".jpg", image)
        if count >= 100 and count < 1000:
            cv2.imwrite("data/Knot_tying/B001//frame0"+str(count)+".jpg", image)
        if count >= 1000:
            cv2.imwrite("data/Knot_tying/B001//frame"+str(count)+".jpg", image)
        return hasFrames
sec = 0
frameRate = 0.03 #//it will capture image in each 0.03 second // 30HZ videos are the jigsaw // 30 frames per second
count=0
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)


