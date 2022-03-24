import cv2
import argparse


def getFrame(sec,task_type, type_of_preprocess):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        if type_of_preprocess == 'Nothing':
            if count < 10:
                cv2.imwrite("data/test_JIGSAWS/"+str(task_type)+ "/"+str(type_of_preprocess)+"/frame000"+str(count)+".jpg", image)
            if count >= 10 and count < 100 :
                cv2.imwrite("data/test_JIGSAWS/"+str(task_type)+"/"+str(type_of_preprocess)+"/frame00"+str(count)+".jpg", image)
            if count >= 100 and count < 1000:
                cv2.imwrite("data/test_JIGSAWS/"+str(task_type)+"/"+str(type_of_preprocess)+"/frame0"+str(count)+".jpg", image)
            if count >= 1000:
                cv2.imwrite("data/test_JIGSAWS/"+str(task_type)+"/"+str(type_of_preprocess)+"/frame"+str(count)+".jpg", image)
        else:
            img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
            if count < 10:
                cv2.imwrite("data/test_JIGSAWS/"+str(task_type)+ "/"+str(type_of_preprocess)+"/frame000"+str(count)+".jpg", img_lab)
            if count >= 10 and count < 100 :
                cv2.imwrite("data/test_JIGSAWS/"+str(task_type)+"/"+str(type_of_preprocess)+"/frame00"+str(count)+".jpg", img_lab)
            if count >= 100 and count < 1000:
                cv2.imwrite("data/test_JIGSAWS/"+str(task_type)+"/"+str(type_of_preprocess)+"/frame0"+str(count)+".jpg", img_lab)
            if count >= 1000:
                cv2.imwrite("data/test_JIGSAWS/"+str(task_type)+"/"+str(type_of_preprocess)+"/frame"+str(count)+".jpg", img_lab)

        return hasFrames

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path_of_video', type=str, default= '', help='path to vieo folder')
    arg('--task_type', type=str, default='Suturing', help='type of surgical task',choices=['Knot_tying', 'Suturing', 'Needle_passing'])
    arg('--type_of_preprocess', type=str, default='Nothing', help='preprocess of the data for different models',choices=['Nothing', 'Lab'])

    
    args = parser.parse_args()
       
    
    vidcap = cv2.VideoCapture(args.path_of_video)

    framespersecond= int(vidcap.get(cv2.CAP_PROP_FPS))
    #print(framespersecond) ---> 30 frame per second

    sec = 0
    frameRate = 0.03           #//it will capture image in each 0.03 second // 30HZ videos are the jigsaw // 30 frames per second
    count=0
    success = getFrame(sec,args.task_type, args.type_of_preprocess)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec, args.task_type, args.type_of_preprocess)

