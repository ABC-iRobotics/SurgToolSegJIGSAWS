# Create new folder with correct numbering of frames (video_to_frame)
# miden 50. frame labeled majd trackel megrpóbálni

import cv2
import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy import ndimage

def tracking(data_path,count):

    
    lk_params = dict( winSize  = (30,30),
                  maxLevel = 6,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    

    file_names = list((data_path).glob('*'))
    prev  = cv2.imread(str(file_names[0]))
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    file_name_masks = list((Path('mask/Suturing/B00'+str(count)+'/')).glob('*'))
    prev_mask = cv2.imread(str(file_name_masks[0]))
    prev_mask = cv2.cvtColor(prev_mask, cv2.COLOR_BGR2GRAY)
    _,prev_mask= cv2.threshold(prev_mask,1,255,cv2.THRESH_BINARY)

    pts0 = np.zeros([1,2])

    for k in range(0, prev_mask.shape[0]):
        for j in range(0, prev_mask.shape[1]):
            if prev_mask[k, j] != 0:
                pts0 = np.append(pts0,[[j,k]], axis = 0)
    
    k = 1
    mask_i = 1

    for file_name in tqdm(list((data_path).glob('*')), desc = 'Tracking'): 
        
        founded = np.zeros(prev.shape)
        f = np.zeros(prev.shape)

        if (k % 30) == 0:
            prev_mask = cv2.imread(str(file_name_masks[mask_i]))
            prev_mask = cv2.cvtColor(prev_mask, cv2.COLOR_BGR2GRAY)
            _,prev_mask= cv2.threshold(prev_mask,1,255,cv2.THRESH_BINARY)

            pts0 = np.zeros([1,2])

            for l in range(0, prev_mask.shape[0]):
                for j in range(0, prev_mask.shape[1]):
                    if prev_mask[l, j] != 0:
                        pts0 = np.append(pts0,[[j,l]], axis = 0)

            mask_i += 1
            founded = prev_mask
        
        next = cv2.imread(str(file_name))
        next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
        
        pts0 = pts0.astype('float32')
        pts1, st, err = cv2.calcOpticalFlowPyrLK(prev, next, pts0, None, **lk_params)

        if pts1 is not None:
            good_new = pts1[(st==1).ravel(),:]
            good_old = pts0[(st==1).ravel(),:]

        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            founded = cv2.circle(founded.astype('uint8'),(int(a),int(b)),1,(255,0,0),-1)
        
        # POST PROCESS
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(founded.astype('uint8'), connectivity=8)
        sizes = stats[1:, -1]; nb_components = nb_components - 1

        img2 = np.zeros((output.shape))
        sizes_sum = 0
        for i in range(0, nb_components):
            if sizes[i] >= 1000:  # 500
                img2[output == i + 1] = 255

        founded = img2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
        founded = cv2.morphologyEx(founded, cv2.MORPH_CLOSE, kernel)

        padded = np.concatenate((founded,np.ones((founded.shape[0],1))), axis = 1)
        padded = np.insert(padded,0,np.ones(founded.shape[0]),axis = 1)
        padded = ndimage.binary_fill_holes(padded).astype(int)
        founded = padded[0:padded.shape[0], 1:padded.shape[1]-1]


        contour,hier = cv2.findContours(founded.astype('uint8')*255,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        cont = np.zeros(founded.shape)
        for cnt in contour:
            cv2.drawContours(cont,[cnt],0,255,-1)
       
        founded = cv2.medianBlur(cont.astype('uint8'), 11)

        prev = next
        pts0 = good_new.reshape(-1,1,2)

        k += 1

        cv2.imwrite(str('ground_truth/Suturing/B00'+str(count)+'/'+ (file_name.stem + '.jpg')),np.uint8(founded),[cv2.IMWRITE_JPEG_QUALITY, 100])

if __name__ == '__main__':
    for count in range(2,3):
        tracking(Path('data/JIGSAWS/Suturing/B00'+str(count)+'/'), count)

