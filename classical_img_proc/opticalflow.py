import cv2
import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
import skimage.morphology, skimage.data
from scipy import ndimage
import os

def tracking(data_path, data_type):

    
    lk_params = dict( winSize  = (30,30),
                  maxLevel = 6,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    file_names = list((data_path/'Cropped'/data_type).glob('*'))
    prev  = cv2.imread(str(file_names[3]))
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    if data_type == 'MICCAI':
        prev = prev[10:prev.shape[0]-9, 10:prev.shape[1]-9]

    file_names = list((data_path/'optical_flow_hsv'/data_type).glob('*'))
    prev_mask = cv2.imread(str(file_names[3]))
    prev_mask = cv2.cvtColor(prev_mask, cv2.COLOR_BGR2GRAY)
    _,prev_mask= cv2.threshold(prev_mask,1,255,cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    prev_mask = cv2.dilate(prev_mask.astype('uint8'),  kernel)

    pts0 = np.zeros([1,2])

   
    for k in range(0, prev_mask.shape[0]):
        for j in range(0, prev_mask.shape[1]):
            if prev_mask[k, j] != 0:
                pts0 = np.append(pts0,[[j,k]], axis = 0)
    
    k = 0
    
    filelist = list((data_path/'track'/data_type).glob('*'))
    for f in filelist:
        os.remove(f)

    for file_name in tqdm(list((data_path/'Cropped'/data_type).glob('*')), desc = 'Tracking'): 

        founded = np.zeros(prev.shape)
        f = np.zeros(prev.shape)

        next = cv2.imread(str(file_name))
        next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)

        next_mask = cv2.imread(str(file_names[k]))
        next_mask = cv2.cvtColor(next_mask, cv2.COLOR_BGR2GRAY)
        _,next_mask= cv2.threshold(next_mask,1,255,cv2.THRESH_BINARY)

        if data_type == 'MICCAI':
            next = next[10:next.shape[0]-9, 10:next.shape[1]-9]
        
        pts0 = pts0.astype('float32')
        pts1, st, err = cv2.calcOpticalFlowPyrLK(prev, next, pts0, None, **lk_params)

        if pts1 is not None:
            good_new = pts1[(st==1).ravel(),:]
            good_old = pts0[(st==1).ravel(),:]

        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            founded = cv2.circle(founded.astype('uint8'),(int(a),int(b)),1,(255,0,0),-1)
        

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(founded, 8 )

        ind = (-stats[:,4]).argsort()
        if num_labels > 2:
            next_tool1 = np.reshape((labels == ind[1]),prev.shape)
            next_tool2 = np.reshape((labels == ind[2]),prev.shape)
        else:
            next_tool1 = np.zeros(prev.shape)
            next_tool2 = np.zeros(prev.shape)
        
        f = next_mask.astype('uint8')> 0
        if np.sum(np.sum(cv2.bitwise_and((next_mask>0).astype('uint8'), (next_tool1>0).astype('uint8')))) < (np.sum(np.sum((next_tool1>0).astype('uint8'))) * 0.5):
            f = ((next_tool1>0).astype('uint8')+(f>0).astype('uint8')) > 0

        if np.sum(np.sum(cv2.bitwise_and((next_mask>0).astype('uint8'), (next_tool2>0).astype('uint8')))) < (np.sum(np.sum((next_tool2>0).astype('uint8'))) * 0.5):
            f = ((next_tool2>0).astype('uint8')+(f>0).astype('uint8')) > 0


        prev = next
        pts0 = good_new.reshape(-1,1,2)

        k += 1

        cv2.imwrite(str(data_path/'track'/ data_type / (file_name.stem + '.jpg')),f.astype('uint8')*255,[cv2.IMWRITE_JPEG_QUALITY, 100])


def optical_flow_with_hsv(data_path,data_type):

    file_names = list((data_path/'Cropped'/data_type).glob('*'))
    prev = cv2.imread(str(file_names[0]))

    if data_type == 'MICCAI':
        prev = prev[10:prev.shape[0]-9, 10:prev.shape[1]-9, :]
        prev = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
    else:
        prev = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)

    filelist = list((data_path/'optical_flow_hsv'/data_type).glob('*'))
    for f in filelist:
        os.remove(f)

    for file_name in tqdm(list((data_path/'Cropped'/data_type).glob('*')), desc = 'Optical Flow'):

        next = cv2.imread(str(file_name))
        
        if data_type == 'MICCAI':
            next = next[10:next.shape[0]-9, 10:next.shape[1]-9, :]

        # CONTRAST ENHANCEMENT
        r_image, g_image, b_image = cv2.split(next)

        r_image_eq = cv2.equalizeHist(r_image)
        g_image_eq = cv2.equalizeHist(g_image)
        b_image_eq = cv2.equalizeHist(b_image)

        image_eq = cv2.merge((r_image_eq, g_image_eq, b_image_eq))


        # INPAINTING
        img_hsv = cv2.cvtColor(image_eq, cv2.COLOR_BGR2HSV)
        h_image, s_image, v_image = cv2.split(img_hsv)

        min_hue = 0
        min_sat = 0
        min_val = 220
        max_hue = 180
        max_sat = 255
        max_val = 255

        mask = cv2.inRange(img_hsv, (min_hue, min_sat, min_val), (max_hue, max_sat, max_val))
        flags = cv2.INPAINT_TELEA  
        inpainted_img  = cv2.inpaint(next, mask, 1, flags=flags)

        r_image, g_image, b_image = cv2.split(inpainted_img)
        img_hsv = cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2HSV)
        h_image, s_image, v_image = cv2.split(img_hsv)

        _,th = cv2.threshold(s_image,30, 255,cv2.THRESH_BINARY_INV)

        next = cv2.cvtColor(image_eq, cv2.COLOR_RGB2GRAY)


        # OPTICAL FLOW
        flow = cv2.calcOpticalFlowFarneback(prev,next, None, 0, 1, 5, 3, 7, 1.5, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        norm = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        _,bin = cv2.threshold(mag,2,255,cv2.THRESH_BINARY)

      
        # MORPHOLOGICAL OPERATIONS 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(20,20))
        cl = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
        
        final = cl*th

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
        final = cv2.dilate(final, kernel)

        prev = next 

        #lines = line_fitting (img2)

        # write to file
        cv2.imwrite(str(data_path/'optical_flow_hsv'/ data_type / (file_name.stem + '.jpg')),final,[cv2.IMWRITE_JPEG_QUALITY, 100])
       

def postprocess(data_path, data_type):

    filelist = list((data_path/'postprocess'/data_type).glob('*'))
    for f in filelist:
        os.remove( f)

    for file_name in tqdm(list((data_path/'track'/data_type).glob('*')), desc = 'Postprocess'):
        
        img = cv2.imread(str(file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _,img= cv2.threshold(img,1,255,cv2.THRESH_BINARY)

        # REMOVE SMALL OBJECTS
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img.astype('uint8'), connectivity=8)
        sizes = stats[1:, -1]; nb_components = nb_components - 1

        img2 = np.zeros((output.shape))
        sizes_sum = 0
        for i in range(0, nb_components):
            if sizes[i] >= 400:  # 500
                img2[output == i + 1] = 255

        # INTERPOLATION
        # downsample
        scale_percent = 20 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img2, dim, interpolation = cv2.INTER_LINEAR )
        # upsample
        scale_percent = 500 # percent of original size
        width = int(resized.shape[1] * scale_percent / 100)
        height = int(resized.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(resized, dim, interpolation = cv2.INTER_LINEAR )


        # MEDIAN FILTER
        contour,hier = cv2.findContours(resized.astype('uint8')*255,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        cont = np.zeros(img.shape)
        for cnt in contour:
            cv2.drawContours(cont,[cnt],0,255,-1)
       
        median = cv2.medianBlur(cont.astype('uint8'), 15) 

        # MORPHOLOGICAL OPERATION
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
        closing = cv2.morphologyEx(median, cv2.MORPH_CLOSE, kernel)

        # FILLING HOLES
        padded = np.pad(closing,(0, 1), 'constant', constant_values=(1, 1))
        padded = ndimage.binary_fill_holes(padded).astype(int)
        padded = padded[0:padded.shape[0]-1, 0:padded.shape[1]-1]

        padded = np.insert(padded,0,np.ones(padded.shape[0]),axis = 1)
        padded = ndimage.binary_fill_holes(padded).astype(int)
        padded = padded[0:padded.shape[0], 1:padded.shape[1]]


        # REMOVE SMALL OBJECTS
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(padded.astype('uint8'), connectivity=8)
        sizes = stats[1:, -1]; nb_components = nb_components - 1

        img2 = np.zeros((output.shape))
        sizes_sum = 0
        for i in range(0, nb_components):
            if sizes[i] >= 1000:  # 500
                img2[output == i + 1] = 255


        cv2.imwrite(str(data_path/'postprocess'/ data_type / (file_name.stem + '.png')),img2,[cv2.IMWRITE_JPEG_QUALITY, 100])

def crop (data_path, original_height, original_width, startx, starty, cropped_train_path):

    filelist = list(cropped_train_path.glob('*'))
    for f in filelist:
        os.remove(f)
    
    for file_name in tqdm(list((data_path).glob('*')), desc = 'Cropping'):
        img = cv2.imread(str(file_name))
        img = img[startx: original_height,starty:original_width]
        cv2.imwrite(str(cropped_train_path/ (file_name.stem +'.jpg')), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])

def evaluate(pred_path, gt_path,startx, original_height,starty,original_width, data_type):

    acc = []
    for file_name in tqdm(list((pred_path).glob('*')), desc = 'Evaluate'):

        pred =  cv2.imread(str(file_name))
        gt = cv2.imread(str(gt_path/file_name.name))
        gt = gt[startx: original_height,starty:original_width]

        if data_type == 'MICCAI':
            gt = gt[10:gt.shape[0]-9, 10:gt.shape[1]-9, :]
            #gt = cv2.cvtColor(gt,cv2.COLOR_BGR2GRAY)
        
        pred = (pred>0).flatten()
        gt = (gt>0).flatten()

        tp = (gt*pred).sum()
        tn = ((pred==0) * (gt ==0)).sum()

        acc +=[ (tp+tn) / (pred.shape)]

    return np.mean(acc)

def evaluate_dice(pred_path, gt_path,startx, original_height,starty,original_width, data_type):

    dice = []
    for file_name in tqdm(list((pred_path).glob('*')), desc = 'Evaluate'):

        pred =  cv2.imread(str(file_name))
        gt = cv2.imread(str(gt_path/file_name.name))
        gt = gt[startx: original_height,starty:original_width]

        if data_type == 'MICCAI':
            gt = gt[10:gt.shape[0]-9, 10:gt.shape[1]-9, :]
            #gt = cv2.cvtColor(gt,cv2.COLOR_BGR2GRAY)
        
        pred = (pred>0).flatten()
        gt = (gt>0).flatten()

        dice +=[ (2 * (gt * pred).sum() + 1e-15) / (gt.sum() + pred.sum() + 1e-15)]

    return np.mean(dice)

def evaluate_jaccard(pred_path, gt_path,startx, original_height,starty,original_width, data_type):

    jaccard = []
    for file_name in tqdm(list((pred_path).glob('*')), desc = 'Evaluate'):

        pred =  cv2.imread(str(file_name))
        gt = cv2.imread(str(gt_path/file_name.name))
        gt = gt[startx: original_height,starty:original_width]

        if data_type == 'MICCAI':
            gt = gt[10:gt.shape[0]-9, 10:gt.shape[1]-9, :]
            #gt = cv2.cvtColor(gt,cv2.COLOR_BGR2GRAY)
        
        pred = (pred>0).flatten()
        gt = (gt>0).flatten()

        intersection = (gt * pred).sum()
        union = gt.sum() + pred.sum() - intersection
        jaccard += [(intersection + 1e-15) / (union + 1e-15)]

    return np.mean(jaccard)


if __name__ == '__main__':

    # DATA 
    #original_height, original_width = 480, 640 # jigsaws
    #original_height, original_width = 538, 701  # synthetic
    #original_height, original_width = 1024, 1280  # miccai

    #startx = 96 # jigsaws
    #starty = 0 # jigsaws, miccai
    #startx = 0 # miccai
    #startx = 58 # synthetic
    #starty = 61 # synthetic

    #data_type = 'JIGSAWS'
    #data_path = Path('data/'+data_type)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--original_height', type=int, default= 480)
    arg('--original_width', type=int, default= 640)
    arg('--startx', type=int, default= 96)
    arg('--starty', type=int, default= 0)
    arg('--data_type', type=str, default= 'JIGSAWS')
    arg('--data_path', type=str, default= 'data/JIGSAWS')
    arg('--cropped_train_path', type=str, default= 'steps/Cropped/JIGSAWS')

    args = parser.parse_args()

    original_height = args.original_height
    original_width = args.original_width
    startx = args.startx
    starty = args.starty
    data_type = args.data_type
    data_path = Path(args.data_path)
    cropped_train_path = Path(args.cropped_train_path)

    for i in range(1,10):

        crop(Path('data/JIGSAWS/instrument_dataset_'+str(i)+'/images'), original_height, original_width, startx, starty, cropped_train_path)
    
        optical_flow_with_hsv(Path('steps'),data_type)
        
        tracking(Path('steps'), data_type)

        postprocess(Path('steps'), data_type)

        acc = evaluate(Path('steps/postprocess/'+ data_type),Path('data/JIGSAWS/instrument_dataset_'+str(i)+'/binary_masks'), startx, original_height,starty,original_width, data_type )
        print('Accuracy of ',i, ' :', acc)
        dice = evaluate_dice(Path('steps/postprocess/'+ data_type),Path('data/JIGSAWS/instrument_dataset_'+str(i)+'/binary_masks'), startx, original_height,starty,original_width, data_type )
        print('Dice of ',i, ' :', dice)
        jaccard = evaluate_jaccard(Path('steps/postprocess/'+ data_type),Path('data/JIGSAWS/instrument_dataset_'+str(i)+'/binary_masks'), startx, original_height,starty,original_width, data_type )
        print('Jaccard of ',i, ' :', jaccard)

