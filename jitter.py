import numpy as np
import cv2

def adjust_jitter(inp_frames):
    n_frames = len(inp_frames)
    h, w, _ = inp_frames[0][1].shape 

    prev = inp_frames[0][1] 
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) 

    prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                            maxCorners=200,
                                            qualityLevel=0.01,
                                            minDistance=30,
                                            blockSize=3)
    out_frames = []
    inv_transforms = []
    for i in range(n_frames-1):
        cur = inp_frames[i+1][1]
        cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY) 

        cur_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_pts, None) 

        idx = np.where(status==1)[0]
        m, _ = cv2.estimateAffinePartial2D(cur_pts[idx], prev_pts[idx])
        m_inv, _ = cv2.estimateAffinePartial2D(prev_pts[idx], cur_pts[idx])
        # Ref: https://docs.opencv.org/3.4.15/d9/d0c/group__calib3d.html#gad767faff73e9cbd8b9d92b955b50062d
        
        cur_transformed = cv2.warpAffine(cur, m, (w,h))
        out_frames.append((inp_frames[i+1][0], cur_transformed))

        inv_transforms.append(m_inv)

        #print("Frame: " + str(i+2) +  "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts[idx])))

    return out_frames, inv_transforms


def correct_masks(masks, inv_transforms):
    h, w = masks[0][1].shape

    corrected_masks = []
    for i in range(len(masks)):
        filename, mask = masks[i]
        
        m = inv_transforms[i]
        corrected_mask = cv2.warpAffine(mask, m, (w,h))
        corrected_masks.append((filename, corrected_mask))
    return corrected_masks