import cv2
import numpy as np

def equalizeHist(inp_frames):
    new_frames = []
    for filename, img in inp_frames:
        new_frames.append((filename, cv2.equalizeHist(img)))
    return new_frames

def normalise_pixel_vals(inp_frames):
    new_frames = []
    for filename, img in inp_frames:
        normalizedImg = np.zeros(img.shape)
        normalizedImg = cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
        new_frames.append((filename, normalizedImg))
    return new_frames

def blur_image(inp_frames, kernel_dim=5):
    new_frames = []
    for filename, img in inp_frames:
        new_frames.append((filename, cv2.GaussianBlur(img,(kernel_dim,kernel_dim),0)))
    return new_frames

def apply_clahe(inp_frames):
    new_frames = []
    for filename, img in inp_frames:
        lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)

        limg = cv2.merge((cl,a,b))

        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        new_frames.append((filename,final))
    return new_frames
    