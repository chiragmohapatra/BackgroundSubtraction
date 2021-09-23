""" This is the skeleton code for main.py
You need to complete the required functions. You may create addition source files and use them by importing here.
"""

import os
import cv2
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Get mIOU of video sequences')
    parser.add_argument('-i', '--inp_path', type=str, default='input', required=True, \
                                                        help="Path for the input images folder")
    parser.add_argument('-o', '--out_path', type=str, default='result', required=True, \
                                                        help="Path for the predicted masks folder")
    parser.add_argument('-c', '--category', type=str, default='b', required=True, \
                                                        help="Scene category. One of baseline, illumination, jitter, dynamic scenes, ptz (b/i/j/m/p)")
    parser.add_argument('-e', '--eval_frames', type=str, default='eval_frames.txt', required=True, \
                                                        help="Path to the eval_frames.txt file")
    args = parser.parse_args()
    return args


def get_input_frames(args, bnw=True):
    '''
    Get list of (image name, image as a numpy array)
    Params
    ------
    bnw : Bool
        convert to black and white
    '''
    filenames = os.listdir(args.inp_path)
    inp_frames = []
    for i, filename in enumerate(filenames):
        img = cv2.imread(os.path.join(args.inp_path, filename))
        if bnw:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inp_frames.append((filename, img))
        print(i)
    inp_frames = sorted(inp_frames, key=lambda x : x[0])
    return inp_frames

def get_eval_indices(args):
    '''
    Get start and end index as given in eval_frames
    '''
    s, e = 0, 0
    with open(args.eval_frames, 'r') as f:
        s, e = map(int, f.readline().split())
    return s, e

def write_output_frames(args, out_frames):
    '''
    Write output frames to file
    '''
    if not os.path.isdir(args.out_path):
        os.mkdir(args.out_path)

    for filename, img in out_frames:
        filename = 'gt' + filename[2:-3] + 'png'
        cv2.imwrite(os.path.join(args.out_path, filename), img)

def baseline_bgs(args):
    inp_frames = get_input_frames(args)

    # Get mean at each pixel
    mean = inp_frames[0][1].astype(float)
    num_frames = len(inp_frames)
    for i in range(1, num_frames):
        mean = np.add(mean, inp_frames[i][1])
    mean = mean / num_frames
    mean = mean.astype(int)

    s, e = get_eval_indices(args)

    k = 40 # threshold
    kernel = np.ones((5,5),np.uint8)
    
    out_frames = []
    # Declare foreground if I - mean <= k
    for i in range(s-1, e):
        filename, img = inp_frames[i]
        mask = (np.abs(img - mean) >= k) * 255
        mask = mask.astype('uint8')
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        out_frames.append((filename, mask))

    write_output_frames(args, out_frames)


def illumination_bgs(args):
    #TODO complete this function
    pass


def jitter_bgs(args):
    #TODO complete this function
    pass


def dynamic_bgs(args):
    #TODO complete this function
    pass


def ptz_bgs(args):
    #TODO: (Optional) complete this function
    pass


def main(args):
    if args.category not in "bijdp":
        raise ValueError("category should be one of b/i/j/m/p - Found: %s"%args.category)
    FUNCTION_MAPPER = {
            "b": baseline_bgs,
            "i": illumination_bgs,
            "j": jitter_bgs,
            "m": dynamic_bgs,
            "p": ptz_bgs
        }

    FUNCTION_MAPPER[args.category](args)

if __name__ == "__main__":
    args = parse_args()
    main(args)