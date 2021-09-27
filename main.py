""" This is the skeleton code for main.py
You need to complete the required functions. You may create addition source files and use them by importing here.
"""

from illumination import blur_image, light_consistent_image, equalizeHist, normalise_pixel_vals, resize_frames
import os
from vibe import get_vibe_masks
import cv2
import argparse
import numpy as np
import baseline

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
    s, e = get_eval_indices(args)

    masks = baseline.median_filter(inp_frames, s, e, True)
    masks = baseline.post_process(masks)

    write_output_frames(args, masks)

def illumination_bgs(args):
    inp_frames = get_input_frames(args , True)
    s, e = get_eval_indices(args)

    inp_frames = light_consistent_image(inp_frames)
    inp_frames = blur_image(inp_frames , kernel_dim=5)

    masks = baseline.gmm(inp_frames, s, e)
    masks = baseline.post_process(masks)

    write_output_frames(args, masks)


def jitter_bgs(args):
    #TODO complete this function
    pass


def dynamic_bgs(args):
    inp_frames = get_input_frames(args)
    s, e = get_eval_indices(args)

    inp_frames = blur_image(inp_frames , kernel_dim=3)

    masks = baseline.median_filter(inp_frames, s, e, True, k=50)
    masks = baseline.post_process(masks)

    write_output_frames(args, masks)


def ptz_bgs(args):
    #TODO: (Optional) complete this function
    pass


def main(args):
    if args.category not in "bijdmp":
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