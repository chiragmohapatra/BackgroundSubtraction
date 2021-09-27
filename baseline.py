import cv2
import numpy as np

def post_process(masks, kernel_dim=5):
    kernel = np.ones((kernel_dim,kernel_dim),np.uint8)
    new_masks = []
    for filename, mask in masks:
        mask = mask.astype('uint8')
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        new_masks.append((filename, mask))
    return new_masks


def mean_filter(inp_frames, s, e, meanOverAll=True, k=40):
    # Get mean at each pixel
    mean = inp_frames[0][1].astype(float)
        
    num_frames = len(inp_frames)
    if not meanOverAll:
        num_frames = s-1
    for i in range(1, num_frames):
        mean = np.add(mean, inp_frames[i][1])
    mean = mean / num_frames
    mean = mean.astype(int)
    
    out_frames = []
    # Declare foreground if I - mean <= k
    for i in range(s-1, e):
        filename , img = inp_frames[i]
        mask = (np.abs(img - mean) >= k) * 255
        out_frames.append((filename, mask))

    return out_frames


def median_filter(inp_frames, s, e, medianOverAll=True, k=40):
    # Get median at each pixel
    nrows, ncols = inp_frames[0][1].shape
    num_frames = len(inp_frames)
    if not medianOverAll:
        num_frames = s-1
    all_inp = np.zeros((num_frames, nrows, ncols))

    for i in range(num_frames):
        all_inp[i,:,:] = inp_frames[i][1]
    median = np.median(all_inp, axis=0)
    median = median.astype(int)
    
    out_frames = []
    # Declare foreground if I - mean <= k
    for i in range(s-1, e):
        filename , img = inp_frames[i]
        mask = (np.abs(img - median) >= k) * 255
        out_frames.append((filename, mask))

    return out_frames

def gmm(inp_frames, s, e, detectShadows=False):
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = detectShadows)

    for i in range(s-1):
        fgbg.apply(inp_frames[i][1])

    out_frames = []
    for i in range(s-1,e):
        mask = fgbg.apply(inp_frames[i][1])
        mask = (mask > 200)*255
        mask = mask.astype('uint8')
        out_frames.append((inp_frames[i][0], mask))
        
    return out_frames

def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output


def running_gaussian_avg(inp_frames, s, e):
    
    variance_matrix = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])

    mean = inp_frames[0][1].astype(float)
    var = convolve2D(mean,variance_matrix,padding=1)
    num_frames = len(inp_frames)

    wro = 0.01

    for i in range(1, num_frames):
        mean = wro*inp_frames[i][1] + (1-wro)*mean
        d = np.abs(mean - inp_frames[i][1])
        d = d*d
        var = wro*d + (1-wro)*var
        
    mean = mean.astype(int)
    stddev = np.sqrt(var)

    out_frames = []
    for i in range(s-1,e):
        mask = ((np.abs((inp_frames[i][1] - mean)/stddev) >= 2.5) * 255).astype('uint8')
        out_frames.append((inp_frames[i][0], mask))
    return out_frames