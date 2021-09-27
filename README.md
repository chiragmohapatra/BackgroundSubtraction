This repository is for the first assignment of COL780 - Computer Vision.

## References

- https://en.wikipedia.org/wiki/Foreground_detection
- https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
- https://docs.opencv.org/3.1.0/db/d5c/tutorial_py_bg_subtraction.html
- https://stackoverflow.com/questions/24341114/simple-illumination-correction-in-images-opencv-c
- https://dsp.stackexchange.com/questions/46174/image-shadow-removal-using-opencv-and-python
- https://github.com/victorgzv/Lighting-correction-with-OpenCV/blob/master/lighting_correction.py
- https://stackoverflow.com/questions/49461314/how-to-detect-moving-object-in-varying-lightillumination-conditions-due-to-clo
- https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html
- https://github.com/felixchenfy/test_background_subtraction
- https://stackoverflow.com/questions/48697667/how-to-combine-background-subtraction-with-dense-optical-flow-tracking-in-opencv
- https://github.com/spmallick/learnopencv/tree/master/VideoStabilization


## Experiments

### Illumination
- Using only GMM (rgb, with detectShadows=False) caused shadows to appear alongwith large swaths of white due to light
- Using GMM with detectShadows=True, rgb
    - kernel 5x5 : mIOU=0.28
    - kernel 7x7, 9x9 : mIOU=0.30
    - kernel 11x11 : mIOU=0.2969
- grayscale, with histogram equalisation: large white portions in bkg
- grayscale with best performing GMM above : 0.2252
- grayscale with best performing GMM with image normalisation : 0.2346
- rgb with best performing GMM with image normalisation : 0.3 (no effect of image normalisation)
- terrible results using median filter
- **Main challenge : remove effect of moving clouds**
    - no improvement on adding gaussian blur to input
    - no improvement on applying CLAHE

# Jitter
- Applying GMM to transformed frames : 0.4419
- After smoothing with
    - 3x3 kernel : 0.5164
    - 5x5 kernel : 0.5247
    - 7x7 kernel : 0.5208
- After applying correction : 0.63 (7x7 kernel)






