import numpy as np
import cv2 as cv


def apply_zoomout(im, scale):
    """
    Zoom out image
    """
    h, w, dim = im.shape
    # create png image
    im = np.zeros((h, w, dim))
    c_h, c_w = int(h/2), int(w/2)
    im_zm = cv.resize(im, None, fx=scale, fy=scale,
                        interpolation=cv.INTER_AREA)
    height, width, d = im_zm.shape
    for i in range(height):
        for j in range(width):
            im[i+(c_h-int(height/2)), j +
                    (c_h-int(width/2))] = im_zm[i, j]
    return im


def apply_zoomin(im, scale):
    """
    zoom in image
    """
    h_o, w_o, _ = im.shape
    im_zm = cv.resize(im, None, fx=scale, fy=scale,
                        interpolation=cv.INTER_LINEAR)
    # crop the image
    height, width, d = im_zm.shape
    c_h, c_w = int(height/2), int(width/2)
    upper_left = (c_w - int(w_o/2), c_h - int(h_o/2))
    bottom_right = (c_w + int(w_o/2), c_h + int(h_o/2))
    im_zm = im_zm[upper_left[1]: bottom_right[1],
                    upper_left[0]: bottom_right[0]]
    return im_zm


def convert_to_png(im):
    """
    convert image to png
    """
    h, w, d = im.shape
    # convert to png
    im_png = np.ones((h, w, 4)) * 255
    im_png[:, :, 0] = im[:, :, 0]
    im_png[:, :, 1] = im[:, :, 1]
    im_png[:, :, 2] = im[:, :, 2]
    return im_png


def letterbox(im, o_w=None, o_h=None):
    """
    resize image, make it bigger or smaller maintaining the aspect ratio
    """
    h, w, d = im.shape
    ratio = 1
    interpolation = cv.INTER_LINEAR
    if o_w is not None and o_h is not None:
        ratio = min(o_w/w, o_h/h)
        o_h = int(round(ratio * h))
        o_w = int(round(ratio * w))

    elif o_w is not None:
        ratio = float(o_w/w)
        o_h = int(round(ratio * h))

    elif o_h is not None:
        ratio = float(o_h/h)
        o_w = int(round(ratio * w))

    print(ratio, o_w, o_h)
    if ratio < 1:
        interpolation = cv.INTER_AREA
    im_ltrbox = cv.resize(im, (o_w, o_h), interpolation=interpolation)
    return im_ltrbox


def yolo_resize(im, bboxes, target_size):
    """
    resize the image and respective bbox
    im: opencv ndarray
    bbox: x,y,width,height
    """
    h, w, d = im.shape
    im_ltrbox = letterbox(im, target_size, target_size)
    h_l, w_l, _ = im_ltrbox.shape
    # check if the image requires padding
    padding = 0
    if h_l < target_size:
        # add vertical padding
        padding = target_size - h_l
        y_t, x_t = padding/2, 0
        
    if w_l < target_size:
        # add horizontal padding
        padding = target_size - w_l
        y_t, x_t = 0, padding/2
    # apply translation
    T = np.float32([[1, 0, x_t], [0, 1, y_t]])
    im_tr = cv.warpAffine(im_ltrbox, T, (target_size, target_size))
    # apply resize + padding to the bboxes
    bboxes_tr = np.array(bboxes,dtype=np.float32)
    bboxes_tr[:,0] = bboxes_tr[:,0] * ((w_l*1.0)/w) + x_t
    bboxes_tr[:,1] = bboxes_tr[:,1] * ((h_l*1.0)/h) + y_t
    bboxes_tr[:,2] = bboxes_tr[:,2] * ((w_l*1.0)/w)
    bboxes_tr[:,3] = bboxes_tr[:,3] * ((h_l*1.0)/h)
    return im_tr, bboxes_tr

