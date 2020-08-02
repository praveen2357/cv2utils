"""
/**
 * @author praveen kumar yalal
 * @email praveen2357@gmail.com
 * @desc General image processing utilities
 */

 ref: https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
"""

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


def stitch_images(images):
    """
    combine all images into a single large image
    images: list of image path locations
    """
    pass


def apply_color_temperature(im, temp):
    """
    add color temperature effect to the image
    """
    pass

