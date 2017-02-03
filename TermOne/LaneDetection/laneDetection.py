import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import sys

reload(sys)
setdefaultencoding('utf-8')


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    x_size = img.shape[1]
    y_size = img.shape[0]
    lines_slope_intercept = np.zeros(shape=(len(lines), 2))
    for index, line in enumerate(lines):
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - x1 * slope
            lines_slope_intercept[index] = [slope, intercept]
    max_slope_line = lines_slope_intercept[lines_slope_intercept.argmax(axis=0)[0]]
    min_slope_line = lines_slope_intercept[lines_slope_intercept.argmin(axis=0)[0]]
    left_slopes = []
    left_intercepts = []
    right_slopes = []
    right_intercepts = []

    # this gets slopes and intercepts of lines similar to the lines with the max (immediate left) and min
    # (immediate right) slopes (i.e. slope and intercept within x%)
    for line in lines_slope_intercept:
        if abs(line[0] - max_slope_line[0]) < 0.15 and abs(line[1] - max_slope_line[1]) < (0.15 * x_size):
            left_slopes.append(line[0])
            left_intercepts.append(line[1])
        elif abs(line[0] - min_slope_line[0]) < 0.15 and abs(line[1] - min_slope_line[1]) < (0.15 * x_size):
            right_slopes.append(line[0])
            right_intercepts.append(line[1])

    # left and right lines are averages of these slopes and intercepts, extrapolate lines to edges and center*
    # *roughly
    new_lines = np.zeros(shape=(1, 2, 4), dtype=np.int32)

    if len(left_slopes) > 0:
        left_line = [sum(left_slopes) / len(left_slopes), sum(left_intercepts) / len(left_intercepts)]
        left_bottom_x = (y_size - left_line[1]) / left_line[0]
        left_top_x = (y_size * .575 - left_line[1]) / left_line[0]
        if (left_bottom_x >= 0):
            new_lines[0][0] = [left_bottom_x, y_size, left_top_x, y_size * .575]
    if len(right_slopes) > 0:
        right_line = [sum(right_slopes) / len(right_slopes), sum(right_intercepts) / len(right_intercepts)]
        right_bottom_x = (y_size - right_line[1]) / right_line[0]
        right_top_x = (y_size * .575 - right_line[1]) / right_line[0]
        if (right_bottom_x <= x_size):
            new_lines[0][1] = [right_bottom_x, y_size, right_top_x, y_size * .575]

    for line in new_lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, _alpha=0.8, _beta=1., _lambda=0.):
    return cv2.addWeighted(initial_img, _alpha, img, _beta, _lambda)


def filter_white_and_yellow_color(image):
    white_threshold = 200
    lower_white = np.array([white_threshold, white_threshold, white_threshold])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(image, lower_white, upper_white)
    white_image = cv2.bitwise_and(image, image, mask=white_mask)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([90, 100, 100])
    upper_yellow = np.array([110, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)

    result = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)
    return result


def detect_line_in_images(image, image_count):
    _image = mpimg.imread(test_images_dir + image)

    x_size = _image.shape[1]
    y_size = _image.shape[0]
    print (x_size)
    print (y_size)

    # Filter out image with white and yellow color
    _filter_img = filter_white_and_yellow_color(_image)
    plt.figure(image_count)
    plt.imshow(_filter_img)

    # Grayscale the image
    _image_gray = grayscale(_filter_img)
    image_count += 1
    plt.figure(image_count)
    plt.imshow(_image_gray)

    # Apply Gaussian smoothing to the image
    _image_blur_gray = gaussian_blur(_image_gray, kernel_size)
    image_count += 1
    plt.figure(image_count)
    plt.imshow(_image_blur_gray)

    # Apply open cv Canny Edge Detector
    _image_edges = canny(_image_blur_gray, low_threshold, high_threshold)
    image_count += 1
    plt.figure(image_count)
    plt.imshow(_image_edges)

    # Region Masking
    # Select region of interest in the image
    _imshape = _image.shape
    vertices = np.array([[(200, _imshape[0]), (450, 300), (600, 300), (960, _imshape[0])]], dtype=np.int32)
    _mask_edges = region_of_interest(_image_edges, vertices)
    image_count += 1
    plt.figure(image_count)
    plt.imshow(_mask_edges)

    _line_image = hough_lines(_mask_edges, rho, theta, threshold, min_line_len, max_line_gap)
    image_count += 1
    plt.figure(image_count)
    plt.imshow(_line_image)

    _weighted_image = weighted_img(_line_image, _image)
    image_count += 1
    plt.figure(image_count)
    plt.imshow(_weighted_image)
