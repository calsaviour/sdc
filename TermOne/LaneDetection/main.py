import math
import numpy as np


def draw_lines(lines):
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
    right_line_coordinates = []
    right_line_slope_intercept = []
    left_line_coordinates = []
    left_line_slope_intercept = []

    for line in lines:
        for x1, y1, x2, y2 in line:

            ## Right line has positive slope and lef line has negative slope
            slope = float(y1 - y2) / float(x1 - x2)
            print (slope)
            if not math.isnan(slope):
                if slope > 0:
                    right_line_coordinates.append([x1, y1])
                    right_line_coordinates.append([x2, y2])
                    right_line_slope_intercept.append([slope, y1 - slope * x1])
                elif slope < 0:
                    left_line_coordinates.append([x1, y1])
                    left_line_coordinates.append([x2, y2])
                    left_line_slope_intercept.append([slope, y1 - slope * x1])

    right_slope = [pair[0] for pair in right_line_slope_intercept]
    right_intercept = [pair[1] for pair in right_line_slope_intercept]

    left_slope = [pair[0] for pair in left_line_slope_intercept]
    left_intercept = [pair[1] for pair in left_line_slope_intercept]

    # Compute the mean for slope and intercept
    mean_right_slope = np.mean(right_slope)
    mean_right_intercept = np.mean(right_intercept)
    mean_left_slope = np.mean(left_slope)
    mean_left_intercept = np.mean(left_intercept)

    intersection_x_coordinate = (mean_right_intercept - mean_left_intercept) / (mean_right_slope - mean_left_slope)
    print (intersection_x_coordinate)
    _x1 = int(intersection_x_coordinate)
    _y1 = int(intersection_x_coordinate * mean_right_slope + mean_right_intercept)

    _x2 = int(intersection_x_coordinate)
    _y2 = int(intersection_x_coordinate * mean_left_slope + mean_left_intercept)
    print (abs(_x1), abs(_y1), abs(_x2), abs(_y2))


if __name__ == "__main__":
    lines = [[[625, 388, 897, 538]], [[605, 380, 837, 514]], [[584, 369, 878, 539]],
             [[290, 462, 441, 340]], [[306, 440, 440, 339]], [[281, 462, 416, 356]],
             [[292, 462, 436, 341]]]
    draw_lines(lines)
