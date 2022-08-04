from DataBase import DataBase
from Kernel import Kernel

try:
    import scipy
    import os
    import json
    import glob
    import argparse
    import pandas as pd

    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage import maximum_filter

    from PIL import Image

    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise


def open_image_as_np_array(path: str):
    """
    Opens image from a given path and convert it to a NumPy array (matrix of RGB).
    :param path: The path to the requested image.
    :return: A NumPy array that represent the image as a 3D array of RGB.
    """
    return np.array(Image.open(path), np.float64)


def kernels_creator(image, start_y, end_y, start_x, end_x, threshold):
    """
    This function create a kernel from a given image as np.array object and the relevant coordinates.
    :param image: Source image for extracting kernel
    :param start_y: The start point on the y axis of the kernel.
    :param end_y: The end point on the y axis of the kernel.
    :param start_x: The start point on the x axis of the kernel.
    :param end_x: The end point on the x axis of the kernel.
    :param threshold: The requested threshold for the normalization process of the kernel.
    :return: A Kernel object that represent the requested kernel according to the given information.
    """
    return Kernel(threshold, image[start_y:end_y, start_x:end_x].copy())


def display_figures(original_image, filtered_red_lights, filtered_green_lights):
    """
    Displays original image and the convolutions images with green and red dots
    that indicates what the program founded has traffic lights.
    :param original_image: The original image.
    :param filtered_red_lights: The x,y coordinates for the detected optional traffic lights with red bulb.
    :param filtered_green_lights: The x,y coordinates for the detected optional traffic lights with green bulb.
    :return: None
    """

    plt.imshow(original_image)
    if len(filtered_green_lights) != 0:
        plt.plot(filtered_green_lights[:, 1], filtered_green_lights[:, 0], 'g.')
    if len(filtered_red_lights) != 0:
        plt.plot(filtered_red_lights[:, 1], filtered_red_lights[:, 0], 'r.')
    plt.autoscale(False)
    plt.axis('off')
    plt.show()


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """

    red_with_info, red_tfl = find_light_coordinates(c_image, kwargs["kernel_red_light"], 0, 300000, kwargs["path"], "Red")
    green_with_info, green_tfl = find_light_coordinates(c_image, kwargs["kernel_green_light"], 1, 18000, kwargs["path"], "Green")

    tfl_with_info = list()

    if not len(red_tfl) and not len(green_tfl):
        return [], [], [], []
    elif not len(red_tfl):
        tfl_with_info = red_with_info
    elif not len(green_tfl):
        tfl_with_info = green_with_info
    else:
        tfl_with_info = np.concatenate([red_with_info, green_with_info])

    current_data_frame = pd.DataFrame(tfl_with_info, columns=["Image", "y-coordinate", "x-coordinate",
                                                              "light", "RGB", "pixel_light"])

    db = DataBase()
    db.add(current_data_frame)

    # display_figures(c_image, red_tfl, green_tfl)

    if not len(red_tfl):
        return [], [], green_tfl[:, 0], green_tfl[:, 1]
    elif not len(green_tfl):
        return red_tfl[:, 0], red_tfl[:, 1], [], []

    return red_tfl[:, 0], red_tfl[:, 1], green_tfl[:, 0], green_tfl[:, 1]


def find_light_coordinates(image: np.array, kernel: Kernel, dimension: int, threshold: int, image_name: str, light_color: str):
    """
    The function get an image and a kernel and return all the coordinates in the image that
    meet the given threshold after a maximum filter operation.
    In addition the function return a list of the coordinates and information for each of them.
    :param image: Original image to convolve.
    :param kernel: A kernel for the convolution process.
    :param dimension: The dimension of the color to filter the image.
    :param threshold: A threshold to extract relevant coordinates.
    :param image_name: The name of the original image.
    :param light_color: The color of the requested light.
    :return: Tuple of two list, one for the coordinates and another for the coordinates with information.
    """
    # Performs the convolution process on the red dimension and the green dimension of the image separately.
    convolution_image_red = kernel.convolution(image[:, :, dimension].copy())

    tfl = np.argwhere(maximum_filter(convolution_image_red, 1) > threshold)

    filtered_tfl = []

    if dimension == 0:
        for row, col in tfl:
            if image[row][col][0] > image[row][col][1] + 50 and image[row][col][0] > image[row][col][2] + 50:
                filtered_tfl += [[row, col]]
    else:
        for row, col in tfl:
            if image[row][col][1] > image[row][col][0] + 30 and image[row][col][2] > image[row][col][0] + 30:
                filtered_tfl += [[row, col]]

    tfl_with_info = list(map(lambda coordinate: [image_name,
                                                 coordinate[0],
                                                 coordinate[1],
                                                 light_color,
                                                 image[coordinate[0]][coordinate[1]],
                                                 convolution_image_red[coordinate[0]][coordinate[1]]
                                                 ], filtered_tfl))

    return tfl_with_info, np.array(filtered_tfl)


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    # Builds the red and the green kernels.
    image_for_red_kernel = open_image_as_np_array('Test/berlin_000455_000019_leftImg8bit.png')

    kernel_red_light = kernels_creator(image_for_red_kernel[:, :, 0], start_y=257, end_y=265, start_x=1124, end_x=1133,
                                       threshold=232)
    kernel_green_light = kernels_creator(image_for_red_kernel[:, :, 1], start_y=257, end_y=265, start_x=1124,
                                         end_x=1133,
                                         threshold=232)

    # Opens each PNG and start the convolution process.
    for root, dirs, files in os.walk('./Resource/leftImg8bit/train'):
        for file in files:
            path = root + '/' + file
            original_image = np.array(Image.open(path))

            find_tfl_lights(original_image, path=path, kernel_red_light=kernel_red_light,
                            kernel_green_light=kernel_green_light)

    db = DataBase()
    db.print_to_file()


if __name__ == '__main__':
    main()
