import pandas

from DataBase import DataBase
from Kernel import Kernel

try:
    import scipy
    import os
    import json
    import glob
    import argparse
    import pandas as pd
    from crop_validation import crops_validation

    pd.set_option('display.width', 200, 'display.max_rows', 200,
                  'display.max_columns', 200, 'max_colwidth', 40)

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


def display_figures(original_image, filtered_red_lights, filtered_green_lights, red_rectangles, green_rectangles):
    """
    Displays original image and the convolutions images with green and red dots
    that indicates what the program founded has traffic lights.
    :param original_image: The original image.
    :param filtered_red_lights: The x,y coordinates for the detected optional traffic lights with red bulb.
    :param filtered_green_lights: The x,y coordinates for the detected optional traffic lights with green bulb.
    :return: None
    """
    figure, ax = plt.subplots(1)
    plt.imshow(original_image)
    if len(filtered_green_lights) != 0:
        plt.plot(filtered_green_lights[:, 1], filtered_green_lights[:, 0], 'g.')
    if len(filtered_red_lights) != 0:
        plt.plot(filtered_red_lights[:, 1], filtered_red_lights[:, 0], 'r.')

    for rec in red_rectangles:
        ax.add_patch(plt.Rectangle((rec[0][1], rec[1][0])
                                   , rec[1][1] - rec[0][1]
                                   , rec[0][0] - rec[1][0],
                                   edgecolor='r',
                                   facecolor="none"))
    for rec in green_rectangles:
        ax.add_patch(plt.Rectangle((rec[0][1], rec[1][0])
                                   , rec[1][1] - rec[0][1]
                                   , rec[0][0] - rec[1][0],
                                   edgecolor='g',
                                   facecolor="none"))
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

    red_with_info, red_tfl, red_rectangles = find_light_coordinates(c_image, kwargs["kernel_red_light"], 0, 300000,
                                                                    kwargs["path"],
                                                                    "Red")
    green_with_info, green_tfl, green_rectangles = find_light_coordinates(c_image, kwargs["kernel_green_light"], 1,
                                                                          1100000,
                                                                          kwargs["path"],
                                                                          "Green")

    if not len(red_tfl) and not len(green_tfl):
        return [], [], [], []
    elif not len(red_tfl):
        tfl_with_info = green_with_info
    elif not len(green_tfl):
        tfl_with_info = red_with_info
    else:
        tfl_with_info = np.concatenate([red_with_info, green_with_info])

    current_data_frame = pd.DataFrame(tfl_with_info, columns=["Image", "bottom_left", "top_right",
                                                              "light", "RGB", "pixel_light"])

    db = DataBase()
    db.add_tfl(current_data_frame)

    display_figures(c_image, red_tfl, green_tfl, red_rectangles, green_rectangles)

    if not len(red_tfl):
        return [], [], green_tfl[:, 0], green_tfl[:, 1]
    elif not len(green_tfl):
        return red_tfl[:, 0], red_tfl[:, 1], [], []

    return red_tfl[:, 0], red_tfl[:, 1], green_tfl[:, 0], green_tfl[:, 1]


def solve(bl, tr, p):
    return bl[1] <= p[1] <= tr[1] and bl[0] >= p[0] >= tr[0]


def get_zoom_rect():
    for index, row in DataBase().get_tfls_coordinates().iterrows():
        expended_rect(row, index)
        # DataBase().print_crop_images()


def expended_rect(row, index):
    rect = [row["bottom_left"], row["top_right"]]
    color = row["light"]

    city = row["Image"].split('_')[0]

    image = Image.open("./Resources/leftImg8bit/train/" + city + '/' + row["Image"])
    rect_height = rect[0][0] - rect[1][0]
    rect_width = rect[1][1] - rect[0][1]

    if color == "Red":
        rect[0][1] -= rect_width * 2
        rect[1][1] += rect_width * 1
        rect[0][0] += rect_width * 5.5
        rect[1][0] -= rect_width * 1.5
    else:
        rect[0][1] -= rect_width * 2
        rect[1][1] += rect_width * 2
        rect_width = rect[1][1] - rect[0][1]
        rect[0][0] += rect_width * 0.5
        rect[1][0] -= rect_width * 2.5

    cropped_image = image.crop((rect[0][1], rect[1][0], rect[1][1], rect[0][0]))
    cropped_image = cropped_image.resize((40, 100))
    cropped_image.save("./Resources/CropImages/" + row["Image"].replace(".png", "_crop_" + str(index) + ".png"))

    zoom = get_zoom_percentage(np.array(image), rect_height * rect_width)
    image_name = row["Image"].replace(".png", "_crop_" + str(index) + ".png")

    df = pandas.DataFrame(
        [[index, image_name, zoom, rect[0][1], rect[1][1], rect[1][0], rect[0][0]]],
        columns=["original", "crop_name", "zoom", "x start", "x end", "y start", "y end"])

    DataBase().add_crop_image(df)


def get_zoom_percentage(image, rect_area):
    image_x, image_y, image_z = image.shape
    area_of_image = image_x * image_y
    return rect_area / area_of_image


def find_light_coordinates(image: np.array, kernel: Kernel, dimension: int, threshold: int, image_name: str,
                           light_color: str):
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
            if image[row][col][1] > image[row][col][0] + 30 and image[row][col][2] > image[row][col][0] + 30 \
                    and image[row][col][1] > image[row][col][0] + image[row][col][2]:
                filtered_tfl += [[row, col]]

    rectangles = []
    max_min = []

    for points_index, point in enumerate(filtered_tfl):
        add_rectangle = True
        if points_index == 0:
            min_x = max_x = point[1]
            min_y = max_y = point[0]
            rectangles.append([[point[0] + 11, point[1] - 13], [point[0] - 11, point[1] + 13]])
            max_min.append([[min_x, max_x], [min_y, max_y]])
        else:
            for rectangles_index, rectangle in enumerate(rectangles):
                if solve(rectangle[0], rectangle[1], point):
                    if max_min[rectangles_index][0][1] < point[1]:
                        max_min[rectangles_index][0][1] = point[1]
                    if max_min[rectangles_index][0][0] > point[1]:
                        max_min[rectangles_index][0][0] = point[1]

                    if max_min[rectangles_index][1][1] < point[0]:
                        max_min[rectangles_index][1][1] = point[0]
                    if max_min[rectangles_index][1][0] > point[0]:
                        max_min[rectangles_index][1][0] = point[0]

                    add_rectangle = False

            if add_rectangle:
                rectangles.append([[point[0] + 11, point[1] - 13], [point[0] - 11, point[1] + 13]])
                min_x = max_x = point[1]
                min_y = max_y = point[0]
                max_min.append([[min_x, max_x], [min_y, max_y]])

    for rectangles_index, rectangle in enumerate(rectangles):
        rectangle[1][1] = max_min[rectangles_index][0][1]
        rectangle[0][1] = max_min[rectangles_index][0][0]

        rectangle[1][0] = max_min[rectangles_index][1][0]
        rectangle[0][0] = max_min[rectangles_index][1][1]

    tfl_with_info = list(map(lambda rect: [image_name.split('/')[-1],
                                           rect[0],
                                           rect[1],
                                           light_color,
                                           image[rect[0][0]][rect[0][1]],
                                           convolution_image_red[rect[0][0]][rect[0][1]]
                                           ], rectangles))

    return tfl_with_info, np.array(filtered_tfl), rectangles


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    # Builds the red and the green kernels.
    image_for_red_kernel = open_image_as_np_array('./Resources/Kernel/berlin_000455_000019_leftImg8bit.png')
    image_for_green_kernel = open_image_as_np_array('./Resources/Kernel/aachen_000012_000019_leftImg8bit.png')

    # plt.imshow(Image.open('./Resources/Kernel/aachen_000012_000019_leftImg8bit.png'))
    # plt.show()
    kernel_red_light = kernels_creator(image_for_red_kernel[:, :, 0], start_y=257, end_y=265, start_x=1124, end_x=1133,
                                       threshold=232)
    kernel_green_light = kernels_creator(image_for_green_kernel[:, :, 1], start_y=194, end_y=210, start_x=1208,
                                         end_x=1223,
                                         threshold=220)

    # Opens each PNG and start the convolution process.
    for root, dirs, files in os.walk('./Resources/leftImg8bit/train'):
        for file in files:
            path = root + '/' + file
            original_image = np.array(Image.open(path))

            find_tfl_lights(original_image, path=path, kernel_red_light=kernel_red_light,
                            kernel_green_light=kernel_green_light)

if __name__ == '__main__':
    main()
    get_zoom_rect()
    crops_validation()
