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
    # Displays original image and the convolutions images.

    plt.imshow(original_image)
    if (len(filtered_green_lights) != 0):
        plt.plot(filtered_green_lights[:, 1], filtered_green_lights[:, 0], 'g.')
    if (len(filtered_red_lights) != 0):
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

    red_with_info, red_tfl = find_light_coordinates(c_image, kwargs["kernel_red_light"], 0, 300000, kwargs["path"],
                                                    "Red")
    green_with_info, green_tfl = find_light_coordinates(c_image, kwargs["kernel_green_light"], 1, 18000, kwargs["path"],
                                                        "Green")

    tfl_with_info = list()

    if not len(red_tfl) and not len(green_tfl):
        return [], [], [], []
    elif not len(red_tfl):
        tfl_with_info = red_with_info
    elif not len(green_tfl):
        tfl_with_info = green_with_info
    else:
        np.concatenate([red_with_info, green_with_info])

    current_data_frame = pd.DataFrame(tfl_with_info, columns=["Image", "y-coordinate", "x-coordinate",
                                                              "light", "RGB", "pixel_light"])

    # cropped_data_frame = pd.DataFrame(tfl_with_info, columns=["index original image", "image","x start",
    #                                                           "x end", "y start", "y end"])
    db = DataBase()
    db.add(current_data_frame)

    display_figures(c_image, red_tfl, green_tfl)

    if not len(red_tfl):
        return [], [], green_tfl[:, 0], green_tfl[:, 1]
    elif not len(green_tfl):
        return red_tfl[:, 0], red_tfl[:, 1], [], []

    return red_tfl[:, 0], red_tfl[:, 1], green_tfl[:, 0], green_tfl[:, 1]


def get_zoom_rect(rectangles):
    df = pd.DataFrame([[]], columns=["Image", "y-coordinate", "x-coordinate",
                                     "light", "RGB", "pixel_light",
                                     "bottom left", "top right"])
    for index, row in df.get_date().iterrows():
        expended_rect(row, index)


def expended_rect(row, index):
    rect = [row["bottom left"], row["top right"]]
    color = row["RGB"]
    image = Image.open(row["image"])
    rect_height = rect[0][0] - rect[1][0]
    rect_width = rect[1][0] - rect[0][0]
    rect[0][1] -= rect_width * 0.5
    rect[1][1] += rect_width * 0.5
    if color == "red":
        rect[0][0] += rect_height * 3
        rect[1][0] -= rect_height * 0.5
    else:
        rect[1][0] -= rect_height * 3
        rect[0][0] += rect_height * 0.5

    cropped_image = image.crop((rect[0][1], rect[0][0], rect[1][1], rect[1][0]))
    cropped_image = cropped_image.resize((40, 40))
    cropped_image.save("crop_image/" + row["image"].replace(".png", "_crop_" + index + ".png"))

    zoom = get_zoom_percentage(image, rect_height * rect_width)
    image_name = row["image"].replace(".png", "_crop_" + index + ".png")

    # df = pandas.DataFrame(
    #     [index, image_name, zoom, rect[0][1], rect[1][1], rect[1][0], rect[0][0]],
    #     columns=["Image", "y-coordinate", "x-coordinate",
    #              "original", "image crop", "zoom", "x start", "x end", "y start", "y end"])



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

    tfl = np.argwhere(maximum_filter(convolution_image_red, 3) > threshold)

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


### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    show_image_and_gt(image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(image)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)


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

    paths = ['Test/berlin_000540_000019_leftImg8bit.png',
             'Test/berlin_000522_000019_leftImg8bit.png',
             'Test/berlin_000455_000019_leftImg8bit.png',
             'Test/bremen_000145_000019_leftImg8bit.png',
             'Test/darmstadt_000053_000019_leftImg8bit.png',
             'Test/jena_000032_000019_leftImg8bit.png',
             'Test/stuttgart_000004_000019_leftImg8bit.png',
             'Test/ulm_000052_000019_leftImg8bit.png',
             'Test/bremen_000004_000019_leftImg8bit.png',
             'Test/darmstadt_000034_000019_leftImg8bit.png',
             'Test/dusseldorf_000143_000019_leftImg8bit.png',
             'Test/krefeld_000000_036299_leftImg8bit.png',
             'Test/stuttgart_000175_000019_leftImg8bit.png',
             'Test/zurich_000080_000019_leftImg8bit.png',
             'Test/berlin_000526_000019_leftImg8bit.png',
             'Test/bremen_000084_000019_leftImg8bit.png',
             'Test/darmstadt_000043_000019_leftImg8bit.png',
             'Test/hamburg_000000_067799_leftImg8bit.png',
             'Test/tubingen_000120_000019_leftImg8bit.png',
             ]

    for path in paths:
        original_image = np.array(Image.open(path))

        find_tfl_lights(original_image, path=path, kernel_red_light=kernel_red_light,
                        kernel_green_light=kernel_green_light)

    plt.show(block=True)

    # parser = argparse.ArgumentParser("Test TFL attention mechanism")
    # parser.add_argument('-i', '--image', type=str, help='Path to an image')
    # parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    # parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    # args = parser.parse_args(argv)
    # default_base = "INSERT_YOUR_DIR_WITH_PNG_AND_JSON_HERE"
    #
    # if args.dir is None:
    #     args.dir = default_base
    # flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
    #
    # for image in flist:
    #     json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
    #
    #     if not os.path.exists(json_fn):
    #         json_fn = None
    #     test_find_tfl_lights(image, json_fn)
    #
    # if len(flist):
    #     print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    # else:
    #     print("Bad configuration?? Didn't find any picture to show")


if __name__ == '__main__':
    main()
