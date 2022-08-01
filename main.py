try:
    import scipy
    import os
    import json
    import glob
    import argparse

    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage import maximum_filter

    from PIL import Image

    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    ### WRITE YOUR CODE HERE ###
    ### USE HELPER FUNCTIONS ###
    return [500, 510, 520], [500, 500, 500], [700, 710], [500, 500]


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


def calculate_matrix_sum(image_arr):
    cells_in_threshold_sum = 0
    cells_not_in_threshold = 0

    for row in image_arr:
        for number in row:
            if number > 100:
                cells_in_threshold_sum += number
            else:
                cells_not_in_threshold += 1
    return cells_in_threshold_sum, cells_not_in_threshold


def normalized_kernel(neg_value: int, kernel):

    for index_row, row in enumerate(kernel):
        for index_col, number in enumerate(row):
            if number < 100:
                kernel[index_row][index_col] = neg_value
    return kernel


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    # parser = argparse.ArgumentParser("Test TFL attention mechanism")
    # parser.add_argument('-i', '--image', type=str, help='Path to an image')
    # parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    # parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    # args = parser.parse_args(argv)
    # default_base = "INSERT_YOUR_DIR_WITH_PNG_AND_JSON_HERE"
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
    # if len(flist):
    #     print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    # else:
    #     print("Bad configuration?? Didn't find any picture to show")


    image_array = np.array(Image.open('Test/berlin_000540_000019_leftImg8bit.png').convert('L'), np.float64)
    kernel_before_normalization = image_array[210:265, 1085:1115]

    # Normalize kernel section
    sum, to_divide = calculate_matrix_sum(kernel_before_normalization)
    neg_value = sum / to_divide
    kernel = normalized_kernel(int(-neg_value), kernel_before_normalization)

    convolution_image = scipy.signal.convolve(image_array, kernel, mode='same')
    plt.imshow(convolution_image)

    plt.show(block=True)


if __name__ == '__main__':
    main()
