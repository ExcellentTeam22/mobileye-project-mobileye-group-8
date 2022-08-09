import numpy as np
from PIL import Image
from skimage import measure
from skimage import filters
import matplotlib.pyplot as plt
from DataBase import DataBase
import pandas as pd
import tables


def crops_validation():
    # Get Data of crop images
    db = DataBase()

    decisions = []

    for index, row in db.get_crops_images().iterrows():
        if is_valid(row):
            decisions.append([str(index), "True"])
        else:
            decisions.append([str(index), "False"])

    db.add_tfls_decisions(pd.DataFrame(decisions, columns=["crop_index", "decision"]))
    db.print_tfl_decision()
    db.export_tfls_decisions_to_h5()


def is_valid(crop_data: pd.Series):
    # Need to build a function for the first DataBase to find original image by given index.

    image_name = DataBase().get_tfl(crop_data["original"])["Image"]

    city = image_name.split('_')[0]

    color_image_path = image_name.replace("leftImg8bit.png", "gtFine_color.png")
    label_image_path = image_name.replace("leftImg8bit.png", "gtFine_labelIds.png")

    color_image = np.array(Image.open("./Resources/gtFine/train/" + city + '/' + color_image_path))
    label_image = np.array(Image.open("./Resources/gtFine/train/" + city + '/' + label_image_path).convert("L"))

    color_image_crop = color_image[int(crop_data["y start"]):int(crop_data["y end"]),
                       int(crop_data["x start"]):int(crop_data["x end"])]

    all_labels = measure.label(label_image)
    blobs_labels = measure.label(label_image, background=0)

    first_tfl_pixel = None
    valid_pixels = 0

    for row in range(int(crop_data["y start"]), int(crop_data["y end"])):
        if row < 0:
            continue
        elif row >= color_image.shape[0]:
            break
        for col in range(int(crop_data["x start"]), int(crop_data["x end"])):
            if 0 < col < color_image.shape[1]:
                if np.array_equal(color_image[row][col], np.array([250, 170, 30, 255])):
                    if not first_tfl_pixel:
                        first_tfl_pixel = (row, col)
                    valid_pixels += 1

    if not first_tfl_pixel:
        return False

    component_num = blobs_labels[first_tfl_pixel[0]][first_tfl_pixel[1]]
    component_index = np.argwhere(blobs_labels == component_num)
    component_size = component_index.shape[0]

    result = (valid_pixels / component_size) * 100

    if result > 35:
        return True
    else:
        return False


if __name__ == '__main__':
    crops_validation()