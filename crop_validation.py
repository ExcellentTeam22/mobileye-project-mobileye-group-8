import numpy as np
from PIL import Image
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
    image_name = crop_data["original"]

    city = image_name.split('_')[0]

    color_image_path = image_name.replace("leftImg8bit.png", "gtFine_color.png")

    color_image = np.array(Image.open("./Resources/gtFine/train/" + city + '/' + color_image_path))

    color_image_crop = color_image[int(crop_data["y start"]):int(crop_data["y end"]),
                       int(crop_data["x start"]):int(crop_data["x end"])]

    # plt.imshow(color_image_crop)
    # plt.show()
    valid_pixels = 0
    for row in color_image_crop:
        for col in row:
            if np.array_equal(col, np.array([250, 170, 30, 255])):
                valid_pixels += 1

    result = valid_pixels / (color_image_crop.shape[0] * color_image_crop.shape[1]) * 100
    if result > 70:
        return True
    else:
        return False


if __name__ == '__main__':
    crops_validation()
