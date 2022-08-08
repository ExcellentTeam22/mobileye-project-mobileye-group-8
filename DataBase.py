from singletonDecorator import singleton

import pandas as pd

"""
This class represents a DataBase that holds all the information about tfl that founds by the program.
"""


@singleton
class DataBase:
    def __init__(self):
        self.tfl_coordinate = pd.DataFrame([], columns=["Image", "y-coordinate", "x-coordinate",
                                                        "light", "RGB", "pixel_light"])
        self.crop_images = pd.DataFrame([["aachen_000035_000019_leftImg8bit.png", "crop_name", "1",
                                          "1255", "1291", "219", "305"],
                                         ["aachen_000035_000019_leftImg8bit.png", "crop_name", "1",
                                          "304", "350", "219", "305"]
                                         ], columns=["original", "crop_name", "zoom",
                                                     "x start", "x end", "y start", "y end"])

        self.tfl_decision = pd.DataFrame([], columns=["crop_index", "decision"])

    def add_tfl(self, df: pd.DataFrame):
        """
        Adds a given DataFrame to the database.
        :param df: The requested DataFrame to has to the database.
        :return: None
        """
        self.tfl_coordinate = pd.concat([self.tfl_coordinate, df], ignore_index=True)

    def add_crop_image(self, df: pd.DataFrame):
        self.crop_images = pd.concat([self.crop_images, df], ignore_index=True)

    def add_tfls_decisions(self, df: pd.DataFrame):
        self.tfl_decision = pd.concat([self.tfl_decision, df], ignore_index=True)

    def get_tfls_coordinates(self):
        """
        :return: DataBase
        """
        return self.tfl_coordinate

    def get_crops_images(self):
        return self.crop_images

    def get_tfls_decisions(self):
        return self.tfl_decision

    def print_tfl_coordinate(self):
        """
        Prints the tfl_coordinate table.
        :return: None
        """
        print(self.tfl_coordinate)

    def print_crop_images(self):
        """
        Prints the crop_images table.
        :return: None
        """
        print(self.crop_images)

    def print_tfl_decision(self):
        """
        Prints the tfl_decision table.
        :return: None
        """
        print(self.tfl_decision)

    def export_tfls_coordinates_to_h5(self):
        self.tfl_coordinate.to_hdf("DataBase.h5", "Traffic_Lights_Coordinates", format="table")

    def export_crops_images_to_h5(self):
        self.crop_images.to_hdf("DataBase.h5", "Traffic_Lights_Crops_Images", format="table")

    def export_tfls_decisions_to_h5(self):
        self.tfl_decision.to_hdf("DataBase.h5", "Traffic_Lights_Decisions", format="table")
