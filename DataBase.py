from singletonDecorator import singleton

import pandas as pd

"""
This class represents a DataBase that holds all the information about tfl that founds by the program.
"""


@singleton
class DataBase:
    def __init__(self):

        self.tfl_coordinate = pd.DataFrame([], columns=["path", "y_bottom_left", "x_bottom_left",
                                                              "y_top_right", "x_top_right",
                                                              "col", "RGB", "pixel_light"])
        self.crop_images = pd.DataFrame([]
                                        , columns=["original", "crop_name", "zoom",
                                                   "x start", "x end", "y start", "y end", "col"])

        self.tfl_decision = pd.DataFrame([], columns=["seq", "is_true", "is_ignore", "path",
                                                      "x0", "x1", "y0", "y1", "col"])

    def get_tfl(self, index):
        return self.tfl_coordinate.iloc[index]

    def get_tfl(self, index):
        return self.tfl_coordinate.iloc[index]

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

    def get_crops(self, index):
        return self.crop_images.iloc[index]

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
        df = self.tfl_coordinate

        df["x_bottom_left"] = df["x_bottom_left"].astype(int)
        df["y_bottom_left"] = df["y_bottom_left"].astype(int)

        df["x_top_right"] = df["x_top_right"].astype(int)
        df["y_top_right"] = df["y_top_right"].astype(int)
        df["RGB"] = df["RGB"].astype(str)
        df["pixel_light"] = df["pixel_light"].astype(float)
        df["path"] = df["path"].astype(str)
        df["col"] = df["col"].astype(str)


        df.to_hdf("./Resources/attention_results/attention_results.h5", "Traffic_Lights_Coordinates", format="table")

    def export_crops_images_to_h5(self):
        df = self.crop_images

        df["x start"] = df["x start"].astype(int)
        df["x end"] = df["x end"].astype(int)

        df["y start"] = df["y start"].astype(int)
        df["y end"] = df["y end"].astype(int)
        df["col"] = df["col"].astype(str)
        df["zoom"] = df["zoom"].astype(float)
        df["original"] = df["original"].astype(str)
        df["crop_name"] = df["crop_name"].astype(str)

        df.to_hdf("./Resources/attention_results/crop_results0.h5", "Traffic_Lights_Coordinates", format="table")

    def export_tfls_decisions_to_h5(self):
        df = self.tfl_decision

        df["seq"] = df["seq"].astype(int)
        df["is_true"] = df["is_true"].astype(bool)
        df["is_ignore"] = df["is_ignore"].astype(bool)

        df["x0"] = df["x0"].astype(int)
        df["x1"] = df["x1"].astype(int)

        df["y0"] = df["y0"].astype(int)
        df["y1"] = df["y1"].astype(int)

        df["col"] = df["col"].astype(str)

        df["path"] = df["path"].astype(str)

        df.to_hdf("./Resources/attention_results/crop_results.h5", "crop_results0", format="table")


