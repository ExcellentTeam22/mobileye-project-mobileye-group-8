from singletonDecorator import singleton

import pandas as pd

"""
This class represents a DataBase that holds all the information about tfl that founds by the program.
"""
@singleton
class DataBase:
    def __init__(self):
        self.data = pd.DataFrame([], columns=["Image", "y-coordinate", "x-coordinate", "light", "RGB", "pixel_light"])

    def add(self, df: pd.DataFrame):
        """
        Adds a given DataFrame to the database.
        :param df: The requested DataFrame to has to the database.
        :return: None
        """
        self.data = pd.concat([self.data, df], ignore_index=True)

    def get_data(self):
        """
        :return: DataBase
        """
        return self.data

    def print_data_base(self):
        """
        Prints the DataBase.
        :return: None
        """
        print(self.data)

    def print_to_file(self):
        self.data.to_csv("DataBase")
