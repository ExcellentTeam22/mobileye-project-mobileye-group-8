import numpy as np
import scipy


class Kernel:
    def __init__(self, threshold, kernel: np.array):
        self.threshold = threshold
        self.kernel = kernel
        self.__normalize_kernel()

    def __normalize_kernel(self):
        matrix_sum, to_divide = self.__calculate_matrix_sum()
        neg_value = -(matrix_sum / to_divide)
        self.__decrease_values(neg_value)

    def __calculate_matrix_sum(self):

        cells_in_threshold_sum = 0
        cells_not_in_threshold = 0

        for row in self.kernel:
            for number in row:
                if number > self.threshold:
                    cells_in_threshold_sum += number
                else:
                    cells_not_in_threshold += 1
        return cells_in_threshold_sum, cells_not_in_threshold

    def __decrease_values(self, neg_value):
        for index_row, row in enumerate(self.kernel):
            for index_col, number in enumerate(row):
                if number <= self.threshold:
                    self.kernel[index_row][index_col] = neg_value

    def convolution(self, image: np.array):
        return scipy.signal.convolve(image.copy(), self.kernel, mode='same')
