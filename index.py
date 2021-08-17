import numpy as np
import cv2 as cv


class ImageProcessing:

    def __zeros_matrix(self, rows, cols):
        matrix = []
        while len(matrix) < rows:
            matrix.append([])
            while len(matrix[-1]) < cols:
                matrix[-1].append(0.0)

        return matrix

    def __check_multiply(self, first_matrix, second_matrix):
        row_b = len(second_matrix)
        col_a = len(first_matrix[0])

        if col_a != row_b:
            raise ArithmeticError('Number of A columns must equal number of B rows.')

    def __check_sum_and_sub(self, first_matrix, second_matrix):
        row_a = len(first_matrix)
        row_b = len(second_matrix)
        col_a = len(first_matrix[0])
        col_b = len(second_matrix[0])

        if row_a != row_b or col_a != col_b:
            raise ArithmeticError('Matrices are not the same size!')

    def add(self, first_matrix, second_matrix):
        self.__check_sum_and_sub(first_matrix, second_matrix)
        row_a = len(first_matrix)
        col_b = len(second_matrix[0])

        new_matrix = self.__zeros_matrix(row_a, col_b)

        for i in range(row_a):
            for j in range(col_b):
                new_matrix[i][j] = first_matrix[i][j] + second_matrix[i][j]

        return new_matrix

    def sub(self, first_matrix, second_matrix):
        self.__check_sum_and_sub(first_matrix, second_matrix)
        row_a = len(first_matrix)
        col_b = len(second_matrix[0])

        new_matrix = self.__zeros_matrix(row_a, col_b)

        for i in range(row_a):
            for j in range(col_b):
                new_matrix[i][j] = first_matrix[i][j] - second_matrix[i][j]

        return new_matrix

    def multiply(self, first_matrix, second_matrix):
        row_a = len(first_matrix)
        col_a = len(first_matrix[0])
        col_b = len(second_matrix[0])

        new_matrix = self.__zeros_matrix(row_a, col_b)

        for i in range(row_a):
            for j in range(col_b):
                total = 0
                for k in range(col_a):
                    total += first_matrix[i][k] * second_matrix[k][j]
                new_matrix[i][j] = total

        return new_matrix
