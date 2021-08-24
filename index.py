from pathlib import Path
import numpy as np
import cv2 as cv

identity_kernel = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]])

outline_kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])

blur_kernel = np.array([[0.0625, 0.125, 0.0625],
                        [0.125, 0.25, 0.125],
                        [0.0625, 0.125, 0.0625]])


class ImageProcessing:

    def __init__(self):
        self.img_a_path = Path('skate.png')
        self.img_b_path = Path('skate_2.jpg')
        self.red_grey_value = 0.299
        self.green_grey_value = 0.587
        self.blue_grey_value = 0.114

        self.__check_file()
        self.__load_original_img()

    def __check_file(self):
        if not self.img_a_path.is_file():
            raise FileNotFoundError('File a not found!')

        if not self.img_b_path.is_file():
            raise FileNotFoundError('File b not found!')

    def __load_original_img(self):
        self.original_a_img = cv.imread('skate.png')
        self.original_b_img = cv.imread('skate_2.jpg')

    def __format_invalid_pixel_result(self, pixel):
        if pixel >= 255:
            return 255
        elif pixel <= 0:
            return 0
        return pixel

    def __sum_pixel_value(self, pixel_a, pixel_b):
        if pixel_a >= 255 or pixel_b >= 255:
            return 255
        result = (int(pixel_a) + int(pixel_b))

        return self.__format_invalid_pixel_result(result)

    def __sub_pixel_value(self, pixel_a, pixel_b):
        result = (int(pixel_a) - int(pixel_b))

        return self.__format_invalid_pixel_result(result)

    def __divide_pixel_value(self, pixel_a, pixel_b):
        if pixel_b == 0:
            return 0
        result = (int(pixel_a) / int(pixel_b))

        return self.__format_invalid_pixel_result(result)

    def __threshold_value(self, pixel):
        if pixel < 127:
            return 0
        return 255

    def get_original_a_img(self):
        return self.original_a_img

    def get_original_b_img(self):
        return self.original_b_img

    def red_channel(self):
        img = self.original_a_img.copy()

        height, width, channels = img.shape

        for i in range(width):
            for j in range(height):
                img[i, j][0] = 0
                img[i, j][1] = 0

        return img

    def blue_channel(self):
        img = self.original_a_img.copy()

        height, width, channels = img.shape

        for i in range(width):
            for j in range(height):
                img[i, j][1] = 0
                img[i, j][2] = 0

        return img

    def green_channel(self):
        img = self.original_a_img.copy()

        height, width, channels = img.shape

        for i in range(width):
            for j in range(height):
                img[i, j][0] = 0
                img[i, j][2] = 0

        return img

    def img_sum(self):
        img_a = self.original_a_img.copy()
        img_b = self.original_b_img.copy()

        (row, column) = img_a.shape[0:2]

        for i in range(row):
            for j in range(column):
                r = self.__sum_pixel_value(img_a[i, j][0], img_b[i, j][0])
                g = self.__sum_pixel_value(img_a[i, j][1], img_b[i, j][1])
                b = self.__sum_pixel_value(img_a[i, j][2], img_b[i, j][2])
                img_a[i, j] = [r, g, b]

        return img_a

    def img_subtract(self):
        img_a = self.original_a_img.copy()
        img_b = self.original_b_img.copy()

        (row, column) = img_a.shape[0:2]

        for i in range(row):
            for j in range(column):
                r = self.__sub_pixel_value(img_a[i, j][0], img_b[i, j][0])
                g = self.__sub_pixel_value(img_a[i, j][1], img_b[i, j][1])
                b = self.__sub_pixel_value(img_a[i, j][2], img_b[i, j][2])
                img_a[i, j] = [r, g, b]

        return img_a

    def img_multiply(self):
        img_a = self.original_a_img.copy()
        img_b = self.original_b_img.copy()

        (row, column) = img_a.shape[0:2]

        for i in range(row):
            for j in range(column):
                r = self.__format_invalid_pixel_result((int(img_a[i, j][0]) * int(img_b[i, j][0])))
                g = self.__format_invalid_pixel_result((int(img_a[i, j][1]) * int(img_b[i, j][1])))
                b = self.__format_invalid_pixel_result((int(img_a[i, j][2]) * int(img_b[i, j][2])))
                img_a[i, j] = [r, g, b]

        return img_a

    def img_divide(self):
        img_a = self.original_a_img.copy()
        img_b = self.original_b_img.copy()

        (row, column) = img_a.shape[0:2]

        for i in range(row):
            for j in range(column):
                r = self.__divide_pixel_value(img_a[i, j][0], img_b[i, j][0])
                g = self.__divide_pixel_value(img_a[i, j][1], img_b[i, j][1])
                b = self.__divide_pixel_value(img_a[i, j][2], img_b[i, j][2])
                img_a[i, j] = [r, g, b]

        return img_a

    def grey_scale_ari(self):
        img_a = self.original_a_img.copy()

        (row, column) = img_a.shape[0:2]

        for i in range(row):
            for j in range(column):
                img_a[i, j] = sum(img_a[i, j]) * 0.33

        return img_a

    def grey_scale_pond(self):
        img_a = self.original_a_img.copy()

        (row, column) = img_a.shape[0:2]

        for i in range(row):
            for j in range(column):
                r = self.red_grey_value * img_a[i, j][0]
                g = self.green_grey_value * img_a[i, j][1]
                b = self.blue_grey_value * img_a[i, j][2]
                img_a[i, j] = r + g + b

        return img_a

    def img_threshold(self):
        img_a = self.original_a_img.copy()

        (row, column) = img_a.shape[0:2]

        for i in range(row):
            for j in range(column):
                r = self.__threshold_value(img_a[i, j][0])
                g = self.__threshold_value(img_a[i, j][1])
                b = self.__threshold_value(img_a[i, j][2])
                img_a[i, j] = [r, g, b]

        return img_a

    def img_convolution(self, kernel, padding=0, strides=1):
        img = self.original_a_img.copy()
        img_conv = np.zeros(img.shape)

        img_height, img_width, img_channels = img.shape
        kernel_height, kernel_width = kernel.shape

        height = int(kernel_height)
        width = int(kernel_width)

        for i in range(height, img_height - height):
            for j in range(width, img_width - width):
                sum = 0
                for k in range(kernel_height):
                    for m in range(kernel_width):
                        sum = sum + kernel[k, m] * img[i - height - k][j - width + m]
                img_conv[i, j] = sum
        return img_conv


img_processing = ImageProcessing()
image_a = img_processing.original_a_img.copy()
image_b = img_processing.original_b_img.copy()

cv.imshow('Algorithm Result', img_processing.img_convolution())

# ret, thresh_one = cv.threshold(image_a, 127, 255, cv.THRESH_BINARY)
# cv.imshow('OpenCV Result', cv.multiply(image_a, image_b))
# cv.imshow('OpenCV Result', cv.divide(image_a, image_b))
# cv.imshow('OpenCV Result', cv.add(image_a, image_b))
# cv.imshow('OpenCV Result', cv.subtract(image_a, image_b))
# cv.imshow('OpenCV Result', cv.cvtColor(image_a, cv.COLOR_BGR2GRAY))
# cv.imshow('OpenCV Result', thresh_one)
cv.imshow('OpenCV Result', cv.filter2D(src=image_a, ddepth=-1, kernel=outline_kernel))
cv.waitKey(0)
cv.destroyAllWindows()
exit()
