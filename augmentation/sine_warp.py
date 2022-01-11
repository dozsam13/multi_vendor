import numpy as np
import math
import random


class SineWarp(object):
    def __init__(self, wave_length):
        self.wave_length = wave_length

    def __call__(self, img):
        l = random.randint(0, self.wave_length)
        img_output = np.zeros(img.shape, dtype=img.dtype)
        rows, cols = img.shape[0], img.shape[1]
        for i in range(rows):
            for j in range(cols):
                offset_x = int(l * math.sin(2 * 3.14 * i / 150))
                offset_y = int(l * math.cos(2 * 3.14 * j / 150))
                if i + offset_y < rows and j + offset_x < cols:
                    img_output[i, j, :] = img[(i + offset_y) % rows, (j + offset_x) % cols, :]
                else:
                    img_output[i, j, :] = 0

        return img_output
