import os
import pickle
import numpy as np
import random


class DataReader:
    def __init__(self, path):
        file_names = [f for f in os.listdir(path) if os.path.isfile(f)]
        patient_file_paths = list(map(lambda f: os.path.join(path, f), file_names))
        self.x = []
        self.y = []
        self.size_dict = {}
        self.vertical_resolution = {}
        for patient_file_path in patient_file_paths:
            current_x = []
            current_y = []
            with (open(patient_file_path, "rb")) as patient_file:
                while True:
                    try:
                        patient_data = pickle.load(patient_file)
                        for ind, (img, cnt) in enumerate(zip(patient_data.images, patient_data.contours)):
                            if img.shape in self.size_dict.keys():
                                self.size_dict[img.shape] += 1
                            else:
                                self.size_dict[img.shape] = 1
                            result_img = np.zeros((256, 256))
                            result_cnt = np.zeros((256, 256))
                            x_axis_l = min(128, img.shape[0] // 2)
                            y_axis_l = min(128, img.shape[1] // 2)
                            result_img[128 - x_axis_l:128 + x_axis_l, 128 - y_axis_l:128 + y_axis_l] = img[img.shape[0] // 2 - x_axis_l:img.shape[0] // 2 + x_axis_l,img.shape[1] // 2 - y_axis_l:img.shape[1] // 2 + y_axis_l]
                            result_cnt[128 - x_axis_l:128 + x_axis_l, 128 - y_axis_l:128 + y_axis_l] = cnt[cnt.shape[0] // 2 - x_axis_l:cnt.shape[0] // 2 + x_axis_l,cnt.shape[1] // 2 - y_axis_l:cnt.shape[1] // 2 + y_axis_l]
                            current_x.append(result_img.reshape(256, 256, 1))
                            current_y.append(result_cnt.reshape(256, 256, 1) / 255.0)
                    except EOFError:
                        break
            if len(current_x) in self.vertical_resolution.keys():
                self.vertical_resolution[len(current_x)] += 1
            else:
                self.vertical_resolution[len(current_x)] = 1
            if len(current_x) > 15:
                l = random.sample(list(zip(current_x, current_y)), 15)
                current_x = list(map(lambda e: e[0], l))
                current_y = list(map(lambda e: e[1], l))
            self.x.append(current_x)
            self.y.append(current_y)