import os
import pickle
import numpy as np


class DataReader:
    def __init__(self, path):
        patient_file_paths = list(map(lambda f: os.path.join(path, f), os.listdir(path)))
        self.x = []
        self.y = []

        for patient_file_path in patient_file_paths:
            current_x = []
            current_y = []
            with (open(patient_file_path, "rb")) as patient_file:
                while True:
                    try:

                        patient_data = pickle.load(patient_file)
                        for ind, (img, cnt) in enumerate(zip(patient_data.images, patient_data.contours)):
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
            self.x.append(current_x)
            self.y.append(current_y)