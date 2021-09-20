import os
import pickle
from data_processing import etlstream
import numpy as np


class GanDataReader:
    vendors = {etlstream.Origin.SB: 'SB', etlstream.Origin.MC7: 'MC7', etlstream.Origin.ST11: 'ST11'}

    def __init__(self, path, sources):
        self.x = []
        self.y = []
        self.vendor = []
        for source_index, source in enumerate(sources):
            current_x = []
            current_y = []
            current_vendor = []
            p = os.path.join(path, GanDataReader.vendors[source])
            patient_file_paths = list(map(lambda f: os.path.join(p, f), os.listdir(p)))
            for patient_file_path in patient_file_paths:
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
                                current_vendor.append(source_index)
                        except EOFError:
                            break
            self.x.append(current_x)
            self.y.append(current_y)
            self.vendor.append(current_vendor)