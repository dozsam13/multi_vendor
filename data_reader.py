import os
import pickle

import numpy as np


class DataReader:
    def __init__(self, path):
        patient_file_paths = list(map(lambda f: os.path.join(path, f), os.listdir(path)))
        self.x = []
        self.y = []
        for patient_file_path in patient_file_paths:
            with (open(patient_file_path, "rb")) as patient_file:
                while True:
                    try:
                        patient_data = pickle.load(patient_file)
                        self.x.extend(list(map(lambda x: np.repeat(x.reshape(256, 256, 1), 3, axis=2).astype('uint8'), patient_data.images)))
                        self.y.extend(list(map(lambda y: y.reshape(1, 256, 256).astype('uint8')/255.0, patient_data.contours)))

                    except EOFError:
                        break