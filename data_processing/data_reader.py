import os
import pickle
from data_processing import etlstream


class DataList:
    def __init__(self):
        self.x = []
        self.y = []
        self.vendor = []

    def __init(self, x, y, vendor):
        self.x = x
        self.y = y
        self.vendor = vendor


class DataReader:
    vendors = {etlstream.Origin.SB: 'SB', etlstream.Origin.MC7: 'MC7', etlstream.Origin.ST11: 'ST11', etlstream.Origin.MC2: 'MC2'}

    def __init__(self, path, sources):
        self.path = path
        self.sources = sources
        self.train = DataList()
        self.dev = DataList()
        self.test = DataList()

    def read(self):
        for source_index, source in enumerate(self.sources):
            source_path = os.path.join(self.path, DataReader.vendors[source])
            self.read_source(source_path, source_index)

    def read_source(self, source_path, source_index):
        splits = ['train', 'dev', 'test']
        datalists = [self.train, self.dev, self.test]
        for i in range(len(splits)):
            source_split_path = os.path.join(source_path, splits[i])
            current_x, current_y = self.read_split(source_split_path)
            datalists[i].x.extend(current_x)
            datalists[i].y.extend(current_y)
            datalists[i].vendor.extend([source_index for i in range(len(current_x))])

    def read_split(self, source_split_path):
        patient_file_paths = list(map(lambda f: os.path.join(source_split_path, f), os.listdir(source_split_path)))
        patient_file_paths = [f for f in patient_file_paths if os.path.isfile(f)]
        current_x = []
        current_y = []
        for patient_file_path in patient_file_paths:
            with (open(patient_file_path, "rb")) as patient_file:
                while True:
                    try:
                        patient_data = pickle.load(patient_file)
                        for ind, (img, cnt) in enumerate(zip(patient_data.images, patient_data.contours)):
                            current_x.append(img.reshape(256, 256, 1))
                            current_y.append(cnt.reshape(256, 256, 1))
                    except EOFError:
                        break
        return current_x, current_y

    def get_all(self):
        x = self.train.x + self.dev.x + self.test.x
        y = self.train.y + self.dev.y + self.test.y
        vendor = self.train.vendor + self.dev.vendor + self.test.vendor

        return DataList(x, y, vendor)
