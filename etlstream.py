"""
This module provides the streaming of unified objects
from the different data sources.
"""
import os
import copy
import shutil
import pickle
import nibabel as nib
import numpy as np
import pandas as pd
from enum import Enum
import pydicom as dicom
from scipy.io import loadmat
import utils
import dicom_reader
import con_reader

logger = utils.get_logger(__name__)
nib_cache = {}


class Gender(Enum):
    X = 0  # unknown
    M = 1
    F = 2


class Origin(Enum):
    UNK = 0  # not defined
    VM = 1  # varosmajor
    SB = 2  # sunnybrook
    MC2 = 3  # miccai 2012
    MC7 = 4  # miccai 2017
    ST11 = 5  # stacom 2011
    ST19 = 6  # stacom 2019


class EndPhase(Enum):
    UNK = 0  # unknown
    SYS = 1  # systole
    DIA = 2  # diastole


class Region(Enum):
    UNK = 0  # unknown
    LN = 1  # left-endo
    LP = 2  # left-epi
    RN = 3  # right-endo
    RP = 4  # right-epi


class Pathology(Enum):
    UNK = 0  # unknown
    HFI = 1  # heart failure with infarction
    HF = 2  # heart failure wtihout infarction
    HYP = 3  # hypertrophy
    CGN = 4  # cogenital cases
    N = 5  # healthy


class Contour:

    def __init__(self, origin_type):
        self.origin_type = origin_type  # origin of the data source
        self.origin_path = None  # full path to the con file
        self.patient_id = None  # an id for the patient, e.g.: study_id + serial_id
        self.length = 0  # length of the contour curve
        self.slice = -1  # the slice on the short-axis where the contour corresponds (-1 if not known)
        self.frame = -1  # frame in the cardiac cycle (-1 if not available)
        self.phase = EndPhase.UNK  # systole or diastole (or unknown)
        self.part = Region.UNK  # e.g.: left-endocardial (can be unknown as well)
        self.contour_mtx = None  # numpy matrix with shape (N, 2)
        self.corresponding_image = None
        self.area = None  # the area of the contour

    @classmethod
    def from_contourmtx(cls, contour_mtx):
        c = cls(Origin.UNK)
        c.contour_mtx = contour_mtx
        return c


class LVQuant:

    def __init__(self, origin_type):
        self.origin_type = origin_type  # origin of the data source
        self.origin_path = None  # full path to the con file
        self.patient_id = None  # an id for the patient, e.g.: study_id + serial_id
        self.frame = -1  # frame in the cardiac cycle (-1 if not available)
        self.phase = EndPhase.UNK  # systole or diastole (or unknown)
        self.corresponding_image = None

        self.epi_mask = None
        self.endo_mask = None
        self.area_cavity = None  # cavity area in mm^2
        self.area_myo = None  # myocardium area in mm^2
        self.dims = None  # dimensions of cavity of three directions (IS-AL, I-A, and IL-AS) in mm
        self.rwt = None  # regional wall thickness (mm), starting at the AS segment in counter clockwise direction
        self.pixel_spacing = None


class Image:

    def __init__(self, origin_type):
        self.origin_type = origin_type  # origin of the data source
        self.origin_path = None  # full path to the dcm file
        self.patient_id = None  # an id for the patient, e.g.: study_id + serial_id
        self.size = None  # height and width tuple
        self.slice = -1  # the slice on the short-axis where the contour corresponds (-1 if not known)
        self.frame = -1  # frame in the cardiac cycle (-1 if not available)
        self.phase = EndPhase.UNK  # systole or diastole (or unknown)
        self.has_gt = False  # shows if the image has any contour, mask etc.
        self.image = None  # numpy matrix representing the image
        self.ground_truths = []  # reference for the stored ground truth values if any

    def __getstate__(self):
        state_to_pickle = self.__dict__.copy()
        del state_to_pickle['image']  # image should not be cached
        return state_to_pickle

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.image = None

    def get_image(self):
        if self.image is None:
            self.load_image()
        return self.image

    def load_image(self):  # use get_image instead of this
        if type(self.origin_path) is int:
            self.image = np.zeros((250, 250))
            self.size = (250, 250)
            logger.error("Wrong folder path at {}".format(self.patient_id))
            return

        if self.origin_path.endswith('.dcm'):
            ds = dicom.dcmread(self.origin_path)
            self.image = ds.pixel_array
            self.size = self.image.shape
        elif self.origin_path.endswith('.mat'):
            mat = loadmat(self.origin_path)
            self.image = mat['image'][:, :, self.frame]
            self.size = self.image.shape
        elif self.origin_path.endswith('.nii.gz'):
            if self.origin_path in nib_cache:
                imgs = nib_cache[self.origin_path]['imgs']
                nib_cache[self.origin_path]['num'] += 1
            else:
                imgs = nib.load(self.origin_path).get_fdata()
                nib_cache[self.origin_path] = {'imgs': imgs, 'num': 1}
            self.image = np.copy(imgs[:, :, self.slice])
            self.size = self.image.shape
            if nib_cache[self.origin_path]['num'] >= imgs.shape[2]:
                del nib_cache[self.origin_path]


class Patient:

    def __init__(self, origin_type):
        self.origin_type = origin_type  # origin of the data source
        self.origin_path = None  # full path to the serial folder
        self.patient_id = None  # an id for the patient, e.g.: study_id + serial_id
        self.path_group = Pathology.UNK
        self.gender = Gender.X
        self.height = None
        self.weight = None
        self.pixel_spacing = None  # MRI specific data
        self.slice_thicknes = None
        self.gap = None
        self.vol_indices = None  # volume data and indices in a dictionary
        self.ground_truths = []  # can be contours, masks, left-ventricle quantification etc.
        self.images = []

    def append_image(self, image):
        self.images.append(image)

    def append_gt(self, gt_data):
        self.ground_truths.append(gt_data)

    def __str__(self):
        def append_vol_info(repr, name, abbreviation):
            if abbreviation in self.vol_indices.keys():
                return repr + "    " + name + ": {} \n".format(self.vol_indices[abbreviation])
            else:
                return repr

        repr = "Origin type: {} \n".format(self.origin_type.name)
        repr = repr + "Origin path: {} \n".format(self.origin_path)
        repr = repr + "Patient ID: {} \n".format(self.patient_id)
        repr = repr + "Pathology group: {} \n".format(self.path_group.name)
        repr = repr + "Height: {} \n".format("Unknown" if self.height is None else self.height)
        repr = repr + "Weight: {} \n".format("Unknown" if self.weight is None else self.weight)
        repr = repr + "Pixel spacing: {} <> {} \n".format(self.pixel_spacing[0], self.pixel_spacing[1])
        repr = repr + "Slice thicknes: {} \n".format(self.slice_thicknes)
        repr = repr + "Gap between slices: {} \n".format(self.gap)
        repr = repr + "VOLUME DATA \n"
        if self.vol_indices is None:
            repr = repr + "    Not available ... \n"
        else:
            repr = append_vol_info(repr, 'Left ventricular end diastole volume', 'lved')
            repr = append_vol_info(repr, 'Left ventricular end systole volume', 'lves')
            repr = append_vol_info(repr, 'Left ventricular Stroke volume', 'lvsv')
            repr = append_vol_info(repr, 'Left ventricular end diastole index', 'lved_i')
            repr = append_vol_info(repr, 'Left ventricular end systole index', 'lves_i')
            repr = append_vol_info(repr, 'Left ventricular Stroke index', 'lvsv_i')

            repr = append_vol_info(repr, 'Right ventricular end diastole volume', 'rved')
            repr = append_vol_info(repr, 'Right ventricular end systole volume', 'rves')
            repr = append_vol_info(repr, 'Right ventricular Stroke volume', 'rvsv')
            repr = append_vol_info(repr, 'Right ventricular end diastole index', 'rved_i')
            repr = append_vol_info(repr, 'Right ventricular end systole index', 'rves_i')
            repr = append_vol_info(repr, 'Right ventricular Stroke index', 'rvsv_i')

        return repr


class StreamFactory:

    @staticmethod
    def __mapping(origin_type):
        mp = {
            Origin.VM: StreamVM,
            Origin.SB: StreamSB,
            Origin.MC2: StreamMC2,
            Origin.MC7: StreamMC7,
            Origin.ST11: StreamST11,
            Origin.ST19: StreamST19
        }
        return mp[origin_type]

    @staticmethod
    def create(origin_type, segment_to_read=None, use_cache=False):
        return StreamFactory.__mapping(origin_type)(segment_to_read, use_cache)

    @staticmethod
    def segment(origin_type, num_of_segments, use_cache=False):
        return StreamFactory.__mapping(origin_type).create_segments(num_of_segments, use_cache)

    @staticmethod
    def recreate_cache(origin_type):
        stream = StreamFactory.__mapping(origin_type)(None, False)
        stream.recreate_cache()


class Stream:

    def __init__(self, segment_to_read, use_cache):
        self.use_cache = use_cache
        self.root = None
        self.cache = None
        self.segment = segment_to_read  # the files or folders to read by a process

    def __iter__(self):  # should use a generator
        if self._cache_exists() and self.use_cache:
            print('Streaming from cache.')
            return self.stream_from_cache()
        else:
            return self.stream_from_source()

    def recreate_cache(self):
        if self._cache_exists():
            shutil.rmtree(self.cache)
        os.mkdir(self.cache)

    def stream_from_source(self):
        pass

    def set_source_folders(self, cache_folder, raw_folder):
        self.root = raw_folder
        self.cache = cache_folder

    def stream_from_cache(self):
        # calculate the number of patients for checking progress
        cntr = 0
        if self.segment is None:
            patients = os.listdir(self.cache)
        else:
            patients = [p for p in self.segment]
        all_patient = len(patients)

        # stream the saved pickle files
        for p in patients:
            p_path = os.path.join(self.cache, p)
            cntr += 1
            with open(p_path, 'br') as f:
                patient = pickle.load(f)
            utils.progress_bar(cntr, all_patient, 50)
            yield patient

    def _cache_exists(self):
        if os.path.exists(self.cache):
            return True
        return False

    @staticmethod
    def _create_segments(root, num_of_segments):
        folders = os.listdir(root)
        segments = []
        segment_size = len(folders) / num_of_segments
        low = 0
        for i in range(1, num_of_segments):
            upper = int(i * segment_size)
            segments.append(folders[low: upper])
            low = upper
        segments.append(folders[low:])
        return segments

    @staticmethod
    def create_segments(num_of_segments, use_cache):
        pass


class StreamVM(Stream):

    root_fd = "sources/varosmajor/sa"
    cache_fd = "cache/vm"

    def __init__(self, segment_to_read, use_cache):
        super(StreamVM, self).__init__(segment_to_read, use_cache)
        self.cntr = 0
        self.root = os.path.join(utils.DB_PATH, StreamVM.root_fd)
        self.cache = os.path.join(utils.DB_PATH, StreamVM.cache_fd)

    def stream_from_source(self):
        # calculate the number of patients for checking progress
        self.cntr = 0
        if self.segment is None:
            patients = os.listdir(self.root)
        else:
            patients = self.segment

        # visit all the patients
        all_patient = len(patients)

        for patient in patients:  # at the level of a patient folder
            patient_path = os.path.join(self.root, patient)
            serials = os.listdir(patient_path)
            self.cntr += 1
            for ser in serials:  # in a patient folder, more examination with different serial id is possible
                ser_path = os.path.join(patient_path, ser)
                dcm_folder = os.path.join(ser_path, 'imgs')
                con_file = os.path.join(ser_path, 'contour.con')
                dcm = dicom_reader.DCMreaderVM(dcm_folder)
                con = con_reader.CONreaderVM(con_file)

                pixel_spacing, slice_thicknes, weight, height, gender = con.get_volume_data()
                contours = con.get_hierarchical_contours()

                p = Patient(Origin.VM)
                p.origin_path = ser_path
                p.patient_id = patient + '_' + ser
                p.pixel_spacing = pixel_spacing
                p.height = height
                p.weight = weight
                p.slice_thicknes = slice_thicknes
                p.gap = abs(dcm.get_slicelocation(1, 0) - dcm.get_slicelocation(0, 0)) - slice_thicknes
                if gender == 'F':
                    p.gender = Gender.F
                elif gender == 'M':
                    p.gender = Gender.M

                for slice in range(dcm.num_slices):
                    for frame in range(dcm.num_frames):
                        # first create the image object
                        img_mtx = dcm.get_image(slice, frame)
                        img_path = dcm.get_dcm_path(slice, frame)

                        image = Image(Origin.VM)
                        image.origin_path = img_path
                        image.patient_id = p.patient_id
                        image.size = img_mtx.shape
                        image.slice = slice
                        image.frame = frame

                        # find the corresponding contours for the image if exists
                        if slice in contours.keys():
                            if frame in contours[slice].keys():
                                for mode in contours[slice][frame]:
                                    contour = Contour(Origin.VM)
                                    contour.origin_path = con_file
                                    contour.patient_id = p.patient_id
                                    contour.slice = slice
                                    contour.frame = frame

                                    if mode == 'red':  # left-endo
                                        contour.part = Region.LN
                                    elif mode == 'green':  # left-epi
                                        contour.part = Region.LP
                                    elif mode == 'yellow':  # right-endo
                                        contour.part = Region.RN
                                    elif mode == 'blue':  # right-epi
                                        contour.part = Region.RP
                                    else:
                                        contour.part = Region.UNK
                                    # create a matrix from the contour points
                                    x = contours[slice][frame][mode][0]['x']
                                    y = contours[slice][frame][mode][0]['y']
                                    N = len(x)
                                    contour_mtx = np.zeros((N, 2))
                                    contour_mtx[:, 0] = np.array(x)
                                    contour_mtx[:, 1] = np.array(y)
                                    contour.length = N
                                    contour.contour_mtx = copy.deepcopy(contour_mtx)

                                    unique_contour = copy.deepcopy(contour)
                                    p.append_gt(unique_contour)
                                    unique_contour.corresponding_image = image
                                    image.ground_truths.append(unique_contour)
                                    image.has_gt = True

                        p.append_image(image)
                pickle_path = self.cache
                pickle_path = os.path.join(pickle_path, p.patient_id + '.pickle')
                with open(pickle_path, 'wb') as f:
                    pickle.dump(p, f)  # create cache
                utils.progress_bar(self.cntr, all_patient, 20)
                yield p

    @staticmethod
    def create_segments(num_of_segments, use_cache):
        if not use_cache:
            root = os.path.join(utils.DB_PATH, StreamVM.root_fd)
        else:
            root = os.path.join(utils.DB_PATH, StreamVM.cache_fd)
        return Stream._create_segments(root, num_of_segments)


class StreamSB(Stream):

    root_fd = "sources/sunnybrook"
    cache_fd = "cache/sb"

    def __init__(self, segment_to_read, use_cache):
        super(StreamSB, self).__init__(segment_to_read, use_cache)

        self.cntr = 0
        self.root = os.path.join(utils.DB_PATH, StreamSB.root_fd)
        self.cache = os.path.join(utils.DB_PATH, StreamSB.cache_fd)

    def stream_from_source(self):
        self.cntr = 0
        img_root = os.path.join(self.root, "SCD_DeidentifiedImages")
        if self.segment is None:
            img_folders = os.listdir(img_root)
        else:
            img_folders = self.segment
        # full roots for the images and the contours
        contour_root = os.path.join(self.root, "scd_manualcontours")
        # read the mapping file (patientID -> originalID)
        df = pd.read_excel(os.path.join(self.root, "scd_patientdata.xlsx"), index_col=0, usecols=[0, 1, 2])
        # iterate through the image folders and find the corresponding contours
        for img_folder in img_folders:
            self.cntr += 1
            img_path = os.path.join(img_root, img_folder)
            # find the corresponding folder name of the contours
            record = df.loc[img_folder]
            temp = record['OriginalID'].split('-')
            if len(temp[-1]) < 2:
                temp[-1] = "0" + temp[-1]  # excel does not contain the 0s
            cntr_folder = '-'.join(temp)
            cntr_path = os.path.join(contour_root, cntr_folder)
            # read the images and the contours
            dcm = dicom_reader.DCMreaderSB(img_path)
            con = con_reader.CONreaderSB(cntr_path)
            p = Patient(Origin.SB)
            p.origin_path = img_path
            p.patient_id = img_folder
            p.gender = Gender.F if record['Gender'] == 'Female' else Gender.M
            p.height = None  # unfortunately not known
            p.weight = None
            p.pixel_spacing = dcm.get_pixelspacing()
            p.slice_thicknes = dcm.get_slicethickness()
            p.gap = dcm.get_gap()

            for k, v in dcm.get_imagepaths().items():
                image = Image(Origin.SB)
                image.origin_path = v['path']
                image.patient_id = p.patient_id
                image.slice = v['slice']
                image.frame = v['frame']

                if k in con.get_contours().keys():
                    for part, cntr in con.get_contours()[k].items():
                        contour = Contour(Origin.SB)
                        contour.origin_path = cntr['path']
                        contour.patient_id = p.patient_id
                        contour.slice = image.slice
                        contour.frame = image.frame
                        contour.length = cntr['contour'].shape[0]
                        contour.contour_mtx = cntr['contour']
                        contour.corresponding_image = image
                        if part == 'i':  # i indicates left-endo
                            contour.part = Region.LN
                        elif part == 'o':
                            contour.part = Region.LP
                        image.ground_truths.append(contour)
                        p.append_gt(contour)
                    image.has_gt = True
                p.append_image(image)
            pickle_path = os.path.join(self.cache, img_folder + '.pickle')
            with open(pickle_path, 'wb') as f:
                pickle.dump(p, f)  # create cache
            utils.progress_bar(self.cntr, len(img_folders), 50)
            yield p

    @staticmethod
    def create_segments(num_of_segments, use_cache):
        if not use_cache:
            root = os.path.join(utils.DB_PATH, StreamSB.root_fd)
            root = os.path.join(root, "SCD_DeidentifiedImages")
        else:
            root = os.path.join(utils.DB_PATH, StreamSB.cache_fd)
        return Stream._create_segments(root, num_of_segments)


class StreamMC2(Stream):

    root_fd = "sources/miccai2012"
    cache_fd = "cache/miccai2012"

    def __init__(self, segment_to_read, use_cache):
        super(StreamMC2, self).__init__(segment_to_read, use_cache)

        self.cntr = 0
        self.root = os.path.join(utils.DB_PATH, StreamMC2.root_fd)
        self.cache = os.path.join(utils.DB_PATH, StreamMC2.cache_fd)

    def stream_from_source(self):
        self.cntr = 0
        # training folder path
        training_path = os.path.join(self.root, "TrainingSet")
        if self.segment is None:
            patient_folders = os.listdir(training_path)
        else:
            patient_folders = self.segment
        # iterate through the patients
        for patient_folder in patient_folders:
            self.cntr += 1
            patient_path = os.path.join(training_path, patient_folder)
            folders = os.listdir(patient_path)
            img_folder = [fd for fd in folders if fd.find('dicom') != -1][0]
            con_folder = [fd for fd in folders if fd.find('contours') != -1][0]
            img_path = os.path.join(patient_path, img_folder)
            con_path = os.path.join(patient_path, con_folder)
            # read the images and the contours
            dcm = dicom_reader.DCMreaderMC2(img_path)
            con = con_reader.CONreaderMC2(con_path)
            p = Patient(Origin.MC2)
            p.origin_path = patient_path
            p.patient_id = patient_folder
            p.gender = dcm.get_gender()
            p.height = dcm.get_height()
            p.weight = dcm.get_weight()
            p.pixel_spacing = dcm.get_pixelspacing()
            p.slice_thicknes = dcm.get_slicethickness()
            p.gap = dcm.get_gap()

            for k, v in dcm.get_imagepaths().items():
                image = Image(Origin.MC2)
                image.origin_path = v['path']
                image.patient_id = p.patient_id
                image.slice = v['slice']
                image.frame = v['frame']

                if k in con.get_contours().keys():
                    for part, cntr in con.get_contours()[k].items():
                        contour = Contour(Origin.MC2)
                        contour.origin_path = cntr['path']
                        contour.patient_id = p.patient_id
                        contour.slice = image.slice
                        contour.frame = image.frame
                        contour.length = cntr['contour'].shape[0]
                        contour.contour_mtx = cntr['contour']
                        contour.corresponding_image = image
                        if part == 'i':  # i indicates right-endo
                            contour.part = Region.RN
                        elif part == 'o':
                            contour.part = Region.RP
                        image.ground_truths.append(contour)
                        p.append_gt(contour)
                    image.has_gt = True
                p.append_image(image)
            pickle_path = os.path.join(self.cache, patient_folder + '.pickle')
            with open(pickle_path, 'wb') as f:
                pickle.dump(p, f)  # create cache
            utils.progress_bar(self.cntr, len(patient_folders), 50)
            yield p

    @staticmethod
    def create_segments(num_of_segments, use_cache):
        if not use_cache:
            root = os.path.join(utils.DB_PATH, StreamMC2.root_fd)
            root = os.path.join(root, "TrainingSet")
        else:
            root = os.path.join(utils.DB_PATH, StreamMC2.cache_fd)
        return Stream._create_segments(root, num_of_segments)


class StreamMC7(Stream):

    root_fd = "sources/miccai2017"
    cache_fd = "cache/miccai2017"

    def __init__(self, segment_to_read, use_cache):
        super(StreamMC7, self).__init__(segment_to_read, use_cache)

        self.cntr = 0
        self.root = os.path.join(utils.DB_PATH, StreamMC7.root_fd)
        self.cache = os.path.join(utils.DB_PATH, StreamMC7.cache_fd)

    def stream_from_source(self):
        self.cntr = 0
        # training folder path
        training_path = os.path.join(self.root, "training")
        if self.segment is None:
            patient_folders = os.listdir(training_path)
        else:
            patient_folders = self.segment
        # iterate through the patients
        for patient_folder in patient_folders:
            self.cntr += 1
            patient_path = os.path.join(training_path, patient_folder)
            # read the images and the contours
            dcm = dicom_reader.DCMreaderMC7(patient_path)
            con = con_reader.CONreaderMC7(patient_path)
            with open(os.path.join(patient_path, 'Info.cfg'), 'rt') as cfg:
                temp = cfg.read()
                meta = dict(pair.split(': ') for pair in temp.split('\n')[0:-1])

            p = Patient(Origin.MC7)
            p.origin_path = patient_path
            p.patient_id = patient_folder
            p.height = float(meta['Height'])
            p.weight = float(meta['Weight'])
            p.gap = 0.0
            p.pixel_spacing = dcm.get_pixelspacing()
            p.slice_thicknes = dcm.get_slicethickness()

            for frame in dcm.get_imagepaths():
                for slc in range(dcm.get_imagepaths()[frame]['num_slices']):
                    image = Image(Origin.MC7)
                    image.origin_path = dcm.get_imagepaths()[frame]['path']
                    image.patient_id = p.patient_id
                    image.slice = slc
                    image.frame = frame
                    image.phase = EndPhase.SYS if int(meta['ES']) == frame else EndPhase.DIA

                    if slc in con.get_contours():
                        if frame in con.get_contours()[slc]:
                            cntr = con.get_contours()[slc][frame]
                            for part in cntr['contours']:
                                contour = Contour(Origin.MC7)
                                contour.origin_path = cntr['path']
                                contour.patient_id = p.patient_id
                                contour.phase = image.phase
                                contour.slice = image.slice
                                contour.frame = image.frame
                                contour.length = cntr['contours'][part].shape[0]
                                contour.contour_mtx = cntr['contours'][part]
                                contour.corresponding_image = image
                                if part == 'RN':  # i indicates right-endo
                                    contour.part = Region.RN
                                elif part == 'LN':
                                    contour.part = Region.LN
                                elif part == 'LP':
                                    contour.part = Region.LP
                                image.ground_truths.append(contour)
                                p.append_gt(contour)
                                image.has_gt = True
                    p.append_image(image)
            pickle_path = os.path.join(self.cache, patient_folder + '.pickle')
            with open(pickle_path, 'wb') as f:
                pickle.dump(p, f)  # create cache
            utils.progress_bar(self.cntr, len(patient_folders), 20)
            yield p

    @staticmethod
    def create_segments(num_of_segments, use_cache):
        if not use_cache:
            root = os.path.join(utils.DB_PATH, StreamMC7.root_fd)
            root = os.path.join(root, "training")
        else:
            root = os.path.join(utils.DB_PATH, StreamMC7.cache_fd)
        return Stream._create_segments(root, num_of_segments)


class StreamST11(Stream):

    root_fd = "sources/stacom2011"
    cache_fd = "cache/st11"

    def __init__(self, segment_to_read, use_cache):
        super(StreamST11, self).__init__(segment_to_read, use_cache)

        self.cntr = 0
        self.root = os.path.join(utils.DB_PATH, StreamST11.root_fd)
        self.cache = os.path.join(utils.DB_PATH, StreamST11.cache_fd)

    def stream_from_source(self):
        self.cntr = 0
        # training folder path
        training_path = os.path.join(self.root, "CAP_challenge_training_set")
        if self.segment is None:
            patient_folders = os.listdir(training_path)
        else:
            patient_folders = self.segment
        patient_folders = [p for p in patient_folders if os.path.isdir(os.path.join(training_path, p))]
        # iterate through the patients
        for patient_folder in patient_folders:
            self.cntr += 1
            patient_path = os.path.join(training_path, patient_folder)
            # read the images and the contours
            dcm = dicom_reader.DCMreaderST(patient_path)
            con = con_reader.CONreaderST(patient_path)
            p = Patient(Origin.ST11)
            p.origin_path = patient_path
            p.patient_id = patient_folder
            p.gap = dcm.get_gap()
            p.pixel_spacing = dcm.get_pixelspacing()
            p.slice_thicknes = dcm.get_slicethickness()
            for slice in dcm.get_imagepaths():
                for frame in dcm.get_imagepaths()[slice]:
                    image = Image(Origin.ST11)
                    image.origin_path = dcm.get_imagepaths()[slice][frame]['path']
                    image.patient_id = p.patient_id
                    image.slice = slice
                    image.frame = frame

                    if slice in con.get_contours():
                        if frame in con.get_contours()[slice]:
                            for part in ['LN', 'LP']:
                                cntr = con.get_contours()[slice][frame]
                                contour = Contour(Origin.ST11)
                                contour.origin_path = cntr['path']
                                contour.patient_id = p.patient_id
                                contour.slice = image.slice
                                contour.frame = image.frame
                                contour.length = cntr['contours'][part].shape[0]
                                contour.contour_mtx = cntr['contours'][part]
                                contour.corresponding_image = image
                                if part == 'LN':  # i indicates left-endo
                                    contour.part = Region.LN
                                elif part == 'LP':
                                    contour.part = Region.LP
                                image.ground_truths.append(contour)
                                p.append_gt(contour)
                                image.has_gt = True
                    p.append_image(image)

            pickle_path = os.path.join(self.cache, patient_folder + '.pickle')
            with open(pickle_path, 'wb') as f:
                pickle.dump(p, f)  # create cache
            utils.progress_bar(self.cntr, len(patient_folders), 50)
            yield p

    @staticmethod
    def create_segments(num_of_segments, use_cache):
        if not use_cache:
            root = os.path.join(utils.DB_PATH, StreamST11.root_fd)
            root = os.path.join(root, "CAP_challenge_training_set")
        else:
            root = os.path.join(utils.DB_PATH, StreamST11.cache_fd)
        return Stream._create_segments(root, num_of_segments)


class StreamST19(Stream):

    root_fd = "sources/stacom2019"
    cache_fd = "cache/st19"

    def __init__(self, segment_to_read, use_cache):
        super(StreamST19, self).__init__(segment_to_read, use_cache)

        self.cntr = 0
        self.root = os.path.join(utils.DB_PATH, StreamST19.root_fd)
        self.cache = os.path.join(utils.DB_PATH, StreamST19.cache_fd)

    def stream_from_source(self):
        self.cntr = 0
        # training folder path
        training_path = os.path.join(self.root, "TrainingData_LVQuan19")
        if self.segment is None:
            patient_datas = os.listdir(training_path)
        else:
            patient_datas = self.segment
        patient_datas = [p for p in patient_datas if p.endswith('.mat')]
        # iterate through the patients
        for patient_file in patient_datas:
            self.cntr += 1
            patient_path = os.path.join(training_path, patient_file)
            # read the images and the contours
            mat = loadmat(patient_path)
            p = Patient(Origin.ST19)
            p.origin_path = patient_path
            p.patient_id = patient_file[:-4]
            p.pixel_spacing = mat['pix_spacing'][0, 0]
            for frame in range(20):
                image = Image(Origin.ST19)
                image.origin_path = p.origin_path
                image.patient_id = p.patient_id
                image.frame = frame
                image.image = mat['image'][:, :, frame]
                image.size = image.image.shape

                lvq = LVQuant(Origin.ST19)
                lvq.origin_path = p.origin_path
                lvq.patient_id = p.patient_id
                lvq.frame = image.frame
                lvq.epi_mask = mat['epi'][:, :, frame]
                lvq.endo_mask = mat['endo'][:, :, frame]
                lvq.area_cavity = mat['areas'][0, frame]
                lvq.area_myo = mat['areas'][1, frame]
                lvq.dims = mat['dims'][:, frame]
                lvq.rwt = mat['rwt'][:, frame]
                lvq.phase = mat['lv_phase']  # systole: 1, diastole: 0
                lvq.corresponding_image = image
                image.ground_truths.append(lvq)
                p.append_gt(lvq)
                image.has_gt = True
                p.append_image(image)

            pickle_path = os.path.join(self.cache, p.patient_id + '.pickle')
            with open(pickle_path, 'wb') as f:
                pickle.dump(p, f)  # create cache
            utils.progress_bar(self.cntr, len(patient_datas), 10)
            yield p

    @staticmethod
    def create_segments(num_of_segments, use_cache):
        if not use_cache:
            root = os.path.join(utils.DB_PATH, StreamST19.root_fd)
            root = os.path.join(root, "TrainingData_LVQuan19")
        else:
            root = os.path.join(utils.DB_PATH, StreamST19.cache_fd)
        return Stream._create_segments(root, num_of_segments)
