from utils import get_logger
from copy import deepcopy
import nibabel as nib
import numpy as np
import cv2
import os

logger = get_logger(__name__)


class CONreaderVM:

    def __init__(self, file_name):
        """
        Reads in a con file and saves the curves grouped according to its corresponding slice, frame and place.
        Finds the tags necessary to calculate the volume metrics.
        """
        self.file_name = file_name
        self.container = []
        self.contours = None

        con_tag = "XYCONTOUR"  # start of the contour data
        stop_tag = "POINT"     # if this is available, prevents from reading unnecessary lines
        volumerelated_tags = [
            'Study_id=',
            'Field_of_view=',
            'Image_resolution=',
            'Slicethickness=',
            'Patient_weight=',
            'Patient_height',
            'Study_description=',
            'Patient_gender='
        ]

        self.volume_data = {
            volumerelated_tags[0]: None, 
            volumerelated_tags[1]: None, 
            volumerelated_tags[2]: None,
            volumerelated_tags[3]: None,
            volumerelated_tags[4]: None,
            volumerelated_tags[5]: None,
            volumerelated_tags[6]: None,
            volumerelated_tags[7]: None
        }

        con = open(file_name, 'rt')
        
        def find_volumerelated_tags(line):
            for tag in volumerelated_tags:
                if line.find(tag) != -1:
                    value = line.split(tag)[1]  # the place of the tag will be an empty string, second part: value
                    self.volume_data[tag] = value
        
        def mode2colornames(mode):
            if mode == 0:
                return 'red'     # red means left (color means the same as in the official software)
            elif mode == 1:
                return 'green'   # left but contains the myocardium
            elif mode == 2:
                return 'blue'    # right (epi)
            elif mode == 5:
                return 'yellow'  # right (endo)
            else:
                logger.warning('Unknown mode {}'.format(mode))
                return 'other'

        def find_xycontour_tag():
            line = con.readline()
            find_volumerelated_tags(line)
            while line.find(con_tag) == -1 and line.find(stop_tag) == -1 and line != "":
                line = con.readline()
                find_volumerelated_tags(line)
            return line

        def identify_slice_frame_mode():
            line = con.readline()
            splitted = line.split(' ')
            return int(splitted[0]), int(splitted[1]), mode2colornames(int(splitted[2]))

        def number_of_contour_points():
            line = con.readline()
            return int(line)

        def read_contour_points(num):
            contour = []
            for _ in range(num):
                line = con.readline()
                xs, ys = line.split(' ')
                contour.append((float(xs), float(ys)))  # unfortubately x and y are interchanged
            return contour

        line = find_xycontour_tag()
        while line.find(stop_tag) == -1 and line != "":

            slice, frame, mode = identify_slice_frame_mode()
            num = number_of_contour_points()
            contour = read_contour_points(num)
            self.container.append((slice, frame, mode, contour))
            line = find_xycontour_tag()

        con.close()
        return

    def get_contours(self):
        return self.container

    def get_hierarchical_contours(self):
        # if it is not initializedyet, then create it
        if self.contours is None:

            self.contours = {}
            for item in self.container:
                slice = item[0]
                frame = item[1]   # frame in a hearth cycle
                mode = item[2]    # mode can be red, green, yellow
                contour = item[3]

                # rearrange the contour
                d = {'x': [], 'y': []}
                for point in contour:
                    d['x'].append(point[0])
                    d['y'].append(point[1])

                if not(slice in self.contours.keys()):
                    self.contours[slice] = {}

                if not(frame in self.contours[slice].keys()):
                    self.contours[slice][frame] = {}

                if not(mode in self.contours[slice][frame].keys()):
                    self.contours[slice][frame][mode] = []

                    self.contours[slice][frame][mode].append(d)

        return self.contours

    def contour_iterator(self, deep=True):
        self.get_hierarchical_contours()
        for slice, frame_level in self.contours.items():
            for frame, mode_level in frame_level.items():
                if deep:
                    mode_level_cp = deepcopy(mode_level)
                else:
                    mode_level_cp = mode_level
                yield slice, frame, mode_level_cp

    def get_volume_data(self):
        # process field of view
        fw_string = self.volume_data['Field_of_view=']
        sizexsize_mm = fw_string.split('x')  # variable name shows the format
        size_h = float(sizexsize_mm[0])
        size_w = float(sizexsize_mm[1].split(' mm')[0])  # I cut the _mm ending

        # process image resolution
        img_res_string = self.volume_data['Image_resolution=']
        sizexsize = img_res_string.split('x')
        res_h = float(sizexsize[0])
        res_w = float(sizexsize[1])

        # process slice thickness
        width_string = self.volume_data['Slicethickness=']
        width_mm = width_string.split(' mm')
        width = float(width_mm[0])

        # process weight
        weight_string = self.volume_data['Patient_weight=']
        weight_kg = weight_string.split(' kg')
        weight = float(weight_kg[0])

        # process height
        # Unfortunately, patient height is not always available.
        # Study description can help in that case but its form changes heavily.
        if 'Patient_height=' in self.volume_data.keys():  
            height_string = self.volume_data['Patient_height=']
            height = height_string.split(" ")[0]
        else:
            height_string = str(self.volume_data['Study_description='])
            height = ''
            for char in height_string:
                if char.isdigit():
                    height += char
        if height == '':
            logger.warning('Unknown height in con file {}'.format(self.file_name))
            height = 178
        else:
            try:
                height = float(height)
            except ValueError:
                height = 165
                logger.error(' Wrong height format in con file {}'.format(self.file_name))

        # gender
        gender = self.volume_data['Patient_gender='].strip(' ').strip('\n').strip('\r').strip(' ')
        
        return (size_h/res_h, size_w/res_w), width, weight, height, gender


class CONreaderSB:

    def __init__(self, folder_name):
        """
        Reads the contour points of all the curves given for a patient.
        """
        self.folder_name = folder_name  # path to the folder with contours for a patient
        self.contours = None  # a dictionary with img number (id) and the cor. contours
    
    def __full_path(self):
        temp = os.path.join(self.folder_name, "contours-manual")
        self.contours_root = os.path.join(temp, "IRCCI-expert")
        self.contours = {}
    
    def __read_coordinates(self, file_path):
        points = []
        with open(file_path, 'rt') as con_file:
            con_str = con_file.read()  # x, y coord. pairs are separated by \n -s
        xy_pairs = con_str.split('\n')
        for xy_pair in xy_pairs:
            xy_pair = xy_pair.split(' ')
            if len(xy_pair) == 2:
                points.append((float(xy_pair[0]), float(xy_pair[1])))
        return np.array(points)  # shape: (N, 2)
    
    def __build_contours_dict(self):
        contour_files = os.listdir(self.contours_root)
        for cf in contour_files:
            cf_path = os.path.join(self.contours_root, cf)
            # read some information from the file name
            temp = cf.split('-')
            img_num = int(temp[2])
            part = temp[3].split('contour')[0]

            # read the points
            cntr_mtx = self.__read_coordinates(cf_path)
            
            if img_num in self.contours.keys():
                self.contours[img_num][part] = {'contour': cntr_mtx, 'path': cf_path}
            else:
                self.contours[img_num] = {part: {'contour': cntr_mtx, 'path': cf_path}}
    
    def get_contours(self):
        if self.contours is None:
            self.__full_path()
            self.__build_contours_dict()
        return self.contours


class CONreaderMC2:

    def __init__(self, folder_name):
        """
        Reads the contour points of all the curves given for a patient.
        """
        self.folder_name = folder_name  # path to the folder with contours for a patient
        self.contours = None  # a dictionary with img number (id) and the cor. contours

    def __read_coordinates(self, file_path):
        points = []
        with open(file_path, 'rt') as con_file:
            con_str = con_file.read()  # x, y coord. pairs are separated by \n -s
        xy_pairs = con_str.split('\n')
        for xy_pair in xy_pairs:
            xy_pair = xy_pair.split(' ')
            if len(xy_pair) == 2:
                points.append((float(xy_pair[0]), float(xy_pair[1])))
        return np.array(points)  # shape: (N, 2)

    def __build_contours_dict(self):
        self.contours = {}
        contour_files = os.listdir(self.folder_name)
        for cf in contour_files:
            cf_path = os.path.join(self.folder_name, cf)
            # read some information from the file name
            temp = cf.split('-')
            img_num = int(temp[1])
            part = temp[2].split('contour')[0]

            # read the points
            cntr_mtx = self.__read_coordinates(cf_path)

            if img_num in self.contours.keys():
                self.contours[img_num][part] = {'contour': cntr_mtx, 'path': cf_path}
            else:
                self.contours[img_num] = {part: {'contour': cntr_mtx, 'path': cf_path}}

    def get_contours(self):
        if self.contours is None:
            self.__build_contours_dict()
        return self.contours


class CONreaderMC7:

    def __init__(self, folder_name):
        """
        Reads the contour points of all the curves given for a patient.
        """
        self.folder_name = folder_name  # path to the folder with contours for a patient
        self.contours = None  # a dictionary with img number (id) and the cor. contours

    def __find_contours(self, contour):
        empty = True
        contours = {}

        def seek_contour_and_add2dict(mask, name):
            if mask.sum() > 0:
                cntrs, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                contours[name] = cntrs[0].squeeze(1)
                return empty and False
            return empty and True

        contour = (contour - contour.min()) / (contour.max() + 1e-5) * 255
        contour = contour.astype(np.uint8)
        # finding the regions
        _, mask_ln = cv2.threshold(contour, 240, 255, cv2.THRESH_BINARY)
        _, mask_lp = cv2.threshold(contour, 150, 255, cv2.THRESH_BINARY)
        _, mask_all = cv2.threshold(contour, 70, 255, cv2.THRESH_BINARY)
        mask_rn = mask_all - mask_lp
        empty = seek_contour_and_add2dict(mask_ln, 'LN')
        empty = seek_contour_and_add2dict(mask_lp, 'LP')
        empty = seek_contour_and_add2dict(mask_rn, 'RN')
        return empty, contours

    def __build_contours_dict(self):
        self.contours = {}
        contour_files = os.listdir(self.folder_name)
        for cf in contour_files:
            if cf.find('_gt') != -1:
                cf_path = os.path.join(self.folder_name, cf)
                # read some information from the file name
                temp = cf.split('_')
                frame = int(temp[1][5:])
                contours = nib.load(cf_path).get_fdata()
                for slc in range(contours.shape[2]):
                    empty, cntrs = self.__find_contours(contours[:, :, slc])
                    if not empty:
                        if not(slc in self.contours):
                            self.contours[slc] = {}
                        self.contours[slc][frame] = {'path': cf_path, 'contours': cntrs}

    def get_contours(self):
        if self.contours is None:
            self.__build_contours_dict()
        return self.contours


class CONreaderST:

    def __init__(self, folder_name):
        """
        Reads the contour points of all the curves given for a patient.
        """
        self.folder_name = folder_name  # path to the folder with contours for a patient
        self.contours = None  # a dictionary with slice, frame and contour (+ path)
    
    def __find_lv_contour(self, myocardial_mask):
        # gives the inner contour of the mask
        # if only one contour can be found it gives None, no ventricle found
        # if the whole image black it gives None, no left ventricle found
        gray = cv2.cvtColor(myocardial_mask, cv2.COLOR_BGR2GRAY)
        cntrs, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if len(cntrs) < 2:
            return None
        cntrs_ordered = sorted(cntrs, key=cv2.contourArea)
        return cntrs_ordered[0].squeeze(1), cntrs_ordered[-1].squeeze(1)

    def __read_segmentations(self):
        # reads in the segmentation images for SA 
        # reads the slice and the phase (frame)
        self.contours = {}
        file_names = os.listdir(self.folder_name)
        for file in file_names:
            if file.endswith('.png') and file.find('_SA') != -1:
                path = os.path.join(self.folder_name, file)
                temp = file.split('_')
                temp = temp[1:]
                slice = int(temp[0][2:])
                temp = temp[1].split('.')[0]
                frame = int(temp[2:])
                mask = cv2.imread(path)
                contours = self.__find_lv_contour(mask)
                if not(contours is None):
                    if not(slice in self.contours):
                        self.contours[slice] = {}
                    self.contours[slice][frame] = {
                        'path': path,
                        'contours': {
                            'LN': contours[0],
                            'LP': contours[1]
                        }
                    }
    
    def get_contours(self):
        if self.contours is None:
            self.__read_segmentations()
        return self.contours
