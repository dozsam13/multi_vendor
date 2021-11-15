from util import get_logger
from pydicom.pixel_data_handlers.util import apply_modality_lut
from pydicom.errors import InvalidDicomError
import pydicom as dicom
import nibabel as nib
import numpy as np
import os

logger = get_logger(__name__)


class DCMreaderVM:

    def __init__(self, folder_name):
        '''
        Reads in the dcm files in a folder which corresponds to a patient.
        It follows carefully the physical slice locations and the frames in a hearth cycle.
        It does not matter if the location is getting higher or lower. 
        '''
        self.num_slices = 0
        self.num_frames = 0
        self.broken = False
        images = []
        slice_locations = []
        file_paths = []        

        dcm_files = sorted(os.listdir(folder_name))
        dcm_files = [d for d in dcm_files if len(d.split('.')[-2]) < 4]
        if len(dcm_files) == 0:  # sometimes the order number is missing at the end
            dcm_files = sorted(os.listdir(folder_name))

        for file in dcm_files:

            if file.find('.dcm') != -1:
                try:
                    temp_ds = dicom.dcmread(os.path.join(folder_name, file))
                    images.append(temp_ds.pixel_array)
                    slice_locations.append(temp_ds.SliceLocation)
                    file_paths.append(os.path.join(folder_name, file))
                except:
                    self.broken = True
                    return
        
        current_sl = -1
        frames = 0
        increasing = False
        indices = []
        for idx, slice_loc in enumerate(slice_locations):
            if abs(slice_loc - current_sl) > 0.01:  # this means a new slice is started
                self.num_slices += 1
                self.num_frames = max(self.num_frames, frames)
                frames = 0
                indices.append(idx)

                if (slice_loc - current_sl) > 0.01:
                    increasing = True
                else:
                    increasing = False
                
                current_sl = slice_loc
            frames += 1

        if self.num_slices != 0 and self.num_frames != 0:
            self.load_matrices(images, indices, increasing, slice_locations, file_paths)
        else:
            logger.warning("There are no frames. This folder should be deleted. Path: {}".format(folder_name))
        self.num_images = len(images)

    def load_matrices(self, images, indices, increasing, slice_locations, file_paths):
        size_h, size_w = images[0].shape
        self.dcm_images = np.ones((self.num_slices, self.num_frames, size_h, size_w))
        self.dcm_slicelocations = np.ones((self.num_slices, self.num_frames, 1))
        self.dcm_file_paths = np.zeros((self.num_slices, self.num_frames), dtype=object)

        for i in range(len(indices) - 1):

            for idx in range(indices[i], indices[i + 1]):
                slice_idx = (i if increasing else (len(indices) - 1 - i))
                frame_idx = idx - indices[i]
                if images[idx].shape == self.dcm_images[slice_idx, frame_idx, :, :].shape:
                    self.dcm_images[slice_idx, frame_idx, :, :] = images[idx]
                    self.dcm_slicelocations[slice_idx, frame_idx, 0] = slice_locations[idx]
                    self.dcm_file_paths[slice_idx, frame_idx] = file_paths[idx]
                else:
                    logger.error('Wrong shape at {}'.format(file_paths[idx]))

        for idx in range(indices[-1], len(images)):
            slice_idx = (len(indices) - 1 if increasing else 0)
            frame_idx = idx - indices[-1]
            if self.dcm_images.shape[1] == frame_idx:
                logger.info(file_paths[idx])
            if images[idx].shape == self.dcm_images[slice_idx, frame_idx, :, :].shape:
                self.dcm_images[slice_idx, frame_idx, :, :] = images[idx]
                self.dcm_slicelocations[slice_idx, frame_idx, 0] = slice_locations[idx]
                self.dcm_file_paths[slice_idx, frame_idx] = file_paths[idx]
            else:
                logger.error('Wrong shape at {}'.format(file_paths[idx]))

    def get_image(self, slice, frame):
        return self.dcm_images[slice, frame, :, :]
    
    def get_slicelocation(self, slice, frame):
        return self.dcm_slicelocations[slice, frame, 0]

    def get_dcm_path(self,slice, frame):
        return self.dcm_file_paths[slice, frame]


class DCMreaderVMsiemens:  # new siemens machine

    def __init__(self, folder_name):
        '''
        Reads in the dcm files in a folder which corresponds to a patient.
        It follows carefully the physical slice locations and the frames in a hearth cycle.
        It does not matter if the location is getting higher or lower. 
        '''
        self.images_unordered = list()  # contains the images and additional data_processing
        
        dcm_files = os.listdir(folder_name)
        for dcm_fl in dcm_files:
            # ignore if not a dcm file
            if not dcm_fl.endswith('.dcm'):
                continue
            # read the image
            path = os.path.join(folder_name, dcm_fl)
            try:
                dcm = dicom.dcmread(path, force=True)
            except InvalidDicomError:
                logger.error("Invalid dicom {}".format(path))
                continue  # if no valid dcm continue with the next file
            # access the required fields
            try:
                _, _, pos_z = self.__get_dcmdata(dcm, "ImagePositionPatient")
                timestamp = self.__get_dcmdata(dcm, "ContentTime")
                image_pixels = dcm.pixel_array  # to get a numpy array
            except AttributeError as ex:
                logger.error("Accessing elements in dcm file {} yielded: {}".format(path, ex))
                continue
            # modality transformation on pixels (mainly for Philips)
            image = apply_modality_lut(image_pixels, dcm)
            # sequence is ordered according to ImagePatientPosition
            data = (pos_z, timestamp, image)
            self.images_unordered.append(data)
        # ordering the images
        _images_ = sorted(self.images_unordered, key=lambda data: data[0])  # apex to base
        _images_ = self.__group_frames(_images_)
        self.__order_frames(_images_)
        # create the image dictionary
        self.images = list()
        for group in _images_:
            img_in_slice = []
            for data in group:
                img_in_slice.append(data[2])
            self.images.append(img_in_slice)

    @staticmethod
    def __get_dcmdata(dcm, name):
        elmnt = dcm.data_element(name)
        if elmnt != None:
            return elmnt.value
        else:
            raise AttributeError("Element {} does not exist.".format(name))
    
    @staticmethod
    def __group_frames(images):
        # frames in the same slice is one group
        # images - tuple of (pos, timestamp, image)
        grouped = list()
        pos = None
        for img in images:
            img_pos = img[0]
            if pos is None or abs(pos - img_pos) > 1e-6:
                grouped.append([img])
                pos = img_pos
            else:
                grouped[-1].append(img)
        return grouped
    
    @staticmethod
    def __order_frames(images):
        # follow the cardiac cycle
        for cycle in images:
            cycle.sort(key=lambda data: data[1])

    def get_image(self, slice, frame):
        return self.images[slice][frame]


class DCMreaderSB:  # sunnybrook

    def __init__(self, folder_name):
        """
        Reads in a patient images from the Sunnybrook data_processing set.
        Creates a dictionary as pairs of image number and the corresponding image path.
        """
        self.folder_name = folder_name  # name of the folder with the dicom images (with full path)
        self.images = None

        self.__build_dictionary()
        self.__calculate_sliceframe()
        self.__find_metadata()

    def __build_dictionary(self):
        self.images = {}
        
        for folder in os.listdir(self.folder_name):
            path = os.path.join(self.folder_name, folder)
            img_names = os.listdir(path)
            for img_name in img_names:
                if img_name.lower().endswith('.dcm'):
                    img_path = os.path.join(path, img_name)
                    # find the important values in the img_name
                    temp = img_name.split('_')
                    date = temp[-2]
                    img_num = int(temp[-1].split('.')[0])

                    if not(img_num in self.images.keys()):
                        self.images[img_num] = {'path': img_path, '_date_': date, 'slice': -1, 'frame': -1}
                    else:
                        if self.images[img_num]['_date_'] < date:
                            self.images[img_num] = {'path': img_path, '_date_': date, 'slice': -1, 'frame': -1}
    
    def __find_metadata(self):
        path = list(self.images.values())[0]['path']
        ds1 = dicom.dcmread(path)
        path = list(self.images.values())[1]['path']
        ds2 = dicom.dcmread(path)
        self.slicethickness = ds1.SliceThickness
        self.pixelspacing = list(map(float, ds1.PixelSpacing))
        spacingbetweenslices = abs(ds2.SliceLocation - ds1.SliceLocation)
        self.gap = spacingbetweenslices - self.slicethickness

    def __calculate_sliceframe(self):
        slice_location = None
        slice = 0
        frame = 0
        for key in sorted(list(self.images.keys())):
            ds = dicom.dcmread(self.images[key]['path'])
            if slice_location is None or abs(slice_location - ds.SliceLocation) > 0.0001:
                slice_location = ds.SliceLocation
                slice += 1
                frame = 1
            else:
                frame += 1
            self.images[key]['slice'] = slice
            self.images[key]['frame'] = frame

    def get_imagepaths(self):
        return self.images

    def get_pixelspacing(self):
        return self.pixelspacing
    
    def get_slicethickness(self):
        return self.slicethickness

    def get_gap(self):
        return self.gap


class DCMreaderMC2:  # miccai

    def __init__(self, folder_name):
        """
        Reads in a patient images from the MICCAI 2012 data_processing set.
        Creates a dictionary as pairs of image number and the corresponding image path.
        """
        self.folder_name = folder_name  # name of the folder with the dicom images (with full path)
        self.images = None

        self.__build_dictionary()
        self.__calculate_sliceframe()
        self.__find_metadata()

    def __build_dictionary(self):
        self.images = {}

        for img_name in os.listdir(self.folder_name):
            img_path = os.path.join(self.folder_name, img_name)
            if img_name.lower().endswith('.dcm'):
                # find the image number in the img_name
                temp = img_name.split('-')[1]
                temp = temp.split('.dcm')[0]
                img_num = int(temp)

                if not (img_num in self.images.keys()):
                    self.images[img_num] = {'path': img_path, 'slice': -1, 'frame': -1}

    def __find_metadata(self):
        path = list(self.images.values())[0]['path']
        ds1 = dicom.dcmread(path)

        self.slicethickness = ds1.SliceThickness
        self.pixelspacing = list(map(float, ds1.PixelSpacing))
        self.gap = self.spacingbetweenslices - self.slicethickness
        self.gender = ds1.PatientSex     # F, M
        self.weight = ds1.PatientWeight  # in kg
        try:
            self.height = ds1.PatientSize    # in meter
        except:
            logger.error('Missing values in {}'.format(path))
            self.height = 1.7

    def __calculate_sliceframe(self):
        slice_location = None
        self.spacingbetweenslices = None
        slice = 0
        frame = 0
        for key in sorted(list(self.images.keys())):
            ds = dicom.dcmread(self.images[key]['path'])
            if slice_location is None or abs(slice_location - ds.SliceLocation) > 0.0001:
                if not(slice_location is None) and self.spacingbetweenslices is None:
                    self.spacingbetweenslices = abs(slice_location - ds.SliceLocation)
                slice_location = ds.SliceLocation
                slice += 1
                frame = 1
            else:
                frame += 1
            self.images[key]['slice'] = slice
            self.images[key]['frame'] = frame

    def get_imagepaths(self):
        return self.images

    def get_pixelspacing(self):
        return self.pixelspacing

    def get_slicethickness(self):
        return self.slicethickness

    def get_gap(self):
        return self.gap

    def get_height(self):
        return self.height

    def get_weight(self):
        return self.weight

    def get_gender(self):
        return self.gender


class DCMreaderMC7:  # miccai

    def __init__(self, folder_name):
        """
        Reads in a patient images from the MICCAI 2017 data_processing set.
        Creates a dictionary as pairs of image number and the corresponding image path.
        """
        self.folder_name = folder_name  # name of the folder with the dicom images (with full path)
        self.images = None
        self.pixelspacing = None
        self.slicethickness = None

        self.__build_dictionary()

    def __build_dictionary(self):
        self.images = {}

        image_files = os.listdir(self.folder_name)
        for img_f in image_files:
            if img_f.find('_gt') == -1 and img_f.find('_frame') != -1:
                img_path = os.path.join(self.folder_name, img_f)
                # read some information from the file name
                temp = img_f.split('_')
                frame = int(temp[1][5:7])
                nib_loaded = nib.load(img_path)
                imgs = nib_loaded.get_fdata()
                num_slices = imgs.shape[2]
                self.images[frame] = {'num_slices': num_slices, 'path': img_path}
                self.pixelspacing = nib_loaded.header.get_zooms()[0:2]
                self.slicethickness = nib_loaded.header.get_zooms()[2]

    def get_imagepaths(self):
        return self.images

    def get_pixelspacing(self):
        return self.pixelspacing

    def get_slicethickness(self):
        return self.slicethickness


class DCMreaderST:  # stacom2011

    def __init__(self, folder_name):
        """
        Reads in a patient images from the STACOM 2011 data_processing set.
        Creates a dictionary as pairs of image number and the corresponding image path.
        """
        self.folder_name = folder_name  # name of the folder with the dicom images (with full path)
        self.images = None

        self.__find_images()
        self.__meatadata()

    def __find_images(self):
        self.images = {}
        dcmcntr_files = os.listdir(self.folder_name)
        for dcmcntr in dcmcntr_files:
            if dcmcntr.endswith('.dcm') and dcmcntr.find('_SA') != -1:
                temp = dcmcntr.split('_')
                temp = temp[1:]
                slice = int(temp[0][2:])
                temp = temp[1].split('.')[0]
                frame = int(temp[2:])
                if not(slice in self.images):
                    self.images[slice] = {}
                self.images[slice][frame] = {
                    'path': os.path.join(self.folder_name, dcmcntr),
                }

    def __meatadata(self):
        img_path = self.images[1][0]['path']
        ds1 = dicom.dcmread(img_path)
        img_path = self.images[2][0]['path']
        ds2 = dicom.dcmread(img_path)
        self.slicethickness = ds1.SliceThickness
        self.pixelspacing = list(map(float, ds1.PixelSpacing))
        try:
            spacingbetweenslices = abs(ds2.SliceLocation - ds1.SliceLocation)
            self.gap = spacingbetweenslices - self.slicethickness
        except AttributeError:
            self.gap = 0.0
            logger.warning("SliceLocation is unknown in {}".format(img_path))

    def get_imagepaths(self):
        return self.images

    def get_pixelspacing(self):
        return self.pixelspacing

    def get_slicethickness(self):
        return self.slicethickness

    def get_gap(self):
        return self.gap
