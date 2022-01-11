import os
import pickle
import sys

import cv2
import numpy as np
import statistics
from data_processing import etlstream, patient
from data_processing.etlstream import StreamFactory
from scipy import ndimage
import random
from skimage.exposure import equalize_adapthist
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm


def run_prerocess(origin, filled):
    to_dir = os.path.join(sys.argv[1], origin.name)
    stream = StreamFactory.create(origin, use_cache=True)

    for p in tqdm(stream.stream_from_source()):
        pickle_file_path = os.path.join(to_dir, p.patient_id + ".p")
        patient_data = patient.Patient()
        for img in p.images:
            img.get_image()
            image = img.image
            left_ventricle_contour_mask = np.zeros(image.shape)

            if not origin == etlstream.Origin.ST19:
                ln_contour = None
                lp_contour = None

                for cont in img.ground_truths:
                    if cont.part == etlstream.Region.LN:
                        ln_contour = cont.contour_mtx.astype(np.int32)
                    elif cont.part == etlstream.Region.LP:
                        lp_contour = cont.contour_mtx.astype(np.int32)

                if not ln_contour is None and not lp_contour is None:
                    cv2.drawContours(left_ventricle_contour_mask, [lp_contour], 0, color=1, thickness=-1)
                    if not filled:
                        cv2.drawContours(left_ventricle_contour_mask, [ln_contour], 0, color=0, thickness=-1)
                    image = image.astype('float64')
                    if origin == etlstream.Origin.MC7:
                        image = np.rot90(image)
                        left_ventricle_contour_mask = np.rot90(left_ventricle_contour_mask)
                    rx = p.pixel_spacing[0] / 1.25
                    ry = p.pixel_spacing[1] / 1.25
                    image = ndimage.zoom(image, [rx, ry], order=1)
                    image *= 255.0 / image.max()
                    if not filled:
                        patient_data.add_data(image.astype('uint8'), cnt)
                    else:
                        cnt = left_ventricle_contour_mask.astype('uint8')
                        cnt = ndimage.zoom(cnt, [rx, ry], order=0)
                        res = np.where(cnt == 1)
                        if not (cnt[statistics.mean(res[0])][statistics.mean(res[1])] == 0 or np.count_nonzero(
                            cnt) < 100 or check_stdev(res) < 0.75):
                            image = crop_center(image)
                            cnt = crop_center(cnt)
                            patient_data.add_data(image, cnt)
            else:
                epi_mask = None
                for cont in img.ground_truths:
                    if type(cont) is etlstream.LVQuant:
                        epi_mask = cont.epi_mask
                if not epi_mask is None:
                    img.image = img.image.astype('float64')
                    rx = (p.pixel_spacing[0] * img.image.shape[0]) / 1.25
                    ry = (p.pixel_spacing[1] * img.image.shape[1]) / 1.25
                    img.image = ndimage.zoom(img.image, [rx, ry])
                    img.image *= 255.0 / img.image.max()
                    patient_data.add_data(img.image.astype('uint8'), epi_mask.astype('uint8'))

        if not patient_data.is_empty():
            if len(patient_data.images) > 20:
                l = random.sample(list(zip(patient_data.images, patient_data.contours)), 20)
                patient_data.images = list(map(lambda e: e[0], l))
                patient_data.contours = list(map(lambda e: e[1], l))
            patient_data.images = list(map(lambda x: histogram_eq(x), patient_data.images))
            os.makedirs(os.path.dirname(pickle_file_path), exist_ok=True)
            with(open(pickle_file_path, "wb")) as pickle_file:
                pickle.dump(patient_data, pickle_file)
    train_test_dev_split(to_dir)


def histogram_eq(img):
    p5, p95 = np.percentile(img, (5, 95))
    img = (img - p5) / (p95 - p5)
    img = equalize_adapthist(np.clip(img, 1e-5, 1), kernel_size=24)[..., np.newaxis]

    return img


def train_test_dev_split(path):
    train_path = os.path.join(path, 'train')
    dev_path = os.path.join(path, 'dev')
    test_path = os.path.join(path, 'test')
    os.mkdir(train_path)
    os.mkdir(dev_path)
    os.mkdir(test_path)

    pickle_file_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    train, test = train_test_split(pickle_file_names, test_size=0.2, random_state=13)
    train, dev = train_test_split(train, test_size=0.25, random_state=13)

    move_files(train, path, train_path)
    move_files(dev, path, dev_path)
    move_files(test, path, test_path)


def move_files(filenames, from_path, to_path):
    for filename in filenames:
        shutil.move(os.path.join(from_path, filename), os.path.join(to_path, filename))


def check_stdev(res):
    l = [statistics.stdev(res[0]), statistics.stdev(res[1])]
    s_std = min(l)
    l_std = max(l)

    return s_std / l_std


def crop_center(img):
    x, y = img.shape

    result_img = np.zeros((256, 256))
    x_axis_l = min(128, x // 2)
    y_axis_l = min(128, y // 2)
    result_img[128 - x_axis_l:128 + x_axis_l, 128 - y_axis_l:128 + y_axis_l] = img[x // 2 - x_axis_l:x // 2 + x_axis_l, y // 2 - y_axis_l:y // 2 + y_axis_l]

    return result_img


if __name__ == "__main__":
    random.seed(13)
    run_prerocess(etlstream.Origin.SB, True)
    run_prerocess(etlstream.Origin.MC7, True)
    #run_prerocess(etlstream.Origin.ST11, True)
