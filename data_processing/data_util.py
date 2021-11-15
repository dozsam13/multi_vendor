import os
import pickle
import shutil
import matplotlib
import sys
import matplotlib.pyplot
import cv2
import numpy as np
import statistics


def create_images_from_preprocessed_data(path):
    file_names = os.listdir(path)
    img_dir = os.path.join(path, 'img')
    os.mkdir(img_dir)
    img_dir = os.path.join(path, 'img')
    for file_name in file_names:
        patient_file_path = os.path.join(path, file_name)
        patient_img_dir = os.path.join(img_dir, file_name)
        os.mkdir(patient_img_dir)
        with (open(patient_file_path, "rb")) as patient_file:
            while True:
                try:
                    patient_data = pickle.load(patient_file)
                    for ind, (img, cnt) in enumerate(zip(patient_data.images, patient_data.contours)):
                        matplotlib.image.imsave(os.path.join(patient_img_dir, str(ind)+'img.png'), img)
                        matplotlib.image.imsave(os.path.join(patient_img_dir, str(ind) + 'cnt.png'), cnt)
                except EOFError:
                    break


def check_circles(path):
    file_names = os.listdir(path)
    c = 0
    for file_name in file_names:
        patient_file_path = os.path.join(path, file_name)
        c += 1
        if c < 50:
            continue
        elif c > 50:
            exit()
        print(file_name)
        l = []
        with (open(patient_file_path, "rb")) as patient_file:
            while True:
                try:
                    patient_data = pickle.load(patient_file)
                    for ind, (img, cnt) in enumerate(zip(patient_data.images, patient_data.contours)):
                        res = np.where(cnt == 255)
                        if cnt[statistics.mean(res[0])][statistics.mean(res[1])] == 0 or np.count_nonzero(cnt) < 100 or check_stdev(res) < 0.75:
                            print(ind)
                except EOFError:
                    break


def check_stdev(res):
    l = [statistics.stdev(res[0]), statistics.stdev(res[1])]
    s_std = min(l)
    l_std = max(l)

    return s_std / l_std

if __name__ == '__main__':
    #create_images_from_preprocessed_data(path=sys.argv[1])
    check_circles(path=sys.argv[1])