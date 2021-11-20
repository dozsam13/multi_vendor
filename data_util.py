import os
import pickle
import matplotlib
import matplotlib.pyplot
import statistics
from data_processing import etlstream
from data_processing.etlstream import StreamFactory
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from augmentation.domain_augmentation import DomainAugmentation
import matplotlib.pyplot as plt


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


def resolution_checker():
    origins = [etlstream.Origin.ST11, etlstream.Origin.SB, etlstream.Origin.MC7]
    pixel_spacings = []
    for i in range(len(origins)):
        stream = StreamFactory.create(origins[i], use_cache=True)
        for p in stream.stream_from_source():
            if p.pixel_spacing is None or not(p.pixel_spacing[0] == p.pixel_spacing[1]):
                print("Problem")
                print(p.pixel_spacing)
            pixel_spacings.append(p.pixel_spacing[0])
    plt.clf()
    plt.hist(pixel_spacings, 20, facecolor='blue', alpha=0.3)
    plt.legend()
    plt.savefig("pixel_spacing.png")
    print(statistics.median(pixel_spacings))


def try_da():
    path = "C:\dev\multi_vendor\out_filled\SB"
    augmenter = DomainAugmentation()
    file_names = os.listdir(path)
    c = 0
    for file_name in file_names:
        if file_name == 'img':
            continue
        patient_file_path = os.path.join(path, file_name)
        with (open(patient_file_path, "rb")) as patient_file:
            while True:
                try:
                    patient_data = pickle.load(patient_file)
                    for ind, (image, contour) in enumerate(zip(patient_data.images, patient_data.contours)):
                        img, cnt = augmenter(image, contour)
                        img = img.reshape((256, 256))
                        cnt = cnt.reshape((256, 256))
                        fig = plt.figure(figsize=(8, 8))
                        fig.add_subplot(2, 2, 1)
                        plt.imshow(img)
                        print(np.max(img), np.min(img), np.max(cnt), np.min(cnt))
                        print(np.max(image), np.min(image), np.max(contour), np.min(contour))
                        fig.add_subplot(2, 2, 2)
                        plt.imshow(cnt)
                        fig.add_subplot(2, 2, 3)
                        plt.imshow(image)
                        fig.add_subplot(2, 2, 4)
                        plt.imshow(contour)
                        plt.show()
                        exit()

                except EOFError:
                    break



if __name__ == '__main__':
    try_da()