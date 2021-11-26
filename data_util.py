import os
import pickle
import matplotlib
import matplotlib.pyplot
import statistics
from data_processing import etlstream
from data_processing.etlstream import StreamFactory
from data_processing.normal.data_reader import DataReader
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from augmentation.domain_augmentation import DomainAugmentation
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
import torch
import cv2
import gryds
from skimage.exposure import equalize_adapthist
from scipy.ndimage import gaussian_filter


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
                        matplotlib.image.imsave(os.path.join(patient_img_dir, str(ind)+'img.png'), img, cmap='gray')
                        matplotlib.image.imsave(os.path.join(patient_img_dir, str(ind) + 'cnt.png'), cnt, cmap='gray')
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
    origins = [etlstream.Origin.SB, etlstream.Origin.ST11, etlstream.Origin.MC7]
    pixel_spacings_m = []
    for i in range(len(origins)):
        pixel_spacings = []
        stream = StreamFactory.create(origins[i], use_cache=True)
        for p in stream.stream_from_source():
            if p.pixel_spacing is None or not(p.pixel_spacing[0] == p.pixel_spacing[1]):
                print("Problem")
                print(p.pixel_spacing)
            pixel_spacings.append(p.pixel_spacing[0])
        pixel_spacings_m.append(pixel_spacings)
    plt.clf()
    plt.hist(pixel_spacings_m, 20, label=['SB', 'ST11', 'MC7'], density=True, histtype='bar', stacked=True)
    plt.legend()
    plt.savefig("pixel_spacing.png")
    print(statistics.median(pixel_spacings))


def try_da():
    path = "C:\dev\multi_vendor\out_filled\MC7"
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

                        p5, p95 = np.percentile(image, (5, 95))
                        image = (image - p5) / (p95 - p5)
                        image = equalize_adapthist(np.clip(image, 1e-5, 1), kernel_size=24)[..., np.newaxis]
                        print(image.shape)
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
                        plt.imshow(image.reshape(256, 256))
                        fig.add_subplot(2, 2, 4)
                        plt.imshow(contour)
                        plt.show()
                        a = torch.tensor(image, dtype=torch.float)
                        b = torch.tensor(img, dtype=torch.float)
                        print(torch.max(a), torch.max(b), torch.min(a), torch.min(b))
                        exit()

                except EOFError:
                    break


def asd():
    fig = plt.figure(figsize=(1, 3))
    origins = [etlstream.Origin.SB, etlstream.Origin.ST11, etlstream.Origin.MC7]
    titles = ['SunnyBrook', 'Stacom11', 'Miccai2017']
    n = 1
    numb = [320, 600, 5]
    plt.clf()
    for i in range(len(origins)):
        stream = StreamFactory.create(origins[i], use_cache=True)
        c = 0
        for p in stream.stream_from_source():
            for img in p.images:
                if c != numb[i]:
                    c += 1
                    continue
                img.get_image()
                print(c)
                img = img.image
                s =fig.add_subplot(1, 3, n)
                s.set_title(titles[i])
                plt.imshow(img, cmap='gray')
                n += 1
                break
            if c == numb[i]:
                break

    plt.show()

                # lp_contour = None
                # for cont in image.ground_truths:
                #     if cont.part == etlstream.Region.LP:
                #         lp_contour = cont.contour_mtx.astype(np.int32)
                #         cv2.drawContours(img, [lp_contour], 0, color=255, thickness=1)


def try_b_spline():
    path = "C:\dev\multi_vendor\out_filled\MC7"
    dr = DataReader(path)
    image = dr.x[0][5].reshape(256, 256)
    label = dr.y[0][5].reshape(256, 256)
    ia = False
    random_grid = np.random.rand(2, 7, 7)
    random_grid -= 0.5
    random_grid /= 12
    # Define a B-spline transformation object
    bspline_trf = gryds.BSplineTransformation(random_grid)
    interpolator = gryds.Interpolator(image[:, :], mode='reflect')

    interpolator_label = gryds.Interpolator(label[:, :], order=0, mode='constant')

    patch = interpolator.transform(bspline_trf)
    patch_label = interpolator_label.transform(bspline_trf)


    # matplotlib.image.imsave("C:\dev\multi_vendor\img1.png", image, cmap='gray')
    # matplotlib.image.imsave("C:\dev\multi_vendor\label1.png", label, cmap='gray')
    # matplotlib.image.imsave("C:\dev\multi_vendor\img2.png", patch, cmap='gray')
    # matplotlib.image.imsave("C:\dev\multi_vendor\label2.png", patch_label, cmap='gray')
    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title("a")
    plt.subplot(1, 4, 2)
    plt.imshow(label, cmap='gray')
    plt.axis('off')
    plt.title("b")
    plt.subplot(1, 4, 3)
    plt.imshow(patch, cmap='gray')
    plt.axis('off')
    plt.title("c")
    plt.subplot(1, 4, 4)
    plt.imshow(patch_label, cmap='gray')
    plt.axis('off')
    plt.title("d")

    plt.show()


def try_noise_aug():
    path = "C:\dev\multi_vendor\out_filled\MC7"
    dr = DataReader(path)
    patch = dr.x[0][5].reshape(256, 256)
    orig = np.copy(patch)
    label = dr.y[0][5].reshape(256, 256)
    ia = True
    if ia:
        intensity_shift = np.random.rand() * .1 - .05
        contrast_shift = np.random.rand() * 0.05 + 0.975

        patch += intensity_shift
        patch = np.sign(patch) * np.power(np.abs(patch), contrast_shift)

    blur = np.random.uniform()
    patch = gaussian_filter(patch, sigma=blur)

    p5, p95 = np.percentile(patch, (5, 95))
    patch = (patch - p5) / (p95 - p5)
    patch = equalize_adapthist(np.clip(patch, 1e-5, 1), kernel_size=24)[..., np.newaxis]
    patch += np.random.normal(scale=0.025, size=patch.shape)
    patch = patch.reshape(256,256)
    plt.subplot(1, 2, 1)
    plt.imshow(orig, cmap='gray')
    plt.axis('off')
    plt.title("a")
    plt.subplot(1, 2, 2)
    plt.imshow(patch, cmap='gray')
    plt.axis('off')
    plt.title("b")
    plt.show()


if __name__ == '__main__':
    # create_images_from_preprocessed_data("C:\dev\multi_vendor\out_filled\ST11")
    try_noise_aug()