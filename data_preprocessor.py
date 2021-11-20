import os
import pickle
import sys

import cv2
import numpy as np
import statistics
from data_processing import etlstream
import patient
from data_processing.etlstream import StreamFactory
from scipy import ndimage


def run_prerocess(origin, filled):
    to_dir = os.path.join(sys.argv[1], origin.name)
    stream = StreamFactory.create(origin, use_cache=True)

    img_shapes = {}
    for p in stream.stream_from_source():
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
                    cv2.drawContours(left_ventricle_contour_mask, [lp_contour], 0, color=255, thickness=-1)
                    if not filled:
                        cv2.drawContours(left_ventricle_contour_mask, [ln_contour], 0, color=0, thickness=-1)
                    image = image.astype('float64')
                    rx = p.pixel_spacing[0] / 1.25
                    ry = p.pixel_spacing[1] / 1.25
                    image = ndimage.zoom(image, [rx, ry], order=1)
                    image *= 255.0 / image.max()
                    if not filled:
                        patient_data.add_data(image.astype('uint8'), cnt)
                    else:
                        cnt = left_ventricle_contour_mask.astype('uint8')
                        cnt = ndimage.zoom(cnt, [rx, ry], order=0)
                        res = np.where(cnt == 255)
                        if not (len(np.unique(cnt)) == 2):
                            print(p.patient_id)
                        if not (cnt[statistics.mean(res[0])][statistics.mean(res[1])] == 0 or np.count_nonzero(
                            cnt) < 100 or check_stdev(res) < 0.75):
                            x_axis_l = min(128, image.shape[0] // 2)
                            y_axis_l = min(128, image.shape[1] // 2)
                            result_img = np.zeros((256, 256)).astype('uint8')
                            result_cnt = np.zeros((256, 256)).astype('uint8')
                            result_img[128 - x_axis_l:128 + x_axis_l, 128 - y_axis_l:128 + y_axis_l] = image[image.shape[0] // 2 - x_axis_l:image.shape[0] // 2 + x_axis_l,image.shape[1] // 2 - y_axis_l:image.shape[1] // 2 + y_axis_l]
                            result_cnt[128 - x_axis_l:128 + x_axis_l, 128 - y_axis_l:128 + y_axis_l] = cnt[cnt.shape[0] // 2 - x_axis_l:cnt.shape[0] // 2 + x_axis_l,cnt.shape[1] // 2 - y_axis_l:cnt.shape[1] // 2 + y_axis_l]
                            patient_data.add_data(result_img, result_cnt)

                if not image.shape in img_shapes.keys():
                    img_shapes[image.shape] = 0
                img_shapes[image.shape] += 1
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

                    if not img.image.shape in img_shapes.keys():
                        img_shapes[img.image.shape] = 0
                    img_shapes[img.image.shape] += 1

        if not patient_data.is_empty():
            os.makedirs(os.path.dirname(pickle_file_path), exist_ok=True)
            with(open(pickle_file_path, "wb")) as pickle_file:
                pickle.dump(patient_data, pickle_file)

    print(img_shapes)


def check_stdev(res):
    l = [statistics.stdev(res[0]), statistics.stdev(res[1])]
    s_std = min(l)
    l_std = max(l)

    return s_std / l_std


if __name__ == "__main__":
    run_prerocess(etlstream.Origin.SB, True)
    run_prerocess(etlstream.Origin.MC7, True)
    run_prerocess(etlstream.Origin.ST11, True)
