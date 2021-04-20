import os
import pickle
import sys

import cv2
import numpy as np

import etlstream
import patient
from etlstream import StreamFactory


def run_prerocess(origin, filled=False):
    to_dir = os.path.join(sys.argv[1], origin.name)
    stream = StreamFactory.create(origin, use_cache=False)

    img_shapes = {}
    correct_img_shapes = set()
    for p in stream.stream_from_source():
        pickle_file_path = os.path.join(to_dir, p.patient_id + ".p")
        patient_data = patient.Patient()
        for img in p.images:
            img.get_image()
            if not img.image.shape in img_shapes.keys():
            	img_shapes[img.image.shape] = 0
            img_shapes[img.image.shape] += 1
            left_ventricle_contour_mask = np.zeros(img.image.shape)
            ln_contour = None
            lp_contour = None
            for cont in img.ground_truths:
                if cont.part == etlstream.Region.LN:
                    ln_contour = cont.contour_mtx.astype(np.int32)
                elif cont.part == etlstream.Region.LP:
                    lp_contour = cont.contour_mtx.astype(np.int32)
            if not ln_contour is None and not lp_contour is None and img.image.shape == (256, 256):
                correct_img_shapes.add(img.image.shape)
                cv2.drawContours(left_ventricle_contour_mask, [lp_contour], 0, color=255, thickness=-1)
                if not filled:
                    cv2.drawContours(left_ventricle_contour_mask, [ln_contour], 0, color=0, thickness=-1)
                img.image = img.image.astype('float64')
                img.image *= 255.0 / img.image.max()
                patient_data.add_data(img.image.astype('uint8'), left_ventricle_contour_mask.astype('uint8'))

        if not patient_data.is_empty():
            os.makedirs(os.path.dirname(pickle_file_path), exist_ok=True)
            with(open(pickle_file_path, "wb")) as pickle_file:
                pickle.dump(patient_data, pickle_file)

    print(img_shapes)
    print(correct_img_shapes)


if __name__ == "__main__":
    run_prerocess(etlstream.Origin.SB, False)
