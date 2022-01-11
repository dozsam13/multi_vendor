import torch
import numpy as np
import gryds
from skimage.exposure import equalize_adapthist
from scipy.ndimage import gaussian_filter


class DomainAugmentation:
    def __init__(self):
        pass

    def __call__(self, image, label):
        image = image.reshape(256, 256)
        label = label.reshape(256, 256)
        ia = False
        random_grid = np.random.rand(2, 7, 7)
        random_grid -= 0.5
        random_grid /= 12
        # Define a B-spline transformation object
        bspline_trf = gryds.BSplineTransformation(random_grid)

        # rotate between -pi/8 and pi/8
        rot = np.random.rand() * np.pi / 4 - np.pi / 8
        # scale between 0.9 and 1.1
        scale_x = np.random.rand() * 0.2 + 0.9
        scale_y = np.random.rand() * 0.2 + 0.9
        # translate between -10% and 10%
        trans_x = np.random.rand() * .2 - .1
        trans_y = np.random.rand() * .2 - .1

        affine_trf = gryds.AffineTransformation(
            ndim=2,
            angles=[rot],  # the rotation angle
            scaling=[scale_x, scale_y],  # the anisotropic scaling
            translation=[trans_x, trans_y],  # translation
            center=[0.5, 0.5]  # center of rotation
        )
        composed_trf = gryds.ComposedTransformation(bspline_trf, affine_trf)

        t_ind = np.random.randint(2)
        interpolator = gryds.Interpolator(image[:, :], mode='reflect')

        interpolator_label = gryds.Interpolator(label[:, :], order=0, mode='constant')

        patch = interpolator.transform(composed_trf)
        patch_label = interpolator_label.transform(composed_trf)

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

        return patch, patch_label

    def __repr__(self):
        return "domain augmentation"
