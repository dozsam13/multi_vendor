import numpy as np
import torch
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

class ElasticTransform(object):
    def __init__(self, alpha=512, sigma=20):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img):
        # source: https://gist.github.com/fmder/e28813c1e8721830ff9c
        random_state = np.random.RandomState(seed=seed)

        if torch.is_tensor(img):
            img = img.numpy()
        shape = img.shape

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma) * self.alpha

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

        out = map_coordinates(img, indices, order=1, mode='reflect').reshape(shape)

        # print('elastic')
        # f, axarr = plt.subplots(1, 2)
        # axarr[0].imshow(img.reshape(256,256), cmap='gray')
        # axarr[1].imshow(out.reshape(256,256), cmap='gray')
        # plt.show()

        # return torch.from_numpy(out)
        return out

    def __repr__(self):
        return self.__class__.__name__ + f'(alpha={self.alpha}, sigma={self.sigma})'