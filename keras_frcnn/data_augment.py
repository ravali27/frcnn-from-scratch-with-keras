from __future__ import division
import numpy as np
from scipy.ndimage import interpolation
import copy

def resize_n(old, new_shape):
    new_f, new_t = new_shape
    old_f, old_t = old.shape
    scale_f, scale_t = new_f/old_f, new_t/old_t
    new = interpolation.zoom(old, (scale_f, scale_t))
    return new 

def augment(img_data, config, augment=True):
	assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	img_data_aug = copy.deepcopy(img_data)
	img_o = np.loadtxt(img_data_aug['filepath'])
	sd = 2126.5
	img = img_o/sd
	img = resize_n(img, (224, 224))

	if augment:
		rows, cols = img.shape[:2]

        if config.use_freq_mask and np.random.randint(0, 2) == 0:
            #print('freq_mask')
            mid = np.random.randint(0, img.shape[0])
            length = np.random.randint(0, 10)
            img_new = img
            start = mid -length
            stop = mid + length
            if start < 0:
                start = 0
            if stop >= img.shape[0]:
                stop = img.shape[0]-1
            img_new[:, start:stop] = np.mean(img[:, start:stop])
            img = img_new

        if config.use_time_mask and np.random.randint(0, 2) == 0:
            #print('time_mask')
            mid = np.random.randint(0, img.shape[1])
            length = np.random.randint(0, 10)
            img_new = img
            start = mid -length
            stop = mid + length
            if start < 0:
                start = 0
            if stop >= img.shape[1]:
                stop = img.shape[1]-1
            img_new[start:stop, :] = np.mean(img[start:stop, :])
            img = img_new

	img = np.stack((img, img, img), axis=2)
	img_data_aug['width'] = img.shape[1]
	img_data_aug['height'] = img.shape[0]
	return img_data_aug, img
