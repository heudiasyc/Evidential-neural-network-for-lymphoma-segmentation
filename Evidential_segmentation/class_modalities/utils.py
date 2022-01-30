import os
import re

import numpy as np

import SimpleITK as sitk
from skimage import filters

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def resample_img(img,
                 target_direction, new_origin,
                 target_voxel_spacing, target_shape,
                 default_value, interpolator):

    transformation = sitk.ResampleImageFilter()
    transformation.SetOutputDirection(target_direction)
    transformation.SetOutputOrigin(new_origin)
    transformation.SetOutputSpacing(target_voxel_spacing)
    transformation.SetSize(target_shape)

    transformation.SetDefaultPixelValue(default_value)
    transformation.SetInterpolator(interpolator)

    return transformation.Execute(img)


def normalize_img(img, window_min, window_max):
    """
    Transform input value from window_min - window_max to 0 - 1
    """
    intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
    intensityWindowingFilter.SetOutputMaximum(1)
    intensityWindowingFilter.SetOutputMinimum(0)
    intensityWindowingFilter.SetWindowMaximum(window_max)
    intensityWindowingFilter.SetWindowMinimum(window_min)
    return intensityWindowingFilter.Execute(img)


def normalize_img_v2(img, shift, scale):
    return sitk.ShiftScale(img, shift=shift, scale=scale)


def threshold_img(img, threshold):
    return sitk.Threshold(img, lower=0.0, upper=threshold, outsideValue=threshold)


def mip(img, threshold=None):
    img_array = sitk.GetArrayFromImage(img)

    if threshold:
        # img_array = np.array(img_array>threshold, dtype=np.int8)
        img_array[img_array > threshold] = threshold
    return np.max(img_array, axis=1), np.max(img_array, axis=2)


def get_info(img):
    print('img information :')
    print('\t Origin    :', img.GetOrigin())
    print('\t Size      :', img.GetSize())
    print('\t Spacing   :', img.GetSpacing())
    print('\t Direction :', img.GetDirection())


def get_study_uid(img_path):
    return re.sub('_nifti_(PT|mask|CT)\.nii(\.gz)?', '', os.path.basename(img_path))


def one_hot_encode(x, n_classes=None):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    if n_classes is None:
        n_classes = np.max(x) + 1
    return np.eye(n_classes)[x]


def roi2tmtv(mask_img, pet_img, threshold='auto'):
    """
    Generate the mask from the ROI of the pet scan
    Args:
        :param mask_img: sitk image, raw mask (i.e ROI)
        :param pet_img: sitk image, the corresponding pet scan
        :param threshold: threshold to apply to the ROI to get the tumor segmentation.
                if set to 'auto', it will take 42% of the maximum
    :return: sitk image, the ground truth segmentation
    """
    # transform to numpy
    mask_array = sitk.GetArrayFromImage(mask_img)
    pet_array = sitk.GetArrayFromImage(pet_img)

    # get 3D meta information
    if len(mask_array.shape) == 3:
        mask_array = np.expand_dims(mask_array, axis=0)

        origin = mask_img.GetOrigin()
        spacing = mask_img.GetSpacing()
        direction = tuple(mask_img.GetDirection())
        size = mask_img.GetSize()
    else:
        origin = mask_img.GetOrigin()[:-1]
        spacing = mask_img.GetSpacing()[:-1]
        direction = tuple(el for i, el in enumerate(mask_img.GetDirection()[:12]) if not (i + 1) % 4 == 0)
        size = mask_img.GetSize()[:-1]

    # print(pet_img.GetOrigin(), origin)
    # print(pet_img.GetSpacing(), spacing)
    # print(pet_img.GetDirection(), direction)
    # print(pet_img.GetSize(), size)
    # print(mask_array.shape)

    # assert meta-info roi == meta-info pet
    assert pet_img.GetOrigin() == origin
    assert pet_img.GetSpacing() == spacing
    assert pet_img.GetDirection() == direction
    assert pet_img.GetSize() == size

    # generate mask from ROIs
    new_mask = np.zeros(mask_array.shape[1:], dtype=np.int8)
    n_voxels_per_roi = np.zeros(mask_array.shape[0])

    for num_slice in range(mask_array.shape[0]):
        mask_slice = mask_array[num_slice]

        # calculate threshold value of the roi
        if threshold == 'auto':
            roi = pet_array[mask_slice > 0]
            if len(roi) > 0:
                SUV_max = np.max(roi)
                threshold_suv = SUV_max * 0.41
            else:
                threshold_suv = 0.0
        elif threshold == 'otsu':
            roi = pet_array[mask_slice > 0]
            if len(roi) > 0:
                threshold_suv = filters.threshold_otsu(roi)
            else:
                threshold_suv = 0.0
        else:
            threshold_suv = threshold

        # apply threshold
        n_voxels_per_roi[num_slice] = np.sum((pet_array >= threshold_suv) & (mask_slice > 0))
        new_mask[np.where((pet_array >= threshold_suv) & (mask_slice > 0))] = 1

    return np.sum(new_mask), n_voxels_per_roi


def plot_hist_roi(mask_img, pet_img):
    mask_array = sitk.GetArrayFromImage(mask_img)
    pet_array = sitk.GetArrayFromImage(pet_img)

    for num_slice in range(mask_array.shape[0]):
        mask_slice = mask_array[num_slice]
        roi = pet_array[mask_slice > 0]

        sns.distplot(roi, hist=True, kde=True, rug=True, rug_kws={'color': 'black'})

        # colors = [cmap(0.), cmap(0.5), cmap(1.0)] # ['red', 'green', 'orange']
        colors = ['red', 'green', 'orange']
        for threshold_val, color in zip([filters.threshold_otsu(roi), 2.5, 0.42 * np.max(roi)], colors):
            plt.plot([threshold_val, threshold_val], [0.0, 1.0], '--', color=color)

        custom_lines = [Line2D([0], [0], color=colors[0], lw=4),
                        Line2D([0], [0], color=colors[1], lw=4),
                        Line2D([0], [0], color=colors[2], lw=4)]

        plt.legend(custom_lines, ['otsu', '2.5 SUV', '42%'])
        #     plt.savefig('plot/train{}_slice{}_hist'.format(idx, num_slice))
        plt.show()
