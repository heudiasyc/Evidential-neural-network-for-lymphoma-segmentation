import sys

import numpy as np
import SimpleITK as sitk
from skimage import filters, measure

from .utils import get_study_uid, one_hot_encode

from monai.transforms.compose import MapTransform, Randomizable

# is instance(keys) == str = > keys=[keys]


class LoadNifti(MapTransform):
    """
    Load Nifti images and returns Simple itk object
    """

    def __init__(self, keys=("pet_img", "ct_img", "mask_img"),
                 dtypes=None,
                 image_only=False):
        super().__init__(keys)

        if dtypes is None:
            dtypes = {'pet_img': sitk.sitkFloat32,
                      'ct_img': sitk.sitkFloat32,
                      'mask_img': sitk.sitkUInt8}
        self.keys = keys
        self.image_only = image_only
        assert not self.image_only

        self.dtypes = dtypes

    def __call__(self, img_dict):
        output = dict()
        output['image_id'] = get_study_uid(img_dict[self.keys[0]])
        for key in self.keys:
            # check img_dict[key] == str
            output[key] = sitk.ReadImage(img_dict[key], self.dtypes[key])

        return output


class Roi2Mask(MapTransform):
    """
    Apply threshold-based method to determine the segmentation from the ROI
    """

    def __init__(self, keys=('pet_img', 'mask_img'), method='otsu', tval=0.0, idx_channel=-1):
        """
        :param keys:
        :param method: method to use for calculate the threshold
                Must be one of 'absolute', 'relative', 'otsu', 'adaptative'
        :param tval: Used only for method= 'absolute' or 'relative'. threshold value of the method.
                for 2.5 SUV threshold: use method='absolute', tval=2.5
                for 41% SUV max threshold: method='relative', tval=0.41
        :param idx_channel: idx of the ROI.
                for example, if ROI image shape is (n_roi, x, y, z) then idx_channel must be 0.
        """
        super().__init__(keys)

        self.keys = keys
        self.method = method.lower()
        self.tval = tval
        self.idx_channel = idx_channel

        assert method in ['absolute', 'relative', 'otsu', 'adaptative']

    def __call__(self, img_dict):

        pet_key = self.keys[0]
        roi_key = self.keys[1]
        #print(img_dict[roi_key])


        img_dict[roi_key] = self.roi2mask(img_dict[roi_key], img_dict[pet_key])
        return img_dict

    def calculate_threshold(self, roi):
        if self.method == 'absolute':
            return self.tval

        elif self.method == 'relative':
            # check len(roi) > 0
            SUV_max = np.max(roi)
            return self.tval * SUV_max

        elif self.method == 'adaptative' or self.method == 'otsu':
            # check len(np.unique(roi)) > 1
            return filters.threshold_otsu(roi)

    def roi2mask(self, mask_img, pet_img):
        """
        Generate the mask from the ROI of the pet scan
        Args:
            :param mask_img: sitk image, raw mask (i.e ROI)
            :param pet_img: sitk image, the corresponding pet scan

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
            # size = mask_img.GetSize()
        else:
            mask_array = np.rollaxis(mask_array, self.idx_channel, 0)

            # convert false-4d meta information to 3d information
            origin = mask_img.GetOrigin()[:-1]
            spacing = mask_img.GetSpacing()[:-1]
            direction = tuple(el for i, el in enumerate(mask_img.GetDirection()[:12]) if not (i + 1) % 4 == 0)
            # size = mask_img.GetSize()[:-1]

        new_mask = np.zeros(mask_array.shape[1:], dtype=np.int8)

        for num_slice in range(mask_array.shape[0]):
            mask_slice = mask_array[num_slice]
            roi = pet_array[mask_slice > 0]

            try:
                threshold = self.calculate_threshold(roi)

                # apply threshold
                new_mask[np.where((pet_array >= threshold) & (mask_slice > 0))] = 1

            except Exception as e:
                print(e)
                print(sys.exc_info()[0])

        # reconvert to sitk and restore information
        new_mask = sitk.GetImageFromArray(new_mask)
        new_mask.SetOrigin(origin)
        new_mask.SetDirection(direction)
        new_mask.SetSpacing(spacing)

        #sitk.WriteImage(new_mask,mask_img)

        return new_mask


class ConnectedComponent(MapTransform):
    """
    Get Connected component and transform to one-hot encoding
    """

    def __init__(self, keys='mask_img', channels_first=True, exclude_background=True):
        super().__init__(keys)
        self.channels_first = channels_first
        self.exclude_background = exclude_background

    def __call__(self, img_dict):

        mask = img_dict[self.keys[0]]
        blobs_labels = measure.label(mask, background=0)
        # convert to one hot: different components = different instance
        mask = one_hot_encode(blobs_labels)
        if self.exclude_background:
            mask = mask[:, :, :, 1:]  # exclude background
        if self.channels_first:
            mask = np.rollaxis(mask, 3)  # (x, y, z, n_object) to (n_object, x, y, z)
            n_obj = mask.shape[0]
        else:
            n_obj = mask.shape[-1]

        img_dict[self.keys[0]] = mask
        img_dict['iscrowd'] = np.zeros(n_obj, dtype=np.int8)
        # torch.zeros((n_obj,), dtype=torch.int64)
        return img_dict


class GenerateBbox(MapTransform):
    """
    Generate Bounding Box from segmentation
    """

    def __init__(self, keys='mask_img', channels_first=True):
        super().__init__(keys)
        self.channels_first = channels_first
        assert self.channels_first

    # y1, y2 = min(indexes[0]), max(indexes[0])
    # x1, x2 = min(indexes[1]), max(indexes[1])

    def __call__(self, img_dict):

        mask = img_dict[self.keys[0]]

        # generate bounding box from the segmentation
        bbox = []
        for i in range(mask.shape[0]):
            indexes = np.where(mask[i])
            x1, x2 = min(indexes[0]), max(indexes[0])
            y1, y2 = min(indexes[1]), max(indexes[1])
            z1, z2 = min(indexes[2]), max(indexes[2])
            bbox.append([x1, y1, z1, x2, y2, z2])

        bbox = np.array(bbox)
        img_dict['boxes'] = bbox

        area = (bbox[:, 3] - bbox[:, 0] + 1) * (bbox[:, 4] - bbox[:, 1] + 1) * (bbox[:, 5] - bbox[:, 2] + 1)
        img_dict['area'] = area
        return img_dict


class FilterObject(object):
    """
    Remove too small bouding boxes
    """

    def __init__(self, tval):
        self.tval = tval

    def __call__(self, img_dict):
        area = img_dict['area']
        # selected only R.O.I/object above the threshold
        idx = (area > self.tval)

        img_dict['area'] = area[idx]
        img_dict['mask_img'] = img_dict['mask_img'][idx]
        img_dict['boxes'] = img_dict['boxes'][idx]

        return img_dict


class ResampleReshapeAlign(MapTransform):
    """
    Resample to the same resolution, Reshape and Align to the same view.
    """

    def __init__(self, target_shape, target_voxel_spacing,
                 keys=('pet_img', 'ct_img', 'mask_img'),
                 origin='head', origin_key='pet_img'):
        """
        :param target_shape: tuple[int], (x, y, z)
        :param target_voxel_spacing: tuple[float], (x, y, z)
        :param keys:
        :param origin: method to set the view. Must be one of 'middle' 'head'
        :param origin_key: image reference for origin
        """
        super().__init__(keys)

        # mode="constant", cval=0,
        # axcodes="RAS", labels=(('R', 'L'), ('A', 'P'), ('I', 'S'))
        # np.flip(img, axis=0)

        self.keys = keys
        self.target_shape = target_shape
        self.target_voxel_spacing = target_voxel_spacing
        self.target_direction = (1, 0, 0, 0, 1, 0, 0, 0, 1)

        self.origin = origin
        self.origin_key = origin_key

        # sitk.sitkLinear, sitk.sitkBSpline, sitk.sitkNearestNeighbor
        self.interpolator = {'pet_img': sitk.sitkBSpline,
                             'ct_img': sitk.sitkBSpline,
                             'mask_img': sitk.sitkNearestNeighbor}

        self.default_value = {'pet_img': 0.0,
                              'ct_img': -1000.0,
                              'mask_img': 0}

    def __call__(self, img_dict):
        # compute transformation parameters
        new_origin = self.compute_new_origin(img_dict[self.origin_key])

        for key in self.keys:
            img_dict[key] = self.resample_img(img_dict[key], new_origin, self.default_value[key],
                                              self.interpolator[key])

        return img_dict

    def compute_new_origin_head2hip(self, pet_img):
        new_shape = self.target_shape
        new_spacing = self.target_voxel_spacing
        pet_size = pet_img.GetSize()
        pet_spacing = pet_img.GetSpacing()
        pet_origin = pet_img.GetOrigin()
        new_origin = (pet_origin[0] + 0.5 * pet_size[0] * pet_spacing[0] - 0.5 * new_shape[0] * new_spacing[0],
                      pet_origin[1] + 0.5 * pet_size[1] * pet_spacing[1] - 0.5 * new_shape[1] * new_spacing[1],
                      pet_origin[2] + 1.0 * pet_size[2] * pet_spacing[2] - 1.0 * new_shape[2] * new_spacing[2])
        return new_origin

    def compute_new_origin_centered_img(self, pet_img):
        origin = np.asarray(pet_img.GetOrigin())
        shape = np.asarray(pet_img.GetSize())
        spacing = np.asarray(pet_img.GetSpacing())
        new_shape = np.asarray(self.target_shape)
        new_spacing = np.asarray(self.target_voxel_spacing)

        return tuple(origin + 0.5 * (shape * spacing - new_shape * new_spacing))

    def compute_new_origin(self, img):
        if self.origin == 'middle':
            return self.compute_new_origin_centered_img(img)
        elif self.origin == 'head':
            return self.compute_new_origin_head2hip(img)

    def resample_img(self, img, new_origin, default_value, interpolator):
        # transformation parametrisation
        transformation = sitk.ResampleImageFilter()
        transformation.SetOutputDirection(self.target_direction)
        transformation.SetOutputOrigin(new_origin)
        transformation.SetOutputSpacing(self.target_voxel_spacing)
        transformation.SetSize(self.target_shape)

        transformation.SetDefaultPixelValue(default_value)
        transformation.SetInterpolator(interpolator)

        return transformation.Execute(img)


class Sitk2Numpy(MapTransform):
    def __init__(self, keys=('pet_img', 'ct_img', 'mask_img')):
        super().__init__(keys)
        self.keys = keys

    def __call__(self, img_dict):

        for key in self.keys:
            img = sitk.GetArrayFromImage(img_dict[key])
            img = np.transpose(img, (2, 1, 0))  # (z, y, x) to (x, y, z)
            img_dict[key] = img

        return img_dict


class ConcatModality(MapTransform):

    def __init__(self, keys=('pet_img', 'ct_img'), channel_first=True, new_key='image', del_keys=True):
        super().__init__(keys)
        self.keys = keys
        self.channel_first = channel_first
        self.new_key = new_key
        self.del_keys = del_keys

    def __call__(self, img_dict):
        idx_channel = 0 if self.channel_first else -1
        imgs = (img_dict[key] for key in self.keys)
        img_dict[self.new_key] = np.stack(imgs, axis=idx_channel)

        if self.del_keys:
            for key in self.keys:
                del img_dict[key]
                # del img_dict[key + '_meta_dict']

        return img_dict
