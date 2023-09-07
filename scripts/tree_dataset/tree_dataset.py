import numpy as np
import rioxarray
import skimage.transform

from rioxarray.exceptions import NoDataInBounds
from sklearn.preprocessing import LabelEncoder


MIN_PIXEL_SIZE_DEF = 32
MAX_PIXEL_SIZE_DEF = 100
PIXEL_SIZE_DEF = 100
MIN_SAMPLE_SIZE_DEF = 10


class TreeDataset:
    def __init__(
            self,
            rgb_paths,
            bboxes,
            classes,
            min_pixel_size=None,
            max_pixel_size=None,
            pixel_size=None,
            min_sample_size=None,
            augment_data=False
    ):
        self.rgb_paths = [rgb_paths] \
            if isinstance(rgb_paths, str) else rgb_paths
        self.bboxes = bboxes
        self.classes = classes

        self.min_pixel_size = min_pixel_size if min_pixel_size is not None \
            else MIN_PIXEL_SIZE_DEF
        self.max_pixel_size = max_pixel_size if max_pixel_size is not None \
            else MAX_PIXEL_SIZE_DEF
        self.pixel_size = pixel_size if pixel_size is not None \
            else PIXEL_SIZE_DEF
        self.min_sample_size = min_sample_size if min_sample_size is not None \
            else MIN_SAMPLE_SIZE_DEF

        self.augment_data = augment_data

    def _generate_cutouts_for_image(self, rgb_path):
        """ Extract all cutouts for a given image. """

        rgb = rioxarray.open_rasterio(rgb_path, masked=True)

        assert self.bboxes.crs == rgb.rio.crs

        # select relevant bboxes
        xmin, ymin, xmax, ymax = rgb.rio.bounds()
        bboxes_clip = self.bboxes.cx[xmin:xmax, ymin:ymax]
        bboxes_clip = bboxes_clip[~bboxes_clip.is_empty]

        for id, bbox in bboxes_clip.items():

            # some polygons may have very small intersections
            try:
                cutout = rgb.rio.clip([bbox], drop=True)
            except NoDataInBounds:
                continue

            img = _to_np_array(cutout)

            # drop large or small cutouts
            if (
                _large_image(img, self.max_pixel_size)
                or _small_image(img, self.min_pixel_size)
            ):
                continue

            label = self.classes[id]
            img = _pad_image(img, self.pixel_size)

            yield id, label, img

    def get_cutouts(self):
        """ Load cutouts from all images. """
        tree_ids = []
        labels = []
        imgs = []
        for rgb_path in self.rgb_paths:
            cutouts = self._generate_cutouts_for_image(rgb_path)
            for tree_id, label, img in cutouts:
                tree_ids.append(tree_id)
                labels.append(label)
                imgs.append(img)

        tree_ids, labels, imgs = _remove_small_classes(
            np.asarray(tree_ids),
            np.asarray(labels),
            np.stack(imgs),
            self.min_sample_size
        )

        if self.augment_data:
            tree_ids, labels, imgs = _augment_data(
                tree_ids, labels, imgs
            )

        # convert class labels to categorical values
        if not np.issubdtype(labels.dtype, np.integer):
            le = LabelEncoder()
            le.fit(labels)
            labels = le.transform(labels)
        return tree_ids, labels, imgs


def _to_np_array(data_array):
    """
    Extract image data from the DataArray, standardizing the
    axis ordering: (y, x, band).
    """
    if data_array.ndim == 3:
        return data_array.transpose('y', 'x', 'band').data
    elif data_array.ndim == 2:
        return data_array.transpose('y', 'x').data
    else:
        raise ValueError('Input data shape not understood.')


def _remove_small_classes(tree_ids, labels, imgs, min_sample_size):
    """
    Balance class compositions by dropping classes with
    few samples.
    """
    labels_unique, counts = np.unique(labels, return_counts=True)
    labels_to_keep = labels_unique[counts >= min_sample_size]
    mask = np.isin(labels, labels_to_keep)
    return (
        tree_ids[mask],
        labels[mask],
        imgs[mask],
    )


def _large_image(data, max_pixel_size):
    if max(data.shape[0:2]) > max_pixel_size:
        return True
    return False


def _small_image(data, min_pixel_size):
    if min(data.shape[0:2]) <= min_pixel_size:
        return True
    return False


def _flip(img):
    # horizontal flip
    return img[::-1, :]


def _rotate(img, angle, resize=False):
    return skimage.transform.rotate(
        img,
        angle,
        resize=resize,
        mode='constant',
        cval=0,
    )


def _augment_data(tree_ids, labels, imgs):
    """
    Augment data by performing 90 degree rotations and flipping to the
    images for all classes.
    """
    # new_imgs = imgs
    # new_labels = labels
    # new_ids = tree_ids
    tree_ids_add = []
    labels_add = []
    imgs_add = []
    for id, lbl, img in zip(tree_ids, labels, imgs):
        # Flip and add image
        flipped_img = _flip(img)
        imgs_add.append(flipped_img)
        labels_add.append(lbl)
        tree_ids_add.append(id)

        # Rotate image by three angles and flip each image
        for angle in [90, 180, 270]:
            rotated_image = _rotate(img, angle)
            imgs_add.append(rotated_image)
            labels_add.append(lbl)
            tree_ids_add.append(id)
            flipped_img = _flip(rotated_image)
            imgs_add.append(flipped_img)
            labels_add.append(lbl)
            tree_ids_add.append(id)

    imgs = np.concatenate([imgs, imgs_add])
    tree_ids = np.concatenate([tree_ids, tree_ids_add])
    labels = np.concatenate([labels, labels_add])

    return tree_ids, labels, imgs


def _pad_image(data, pixel_size):
    """ Pad each image """
    pad_width_x1 = np.floor((pixel_size - data.shape[1])/2).astype(int)
    pad_width_x2 = pixel_size - data.shape[1] - pad_width_x1
    pad_width_y1 = np.floor((pixel_size - data.shape[0])/2).astype(int)
    pad_width_y2 = pixel_size - data.shape[0] - pad_width_y1
    data = np.pad(
        data,
        pad_width=[
            (pad_width_y1, pad_width_y2),
            (pad_width_x1, pad_width_x2),
            (0, 0),
        ],
        mode='constant'
    )
    return data
