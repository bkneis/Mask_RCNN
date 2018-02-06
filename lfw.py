import cv2
import os
import utils
import numpy as np
from config import Config
from random import randint


def _build_paths(image_dir, label_dir, image_subdir, label_file):
    image_file = label_file[:-3] + 'jpg'
    image_path = os.path.join(image_dir, image_subdir, image_file)
    label_path = os.path.join(label_dir, label_file)
    return [image_path, label_path]


def get_image_paths(image_dir, label_dir):
    label_files = []
    train_files = []

    for filename in os.listdir(label_dir):
        if filename.endswith(".ppm"):
            label_files.append(filename)

    for label_file in label_files:
        image_file = label_file[:-9]
        train_files.append(_build_paths(image_dir, label_dir, image_file, label_file))

    return train_files


class FacesConfig(Config):
    """Configuration for training on the lfw dataset
    Derives from the base Config class and overrides values specific
    to the lfw dataset.
    """
    # Give the configuration a recognizable name
    NAME = "faces"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + face and hair

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


class FacesDataset(utils.Dataset):
    """Generates the lfw dataset """

    def __init__(self, images, obscured=False):
        self.images = images
        self.obscured = obscured
        super().__init__()

    def load_faces(self, start, end, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("faces", 1, "face")
        # self.add_class("faces", 2, "hair")

        center = 128
        _width = 80
        x = randint(center - _width, center + _width)
        y = randint(center - _width, center + _width)

        for idx, img in enumerate(self.images):
            if idx < start:
                continue
            if idx > end:
                continue
            bg_color = np.array([0, 0, 255])
            self.add_image("faces", image_id=idx, path=img[0],
                           width=width, height=height,
                           bg_color=bg_color, obscured=self.obscured, x=x, y=y)

    def load_image(self, image_id):
        img = cv2.imread(self.images[image_id][0])
        height, width = img.shape[:2]
        img = cv2.resize(img, (width + 6, height + 6), interpolation=cv2.INTER_CUBIC)
        info = self.image_info[image_id]

        if self.obscured:
            # Add a random shape in the image to obstruct the face
            cv2.circle(img, (info['x'], info['y']), self.obscured, (0, 0, 0), -1)

        return img

    def image_reference(self, image_id):
        """Return the face data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "faces":
            return info
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for faces of the given image ID.
        """
        mask = cv2.imread(self.images[image_id][1])
        height, width = mask.shape[:2]
        mask = cv2.resize(mask, (width + 6, height + 6), interpolation=cv2.INTER_CUBIC)
        class_ids = np.array([1])

        info = self.image_info[image_id]
        if self.obscured:
            # Add a random shape in the image to obstruct the face
            cv2.circle(mask, (info['x'], info['y']), self.obscured, (0, 0, 0), -1)

        binary_mask = np.zeros([info['height'], info['width'], 1], dtype=np.uint8)

        for r in range(mask.shape[0]):
            for c in range(mask.shape[1]):
                g = mask[r][c][1]
                if g == 255:
                    binary_mask[r][c] = 1

        return binary_mask, class_ids.astype(np.int32)
