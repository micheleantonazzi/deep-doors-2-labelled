import os
from typing import List, NoReturn, Tuple

import cv2
import numpy as np
from generic_dataset.data_pipeline import DataPipeline
from generic_dataset.generic_sample import synchronize_on_fields
from generic_dataset.sample_generator import SampleGenerator
from generic_dataset.utilities.save_load_methods import save_compressed_dictionary, load_compressed_dictionary, save_cv2_image_bgr, load_cv2_image_bgr, save_compressed_numpy_array, load_compressed_numpy_array


pipeline_fix_gbr_image = DataPipeline().add_operation(lambda d, e: (d[..., [2, 1, 0]], e))

@synchronize_on_fields(field_names={'bgr_image', 'depth_image', 'bounding_boxes'}, check_pipeline=True)
def visualize(self) -> NoReturn:
    """
    This method visualizes the sample, showing all its fields.
    :return:
    """
    print(f'Label: {self.get_label()}')
    print(f'Robot pose: {self.get_robot_pose()}')
    bgr_image = self.get_bgr_image()
    depth_image = self.get_depth_image()
    semantic_image = self.get_semantic_image()
    pretty_semantic_image = self.get_pretty_semantic_image()

    row_1 = np.concatenate((bgr_image, cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)), axis=1)
    row_2 = np.concatenate((semantic_image, pretty_semantic_image), axis=1)
    image = np.concatenate((row_1, row_2), axis=0)

    cv2.imshow('Sample', image)
    cv2.waitKey()


# The bounding_boxes field is a numpy array of tuple [(label, x1, y1, width, height)],
# where label is the bounding box label and (x1, y1) are the coordinates of the top left point and width height the bbox dimension

DOOR_LABELS = {0: 'Closed door', 1: 'Semi opened door', 2: 'Opened doorw'}

DoorSample = SampleGenerator(name='DoorSample', label_set={0, 1}) \
    .add_dataset_field(field_name='bgr_image', field_type=np.ndarray, save_function=save_cv2_image_bgr, load_function=load_cv2_image_bgr) \
    .add_dataset_field(field_name='depth_image', field_type=np.ndarray, save_function=save_cv2_image_bgr, load_function=load_cv2_image_bgr) \
    .add_dataset_field(field_name='bounding_boxes', field_type=np.ndarray, default_value=np.array([]), load_function=load_compressed_numpy_array, save_function=save_compressed_numpy_array) \
    .add_custom_method(method_name='visualize', function=visualize) \
    .generate_sample_class()