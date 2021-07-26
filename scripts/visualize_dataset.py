import cv2
import numpy as np
from generic_dataset.dataset_folder_manager import DatasetFolderManager
from generic_dataset.dataset_manager import DatasetManager

from deep_doors_2.door_sample import DoorSample

dataset_path ='/home/michele/myfiles/deep_doors_2'

# Create the DatasetFolderManager instance and read sample
folder_manager = DatasetFolderManager(dataset_path=dataset_path, folder_name='deep_doors_2', sample_class=DoorSample)

for sample_absolute_count in folder_manager.get_samples_absolute_counts(label=1):

    # Load a sample (positive, label = 1)
    sample: DoorSample = folder_manager.load_sample_using_absolute_count(absolute_count=sample_absolute_count, use_thread=False)
    sample.visualize()