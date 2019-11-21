import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import re


class GaneratedHandsDataset(Dataset):
    """Ganerated Hands's images labeled with joints lanmarks dataset."""

    def __init__(self, dataset_csv_path, transform=None):
        """
        Args:
            labels_csv_path (string): Path to the csv file with annotations.
            images_root_directory_path (string): Directory with all the images.
        """
        self.transform = transform
        self.dataset = pd.read_csv(dataset_csv_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        query data2 from the csv file
        :param idx: row index in csv file
        :return: PIL grayscale image , 21 landmarks label
        """
        record = self.dataset.iloc[idx]
        image_path = record[1]
        image = Image.open(image_path)
        landmarks = re.findall("\d\d*\.?\d*", record[2])

        landmarks = np.array(landmarks, dtype='float')
        label = landmarks.reshape(-1, 2)

        if self.transform:
            image,label = self.transform(image,label)

        return image, label
