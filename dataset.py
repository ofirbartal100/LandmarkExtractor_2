import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import re
import time
import os
import json

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


class FreiHandDataset(Dataset):
    """Ganerated Hands's images labeled with joints lanmarks dataset."""

    def __init__(self, dataset_directory_path,set_name='training',percent = 0.01, transform=None):
        """
        Args:
            labels_csv_path (string): Path to the csv file with annotations.
            images_root_directory_path (string): Directory with all the images.
        """
        self.transform = transform
        self.set_name = set_name
        self.dataset_directory_path = dataset_directory_path
        self.dataset_annotations = list(FreiHandDataset.load_db_annotation(dataset_directory_path,percent))

    @staticmethod
    def load_db_annotation(base_path,percent, set_name=None):
        def _assert_exist(p):
            msg = 'File does not exists: %s' % p
            assert os.path.exists(p), msg

        def json_load(p):
            _assert_exist(p)
            with open(p, 'r') as fi:
                d = json.load(fi)
            return d

        if set_name is None:
            # only training set annotations are released so this is a valid default choice
            set_name = 'training'

        print('Loading FreiHAND dataset index ...')
        t = time.time()

        # assumed paths to data containers
        k_path = os.path.join(base_path, '%s_K.json' % set_name)
        mano_path = os.path.join(base_path, '%s_mano.json' % set_name)
        xyz_path = os.path.join(base_path, '%s_xyz.json' % set_name)

        # load if exist
        K_list = json_load(k_path)
        mano_list = json_load(mano_path)
        xyz_list = json_load(xyz_path)

        # should have all the same length
        assert len(K_list) == len(mano_list), 'Size mismatch.'
        assert len(K_list) == len(xyz_list), 'Size mismatch.'

        percent_trim = int(len(K_list)*percent)
        print('Loading of %d samples done in %.2f seconds' % (percent_trim, time.time() - t))
        return zip(K_list[0:percent_trim], mano_list[0:percent_trim], xyz_list[0:percent_trim])

    @staticmethod
    def projectPoints(xyz, K):
        """ Project 3D coordinates into image space. """
        xyz = np.array(xyz)
        K = np.array(K)
        uv = np.matmul(K, xyz.T).T
        return uv[:, :2] / uv[:, -1:]

    def __len__(self):
        return len(self.dataset_annotations)

    def __getitem__(self, idx):
        """
        query data2 from the csv file
        :param idx: row index in csv file
        :return: PIL grayscale image , 21 landmarks label
        """
        # i dont use the mano
        K, mano, xyz = self.dataset_annotations[idx]
        K, mano, xyz = [np.array(x) for x in [K, mano, xyz]]
        image_path = os.path.join(self.dataset_directory_path, self.set_name, 'rgb','%08d.jpg' % idx)
        image = Image.open(image_path)
        landmarks = self.projectPoints(xyz, K)
        label = landmarks.reshape(-1, 2)

        if self.transform:
            image,label = self.transform(image,label)

        return image, label
