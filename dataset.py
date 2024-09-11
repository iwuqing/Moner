# -*- coding: utf-8 -*-
import utils
import numpy as np
import SimpleITK as sitk
from torch.utils import data


class TestDataDirect(data.Dataset):
    def __init__(self, h, w):
        self.rays = utils.grid_coordinate(h=h, w=w).reshape(1, -1, 2)

    def __len__(self):
        return 1

    def __getitem__(self, item):
        ray = self.rays[item]  # (h*w, 2)
        return ray


class TrainData(data.Dataset):
    def __init__(self, proj_path, num_sample_ray, sample_index):
        self.num_sample_ray = num_sample_ray
        # load sparse-view projction data
        self.proj = sitk.GetArrayFromImage(sitk.ReadImage(proj_path))[sample_index]
        self.num_angle, self.SOD, _ = self.proj.shape
        self.rays = utils.grid_coordinate(h=self.SOD, w=self.SOD)   # (SOD, SOD, 2)
        self.index_max = self.SOD - self.num_sample_ray

    def __len__(self):
        return self.num_angle

    def __getitem__(self, item):
        proj = self.proj[item]      # (SOD, 2)
        # sample ray
        index = np.random.randint(0, self.index_max, size=1)[0]
        ray_sample = self.rays[index:index + self.num_sample_ray]  # (num_sample_ray, 2*SOD, 2)
        proj_sample = proj[index:index + self.num_sample_ray]  # (num_sample_ray, )
        return item, ray_sample, proj_sample
