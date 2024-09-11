# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import utils


class PoseRefiner(nn.Module):
    def __init__(self, theta, theta_view, offset, theta_is_train, offset_is_train):
        super(PoseRefiner, self).__init__()
        # dim [num_angle, 1, 1]
        self.theta_view = nn.Parameter(torch.tensor(theta_view, dtype=torch.float), requires_grad=False)
        # dim [num_stage, 1, 1]
        self.theta = nn.Parameter(torch.tensor(theta, dtype=torch.float), requires_grad=theta_is_train)
        # dim [num_stage, 1, 2]
        self.offset = nn.Parameter(torch.tensor(offset, dtype=torch.float), requires_grad=offset_is_train)
        self.num_stage = self.offset.shape[0]

    def forward(self, img_id, ray_sample):
        # N, 2, 2, -> N, 1, 2, 2
        rotation_matrix = utils.angle_to_rotation_matrix((self.theta_view[img_id]+
                                                          self.theta[img_id%self.num_stage]).squeeze(1).squeeze(1)).unsqueeze(1)
        translation_vec = self.offset[img_id%self.num_stage].unsqueeze(1)
        return torch.matmul(ray_sample, rotation_matrix) + translation_vec
