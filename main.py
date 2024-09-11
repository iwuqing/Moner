# -*- coding: utf-8 -*-
# ----------------------------------------------#
# Project  : Radial_CT
# File     : main.py
# Date     : 2024/6/2
# Author   : Qing Wu
# Email    : wuqing@shanghaitech.edu.cn
# ----------------------------------------------#
import train
import utils
import numpy as np
import SimpleITK as sitk


if __name__ == '__main__':
    # AF=2x, MR=\pm5
    spoke_num, mot = 360, 5
    # load config file
    config_path = 'config.yaml'
    config = utils.load_config(config_path)
    # results
    rot_list = []
    shift_x_list = []
    shift_y_list = []
    recon_list = np.zeros(shape=(5, config['file']['h'], config['file']['w']), dtype=np.float32)

    for i in range(5):
        rot, shift_x, shift_y, recon = train.train(config_path=config_path, rot=mot,
                                                   spoke_num=spoke_num, sample_index=i)
        rot_list.append(rot)
        shift_x_list.append(shift_x)
        shift_y_list.append(shift_y)
        recon_list[i, :, :] = recon

        np.savetxt('{}/rot_{}_{}.txt'.format(config["file"]["out_dir"], spoke_num, mot), np.array(rot_list))
        np.savetxt('{}/shift_x_{}_{}.txt'.format(config["file"]["out_dir"], spoke_num, mot), np.array(shift_x_list))
        np.savetxt('{}/shift_y_{}_{}.txt'.format(config["file"]["out_dir"], spoke_num, mot), np.array(shift_y_list))
        sitk.WriteImage(sitk.GetImageFromArray(np.array(recon_list)), '{}/recon_{}_{}.nii'.format(config["file"]["out_dir"], spoke_num, mot))