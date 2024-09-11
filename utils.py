# -*- coding: utf-8 -*-
import torch
import yaml
import numpy as np
import h5py
import torchkbnufft as tkbn
from tqdm import tqdm
import SimpleITK as sitk
from skimage.transform import iradon
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from skimage.transform import SimilarityTransform
from skimage.transform import warp


def load_config(path):
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def cal_ssim(ref, img):
    data_range = 1
    return structural_similarity(ref, img, data_range=data_range)


def cal_psnr(ref, img):
    data_range = 1
    return peak_signal_noise_ratio(ref, img, data_range=data_range)


def ifft1d(data):
    return torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(data, dim=-1), dim=-1), dim=-1)


def fft1d(data):
    return torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(data, dim=-1), dim=-1), dim=-1)


def normalization(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def get_traj_freqency(theta, spoke_size):
    angles = torch.tensor(np.deg2rad(theta), dtype=torch.float64).unsqueeze(1)
    pos = torch.linspace(-np.pi, np.pi, spoke_size, dtype=torch.float64).unsqueeze(0)
    kx = torch.mm(torch.cos(angles), pos)
    ky = torch.mm(torch.sin(angles), pos)
    grid_coord = torch.stack([kx, ky], dim=0)
    return grid_coord


def grid_coordinate(h, w):
    x = np.linspace(-1, 1, h, endpoint=True)
    y = np.linspace(-1, 1, w, endpoint=True)
    x, y = np.meshgrid(x, y, indexing='ij')  # (h, w)
    xy = np.stack([x, y], -1)  # (h, w, 2)
    return xy


def angle_to_rotation_matrix(angle):
    # angle: (n, ), torch.tensor
    rotation_matrix = torch.zeros(size=(angle.shape[0], 2, 2), requires_grad=True).to(angle.device)
    rotation_matrix[:, 0, 0] = torch.cos(angle)
    rotation_matrix[:, 0, 1] = torch.sin(angle)
    rotation_matrix[:, 1, 0] = -torch.sin(angle)
    rotation_matrix[:, 1, 1] = torch.cos(angle)
    return rotation_matrix


def radom_motion(num_point, motion_range):
    motion = motion_range * (2*np.random.rand(num_point) - 1)
    return motion



def data_simulation(gt_all, spoke_num, stage_num, spoke_size, mot, out_dir, name):
    rot_list = []
    shift_x_list = []
    shift_y_list = []
    gt_list = []
    zf_list = []
    kdata_list = []
    for i in tqdm(range(gt_all.shape[0])):
        # gt imae
        gt = gt_all[i]
        h, w = gt_all.shape[-2], gt_all.shape[-1]
        if name == 'fastmri':
            gt_img = np.pad(gt, ((95, 96), (95, 96)))
        if name == 'brain':
            gt_img = np.pad(gt, ((127, 128), (127, 128)))

        # motion
        # ------------------------
        rot = radom_motion(num_point=stage_num, motion_range=mot)
        shift_x = radom_motion(num_point=stage_num, motion_range=mot)
        shift_y = radom_motion(num_point=stage_num, motion_range=mot)

        # save
        # ------------------------
        gt_list.append(np.abs(gt))
        rot_list.append(rot)
        shift_x_list.append(shift_x)
        shift_y_list.append(shift_y)

        # traj
        # ------------------------
        theta = np.linspace(0, 180, spoke_num, endpoint=False)
        index = np.array([i for i in range(0, spoke_num, stage_num)])

        # simulate
        # ------------------------
        nufft_ob = tkbn.KbNufft(im_size=(spoke_size, spoke_size)).to(torch.complex128)
        k_data = torch.zeros(size=(spoke_num, spoke_size), dtype=torch.complex128)
        for i in range(stage_num):
            # rotation
            tform = SimilarityTransform(translation=(shift_x[i], shift_y[i]))
            img_shift = np.complex128(warp(gt_img.real, tform, clip=True) + 1j * warp(gt_img.imag, tform, clip=True))
            # nufft
            ktraj_motion = get_traj_freqency(theta=theta[index + i] + rot[i], spoke_size=spoke_size)
            k_data[index + i, :] = nufft_ob(torch.tensor(img_shift).unsqueeze(0).unsqueeze(0).to(torch.complex128),
                                            ktraj_motion.view(2, -1)).reshape(-1, spoke_size)

        # zero filling reconstruction
        # ------------------------
        sino_real = ifft1d(k_data).numpy().real
        sino_imag = ifft1d(k_data).numpy().imag

        fbp_recon_real = iradon(sino_real.T, theta=theta - 90, circle=False, output_size=h)
        fbp_recon_imag = iradon(sino_imag.T, theta=theta - 90, circle=False, output_size=h)
        fbp_recon = np.complex128(fbp_recon_real + 1j * fbp_recon_imag)
        sino_nufft = np.stack([sino_real, sino_imag], axis=-1)

        zf_list.append(np.abs(fbp_recon))
        kdata_list.append(sino_nufft)

    np.savetxt('{}/rot_{}_{}.txt'.format(out_dir, spoke_num, mot), np.array(rot_list))
    np.savetxt('{}/shift_x_{}_{}.txt'.format(out_dir, spoke_num, mot), np.array(shift_x_list))
    np.savetxt('{}/shift_y_{}_{}.txt'.format(out_dir, spoke_num, mot), np.array(shift_y_list))

    sitk.WriteImage(sitk.GetImageFromArray(np.array(gt_list)), '{}/gt.nii'.format(out_dir))
    sitk.WriteImage(sitk.GetImageFromArray(np.array(zf_list)), '{}/zf_{}_{}.nii'.format(out_dir, spoke_num, mot))
    sitk.WriteImage(sitk.GetImageFromArray(np.array(kdata_list)), '{}/kdata_{}_{}.nii'.format(out_dir, spoke_num, mot))