# -*- coding: utf-8 -*-
import SimpleITK as sitk
import numpy as np
import torch
import model
import dataset
import tinycudann as tcnn
from torch.utils import data
from torch.optim import lr_scheduler
import utils


def train(config_path, rot, sample_index, spoke_num):

    # load config.yaml
    # -----------------------
    config = utils.load_config(config_path)

    # file
    # -----------------------
    in_path = config["file"]["in_dir"]
    model_path = config["file"]["model_dir"]
    proj_path = '{}/kdata_{}_{}.nii'.format(in_path, spoke_num, rot)
    h, w = config["file"]["h"], config["file"]["w"]
    num_angle, SOD, _ = sitk.GetArrayFromImage(sitk.ReadImage(proj_path))[sample_index].shape
    num_stage = config["file"]["num_stage"]

    # parameter
    # -----------------------
    lr = config["train"]["lr"]
    gpu = config["train"]["gpu"]
    epoch = config["train"]["epoch"]
    save_epoch = config["train"]["save_epoch"]
    lr_decay_epoch = config["train"]["lr_decay_epoch"]
    lr_decay_rate = config["train"]["lr_decay_rate"]
    batch_size = config["train"]["batch_size"]
    num_sample_ray = config["train"]["num_sample_ray"]
    device = torch.device('cuda:{}'.format(str(gpu) if torch.cuda.is_available() else 'cpu'))

    # data
    # -----------------------
    train_loader = data.DataLoader(
        dataset=dataset.TrainData(proj_path=proj_path, num_sample_ray=num_sample_ray, sample_index=sample_index),
        batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(
        dataset=dataset.TestDataDirect(h=SOD, w=SOD), batch_size=1, shuffle=False)

    # model & optimizer
    # -----------------------
    dc_loss = torch.nn.L1Loss().to(device)
    hash_encoder = tcnn.Encoding(n_input_dims=2, encoding_config=config['encoding']).to(device)
    network = tcnn.Network(n_input_dims=hash_encoder.n_output_dims, n_output_dims=2,
                           network_config=config['mlp']).to(device)
    optimizer_hash = torch.optim.Adam(params=hash_encoder.parameters(), lr=lr)
    scheduler_hash = lr_scheduler.StepLR(optimizer_hash, step_size=lr_decay_epoch, gamma=lr_decay_rate)
    optimizer_network = torch.optim.Adam(params=network.parameters(), lr=lr)
    scheduler_network = lr_scheduler.StepLR(optimizer_network, step_size=lr_decay_epoch, gamma=lr_decay_rate)

    # initial pose
    theta_is_train, offset_is_train = True, True
    theta_view = np.deg2rad(np.linspace(0., 180., num=num_angle, endpoint=False)).reshape(num_angle, 1, 1)
    theta = np.zeros(shape=(num_stage, 1, 1), dtype=float)
    offset = np.zeros(shape=(num_stage, 1, 2), dtype=float)

    pose_refiner = model.PoseRefiner(theta=theta, theta_view=theta_view, offset=offset,
                                     theta_is_train=theta_is_train, offset_is_train=offset_is_train).to(device)
    optimizer_pose = torch.optim.Adam(params=pose_refiner.parameters(), lr=lr)
    scheduler_pose = lr_scheduler.StepLR(optimizer_pose, step_size=lr_decay_epoch, gamma=lr_decay_rate)

    for e in range(epoch):
        # mask vectors \alpha
        mask = torch.zeros(size=(1, int(config['encoding']['n_levels'] *
                                        config['encoding']['n_features_per_level']))).to(device)
        # update \alpha as the training process
        if (e+1) < 1000:
            mask[:, :int(4 * config['encoding']['n_features_per_level'])] = 1
        if 1000 < (e+1) < 3000:
            mask[:, :int(10 * config['encoding']['n_features_per_level'])] = 1
        if (e+1) > 3000:
            mask[:, :int(16 * config['encoding']['n_features_per_level'])] = 1
        loss_log = 0
        for i, (pro_id, ray, proj) in enumerate(train_loader):
            ray = ray.to(device).float()  # (batch_size, num_sample_ray, SOD, 2)
            proj = proj.to(device).float()  # (batch_size, num_sample_ray, 2)
            # pose correction
            xyz = pose_refiner(pro_id, ray)  # (batch_size, num_sample_ray, SOD, 2)
            feature = hash_encoder(xyz.view(-1, 2)) * mask
            # intensity prediction  # (batch_size, num_sample_ray, SOD, 2)
            intensity_pre = network(feature).view(-1, num_sample_ray, SOD, 2)
            # forward
            proj_pre = torch.sum(intensity_pre, dim=2)  # (batch_size, num_sample_ray, 2)
            # compute loss
            loss = dc_loss(proj_pre, proj.to(proj_pre.dtype))
            # backward
            optimizer_network.zero_grad()
            optimizer_hash.zero_grad()
            optimizer_pose.zero_grad()
            loss.backward()
            optimizer_network.step()
            optimizer_hash.step()
            optimizer_pose.step()
            # record and print loss
            loss_log += loss.item()
        print('(TRAIN0) Epoch[{}/{}], Lr_pose:{}, Lr_scope:{}, Loss:{:.6f}'.format(e + 1, epoch,
                                                                                   scheduler_pose.get_last_lr()[0],
                                                                                   scheduler_network.get_last_lr()[0],
                                                                                   loss_log/len(train_loader)))
        scheduler_network.step()
        scheduler_hash.step()
        scheduler_pose.step()
        # model save & reconstruction
        if (e + 1) % save_epoch == 0:
            with torch.no_grad():
                torch.save(network.state_dict(), '{}/model_{}_{}_{}.pkl'.format(model_path, spoke_num,
                                                                                rot, sample_index))
                for i, (ray) in enumerate(test_loader):
                    ray = ray.to(device).float().view(-1, 2)    # (SOD*SOD, 2)
                    # forward
                    feature = hash_encoder(ray) * mask
                    img_pre = network(feature).view(SOD, SOD, 2)
                    img_pre = img_pre.float().cpu().detach().numpy()
                # save
                angle_corrected = pose_refiner.theta.view(-1, ).float().cpu().detach().numpy()
                x_corrected = pose_refiner.offset.view(-1, 2)[:, 0].float().cpu().detach().numpy() * (SOD/2)
                y_corrected = pose_refiner.offset.view(-1, 2)[:, 1].float().cpu().detach().numpy() * (SOD/2)

                img_pre = img_pre[int((SOD-h)/2):int((SOD-h)/2)+h, int((SOD-w)/2):int((SOD-w)/2)+w, :]
                img_pre = np.complex128(img_pre[:, :, 0] + 1j*img_pre[:, :, 1])
                img_pre_save = np.abs(img_pre)

    return angle_corrected, x_corrected, y_corrected, img_pre_save