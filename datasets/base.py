# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de


import os
import re
from abc import ABC
from functools import reduce
from pathlib import Path

import loguru
import numpy as np
import torch
from loguru import logger
from skimage.io import imread
from torch.utils.data import Dataset
from torchvision import transforms
from utils import mesh


class BaseDataset(Dataset, ABC):
    def __init__(self, name, config, device, isEval):
        self.K = config.K
        self.use_shape_params = config.use_shape_params
        self.isEval = isEval
        self.n_train = np.Inf
        self.imagepaths = []
        self.face_dict = {}
        self.name = name
        self.device = device
        self.min_max_K = 0
        self.cluster = False
        self.dataset_root = config.root
        self.random_flip = config.random_flip
        self.random_flip_prob = config.random_flip_prob
        self.add_noise = config.add_noise
        self.add_noise_std = config.add_noise_std
        self.total_images = 0
        self.image_folder = 'arcface_input'
        self.flame_folder = 'FLAME_parameters'
        self.obj_folder = 'registrations'
        self.initialize()

    def initialize(self):
        logger.info(f'[{self.name}] Initialization')
        image_list = f'{os.path.abspath(os.path.dirname(__file__))}/image_paths/{self.name}.npy'
        logger.info(f'[{self.name}] Load cached file list: ' + image_list)
        self.face_dict = np.load(image_list, allow_pickle=True).item()
        self.imagepaths = list(self.face_dict.keys())
        logger.info(f'[Dataset {self.name}] Total {len(self.imagepaths)} actors loaded!')
        self.set_smallest_k()

    def set_smallest_k(self):
        self.min_max_K = np.Inf
        max_min_k = -np.Inf
        for key in self.face_dict.keys():
            length = len(self.face_dict[key][0])
            if length < self.min_max_K:
                self.min_max_K = length
            if length > max_min_k:
                max_min_k = length

        self.total_images = reduce(lambda k, l: l + k, map(lambda e: len(self.face_dict[e][0]), self.imagepaths))
        loguru.logger.info(f'Dataset {self.name} with min K = {self.min_max_K} max K = {max_min_k} length = {len(self.face_dict)} total images = {self.total_images}')
        return self.min_max_K

    def compose_transforms(self, *args):
        self.transforms = transforms.Compose([t for t in args])

    def get_arcface_path(self, image_path):
        return re.sub('png|jpg|PNG|JPG', 'npy', str(image_path))

    def __len__(self):
        return len(self.imagepaths)

    def __getitem__(self, index):
        actor = self.imagepaths[index]
        images, params_path = self.face_dict[actor]
        images = [Path(self.dataset_root, self.name, self.image_folder, path) for path in images]
        sample_list = np.array(np.random.choice(range(len(images)), size=self.K, replace=self.K > len(images)))

        K = self.K
        if self.isEval:
            K = max(0, min(200, self.min_max_K))
            sample_list = np.array(range(len(images))[:K])

        if self.use_shape_params:
            params = np.load(os.path.join(self.dataset_root, self.name, self.flame_folder, params_path), allow_pickle=True)
            pose = torch.tensor(params['pose']).float()
            betas = torch.tensor(params['betas']).float()

            flame = {
                'shape_params': torch.cat(K * [betas[:300][None]], dim=0),
                'expression_params': torch.cat(K * [betas[300:][None]], dim=0),
                'pose_params': torch.cat(K * [torch.cat([pose[:3], pose[6:9]])[None]], dim=0),
            }

            images_list = []
            arcface_list = []

            for i in sample_list:
                image_path = images[i]
                image = np.array(imread(image_path))
                image = image / 255.
                image = image.transpose(2, 0, 1)
                arcface_image = np.load(self.get_arcface_path(image_path), allow_pickle=True)

                images_list.append(torch.from_numpy(image.copy()))
                arcface_list.append(torch.from_numpy(arcface_image.copy()))

            images_array = torch.stack(images_list).float()
            arcface_array = torch.stack(arcface_list).float()

        else:
            # Use files
            vertex_path = Path(self.dataset_root, self.name, self.obj_folder, actor)

            vertex_file = list(vertex_path.glob("*.npy"))
            if not vertex_file:
                raise FileNotFoundError(f"Vertex file not found in: {vertex_path}")
            
            vertices = np.load(vertex_file[0])
            
            # Make sure meshes are centred!
            vertices -= vertices.mean(axis=0)

            images_list = []
            arcface_list = []
            vertices_list = []

            for i in sample_list:
                image_path = images[i]
                image = np.array(imread(image_path))
                image = image / 255.
                image = image.transpose(2, 0, 1)
                arcface_image = np.load(self.get_arcface_path(image_path), allow_pickle=True)
                flame_vertices = vertices.copy()

                ### Augmentation ###
                if not self.isEval:
                    
                    if self.random_flip:

                        if np.random.rand() < self.random_flip_prob:
                            image = np.flip(image, axis=-1)
                            arcface_image = np.flip(arcface_image, axis=-1)
                            flame_vertices[:, 0] = -flame_vertices[:, 0]

                    if self.add_noise:
                        noise = np.random.normal(0, self.add_noise_std, arcface_image.shape)
                        arcface_image = np.clip(arcface_image + noise, -1., 1.)

                ####################

                images_list.append(torch.from_numpy(image.copy()))
                arcface_list.append(torch.from_numpy(arcface_image.copy()))
                vertices_list.append(torch.from_numpy(flame_vertices.copy()))

            images_array = torch.stack(images_list).float()
            arcface_array = torch.stack(arcface_list).float()
            flame_array = torch.stack(vertices_list).float()

            flame = {'vertices': flame_array}

        return {
            'image': images_array,
            'arcface': arcface_array,
            'imagename': actor,
            'dataset': self.name,
            'flame': flame,
        }
