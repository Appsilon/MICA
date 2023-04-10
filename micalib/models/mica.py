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
import sys

sys.path.append("./nfclib")

import torch
import torch.nn.functional as F

from models.arcface import Arcface
from models.generator import Generator
from micalib.base_model import BaseModel
from utils.loss import chamfer_distance

from loguru import logger


class MICA(BaseModel):
    def __init__(self, config=None, device=None, tag='MICA'):
        super(MICA, self).__init__(config, device, tag)

        self.initialize()

    def create_model(self, model_cfg):

        pretrained_path = model_cfg.arcface_pretrained_model if model_cfg.arcface_use_pretrained else None
        self.arcface = Arcface(pretrained_path=pretrained_path, unfreeze=model_cfg.arcface_unfreeze).to(self.device)
        self.flameModel = Generator(
            512, 
            model_cfg.hidden_layers_size, 
            model_cfg.n_shape, 
            model_cfg.mapping_layers, 
            model_cfg, 
            self.device)

    def load_model(self):

        model_path = ''

        if self.testing or not self.cfg.train.fresh:
            # Look for existing model first
            model_path = os.path.join(self.cfg.output_dir, 'model.tar')

        if not os.path.exists(model_path) and self.cfg.use_pretrained:
            # Load pretrained model
            model_path = self.cfg.pretrained_model_path

        if os.path.exists(model_path):
            logger.info(f'[{self.tag}] Trained model found. Path: {model_path} | GPU: {self.device}')
            checkpoint = torch.load(model_path)
            if 'arcface' in checkpoint:
                self.arcface.load_state_dict(checkpoint['arcface'])
            if 'flameModel' in checkpoint:
                self.flameModel.load_state_dict(checkpoint['flameModel'])
        else:
            logger.info(f'[{self.tag}] Starting from scratch!')

    def model_dict(self):
        return {
            'flameModel': self.flameModel.state_dict(),
            'arcface': self.arcface.state_dict()
        }

    def parameters_to_optimize(self):
        return [
            {'params': self.flameModel.parameters(), 'lr': self.cfg.train.lr},
            {'params': self.arcface.parameters(), 'lr': self.cfg.train.arcface_lr},
        ]

    def encode(self, images, arcface_imgs):
        codedict = {}

        codedict['arcface'] = F.normalize(self.arcface(arcface_imgs))
        codedict['images'] = images

        return codedict

    def decode(self, codedict, epoch=0):
        self.epoch = epoch

        flame_verts_shape = None
        shapecode = None

        if not self.testing:

            flame = codedict['flame']
            if self.cfg.dataset.use_shape_params:
                
                shapecode = flame['shape_params'].view(-1, flame['shape_params'].shape[2])
                shapecode = shapecode.to(self.device)[:, :self.cfg.model.n_shape]
                with torch.no_grad():
                    flame_verts_shape, _, _ = self.flame(shape_params=shapecode)
            
            else:
                flame_verts_shape = flame['vertices'].view(-1, flame['vertices'].shape[2], flame['vertices'].shape[3]).to(self.device)

        identity_code = codedict['arcface']
        pred_canonical_vertices, pred_shape_code = self.flameModel(identity_code)

        output = {
            'flame_verts_shape': flame_verts_shape,
            'flame_shape_code': shapecode,
            'pred_canonical_shape_vertices': pred_canonical_vertices,
            'pred_shape_code': pred_shape_code,
            'faceid': codedict['arcface']
        }

        return output

    def compute_losses(self, input, encoder_output, decoder_output):
        regular_metrics = {}

        pred_verts = decoder_output['pred_canonical_shape_vertices'] * 1000.0
        gt_verts = decoder_output['flame_verts_shape'].detach() * 1000.0
        pred_shape_code = decoder_output['pred_shape_code']

        if self.cfg.dataset.align_faces:
            face_ids = self.masking.masks.face
            pred_verts -= pred_verts[:, face_ids].mean(axis=1).unsqueeze(1)
            gt_verts -= gt_verts[:, face_ids].mean(axis=1).unsqueeze(1)

        ## Vertex based metrics
        # Require prediction and ground truth with same order
        regular_metrics['mica_distance'] = (pred_verts - gt_verts).abs()
        regular_metrics['pred_distance_l1'] = (pred_verts - gt_verts).abs().sum(-1)
        regular_metrics['pred_distance_l2'] = ((pred_verts - gt_verts) ** 2).sum(-1) ** 0.5
        # Should work with arbitrary point clouds
        regular_metrics['pred_chamfer_distance'], _ = chamfer_distance(pred_verts, gt_verts, norm=self.cfg.train.norm)

        ## Get masked metrics
        masked_metrics = {}
        for key, val in regular_metrics.items():
            if len(val.shape) == 3:
                masked_val = val * self.vertices_mask
            else:
                masked_val = val * self.vertices_mask[:, :, 0]

            masked_metrics[key] = masked_val

        ## Region metrics
        region_metrics = {}
        for key, ids in self.masking.masks.__dict__.items():
            region_metrics[f"{key}_l1"] = regular_metrics['pred_distance_l1'][:, ids]

        ## Shape parameters based metrics
        regular_metrics['std_shape_code'] = pred_shape_code.std()

        if self.cfg.train.max_shape_code < float("inf"):
            
            too_big = pred_shape_code.abs() > self.cfg.train.max_shape_code
            regular_metrics['penalty_shape_code'] = \
                torch.mean(1000.0 * (pred_shape_code[too_big] - self.cfg.train.max_shape_code)**2) \
                if too_big.any() else \
                torch.mean(1e-7 * pred_shape_code[~too_big]**2)
        else:
            regular_metrics['penalty_shape_code'] = torch.tensor(0)

        ## Reduce and combine
        combined_metrics = {
            "regular": {key: torch.mean(val) for key, val in regular_metrics.items()},
            "masked": {key: torch.mean(val) for key, val in masked_metrics.items()},
            "region": {key: torch.mean(val) for key, val in region_metrics.items()},
        }

        return combined_metrics
