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
import random
import sys
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
from configs.config import cfg
from utils import util

sys.path.append("./micalib")
from micalib.base_model import BaseModel
from micalib.validator import Validator


def print_info(rank):
    props = torch.cuda.get_device_properties(rank)

    logger.info(f'[INFO]            {torch.cuda.get_device_name(rank)}')
    logger.info(f'[INFO] Rank:      {str(rank)}')
    logger.info(f'[INFO] Memory:    {round(props.total_memory / 1024 ** 3, 1)} GB')
    logger.info(f'[INFO] Allocated: {round(torch.cuda.memory_allocated(rank) / 1024 ** 3, 1)} GB')
    logger.info(f'[INFO] Cached:    {round(torch.cuda.memory_reserved(rank) / 1024 ** 3, 1)} GB')


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Trainer(object):
    def __init__(self, nfc_model: BaseModel, config=None, device=None):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config

        logger.add(os.path.join(self.cfg.output_dir, self.cfg.train.log_dir, 'train.log'))

        self.device = device
        self.batch_size = self.cfg.dataset.batch_size
        self.K = self.cfg.dataset.K
        self.prepare_data()

        # deca model
        self.nfc = nfc_model.to(self.device)

        self.validator = Validator(self)
        self.configure_optimizers()

        self.epoch = 0
        self.global_step = 0
        
        if not self.cfg.train.fresh:
            self.load_checkpoint()

        # reset optimizer if loaded from pretrained model
        if self.cfg.train.reset_optimizer:
            self.configure_optimizers()  # reset optimizer
            logger.info(f"[TRAINER] Optimizer was reset")

        if self.cfg.train.write_summary and self.device == 0:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=os.path.join(self.cfg.output_dir, self.cfg.train.log_dir))

        print_info(device)

    def configure_optimizers(self):
        self.opt = torch.optim.AdamW(
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
            params=self.nfc.parameters_to_optimize(),
            amsgrad=False)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=1, gamma=0.1)

    def load_checkpoint(self):
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.device}
        model_path = os.path.join(self.cfg.output_dir, 'model.tar')
        
        if not os.path.exists(model_path) and self.cfg.use_pretrained:
            model_path = self.cfg.pretrained_model_path
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location)
            if 'opt' in checkpoint:
                self.opt.load_state_dict(checkpoint['opt'])
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            if 'epoch' in checkpoint:
                self.epoch = checkpoint['epoch']
            if 'global_step' in checkpoint:
                self.global_step = checkpoint['global_step']
            logger.info(f"[TRAINER] Resume training from {model_path}")
            logger.info(f"[TRAINER] Start from step {self.global_step}")
            logger.info(f"[TRAINER] Start from epoch {self.epoch}")
        else:
            logger.info('[TRAINER] Model path not found, start training from scratch')

    def save_checkpoint(self, filename):
        if self.device == 0:
            model_dict = self.nfc.model_dict()

            model_dict['opt'] = self.opt.state_dict()
            model_dict['scheduler'] = self.scheduler.state_dict()
            model_dict['validator'] = self.validator.state_dict()
            model_dict['epoch'] = self.epoch
            model_dict['global_step'] = self.global_step
            model_dict['batch_size'] = self.batch_size

            torch.save(model_dict, filename)

    def training_step(self, batch):
        self.nfc.train()

        images = batch['image'].to(self.device)
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        flame = batch['flame']
        arcface = batch['arcface']
        arcface = arcface.view(-1, arcface.shape[-3], arcface.shape[-2], arcface.shape[-1]).to(self.device)

        inputs = {
            'images': images,
            'dataset': batch['dataset'][0]
        }

        encoder_output = self.nfc.encode(images, arcface)
        encoder_output['flame'] = flame

        decoder_output = self.nfc.decode(encoder_output, self.epoch)
        metrics = self.nfc.compute_losses(inputs, encoder_output, decoder_output)

        total_loss = 0.
        losses = {}
        for loss_cfg in self.cfg.train.losses:
            
            key = loss_cfg["name"]
            weight = loss_cfg["weight"]

            if key in metrics["masked"]:
                loss = metrics["masked"][key]
            else:
                loss = metrics["regular"][key]
        
            losses[key] = loss * weight
            total_loss += losses[key]

        losses["total"] = total_loss

        opdict = \
            {
                'images': images,
                'flame_verts_shape': decoder_output['flame_verts_shape'],
                'pred_shape_code': decoder_output['pred_shape_code'],
                'pred_canonical_shape_vertices': decoder_output['pred_canonical_shape_vertices'],
            }

        if 'deca' in decoder_output:
            opdict['deca'] = decoder_output['deca']

        return losses, metrics, opdict

    def validation_step(self):
        self.validator.run()

    def evaluation_step(self):
        pass

    def prepare_data(self):
        generator = torch.Generator()
        generator.manual_seed(self.device)

        self.train_dataset, total_images = datasets.build_train(self.cfg.dataset, self.device)
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=generator)

        logger.info(f'[TRAINER] Training dataset is ready with {len(self.train_dataset)} actors and {total_images} images.')

    def fit(self):

        max_epochs = self.steps2epochs(self.cfg.train.max_steps)
        start_epoch = self.epoch
        for epoch in range(start_epoch, max_epochs):

            train_iter = iter(self.train_dataloader)

            for step, batch in enumerate(tqdm(train_iter, total=len(train_iter), desc=f"Epoch[{epoch + 1}/{max_epochs}]")):
                if self.global_step > self.cfg.train.max_steps:
                    break

                visualizeTraining = self.global_step % self.cfg.train.vis_steps == 0

                self.opt.zero_grad()
                losses, all_metrics, opdict = self.training_step(batch)

                losses["total"].backward()
                self.opt.step()
                self.global_step += 1

                if self.global_step % self.cfg.train.log_steps == 0 and self.device == 0:
                    loss_info = f"\n" \
                                f"  Epoch: {epoch}\n" \
                                f"  Step: {self.global_step}\n" \
                                f"  Iter: {step}/{len(train_iter)}\n" \
                                f"  LR: {self.opt.param_groups[0]['lr']}\n" \
                                f"  Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}\n"
                    
                    for k, v in losses.items():
                        
                        loss_info = loss_info + f'  loss_{k}: {v:.4f}\n'

                        if self.cfg.train.write_summary:
                            self.writer.add_scalar('train_metrics_loss/' + k, v, global_step=self.global_step)

                    for metric_type, metrics in all_metrics.items():
                        
                        for k, v in metrics.items():
                            
                            # Avoid repeating metrics used for loss
                            if k not in losses:
                                loss_info = loss_info + f'  {metric_type}_{k}: {v:.4f}\n'

                            if self.cfg.train.write_summary:
                                self.writer.add_scalar(f'train_metrics_{metric_type}/' + k, v, global_step=self.global_step)

                    logger.info(loss_info)

                if visualizeTraining and self.device == 0:

                    pred_canonical_shape_vertices = torch.empty(0, 3, 512, 512).cuda()
                    flame_verts_shape = torch.empty(0, 3, 512, 512).cuda()
                    deca_images = torch.empty(0, 3, 512, 512).cuda()
                    input_images = torch.empty(0, 3, 224, 224).cuda()
                    L = opdict['pred_canonical_shape_vertices'].shape[0]
                    S = 4 if L > 4 else L
                    for n in np.random.choice(range(L), size=S, replace=False):
                        rendering = self.nfc.render.render_mesh(opdict['pred_canonical_shape_vertices'][n:n + 1, ...])
                        pred_canonical_shape_vertices = torch.cat([pred_canonical_shape_vertices, rendering])
                        rendering = self.nfc.render.render_mesh(opdict['flame_verts_shape'][n:n + 1, ...])
                        flame_verts_shape = torch.cat([flame_verts_shape, rendering])
                        input_images = torch.cat([input_images, opdict['images'][n:n + 1, ...]])
                        if 'deca' in opdict:
                            deca = self.nfc.render.render_mesh(opdict['deca'][n:n + 1, ...])
                            deca_images = torch.cat([deca_images, deca])

                    visdict = {}

                    if 'deca' in opdict:
                        visdict['deca'] = deca_images

                    visdict["pred_canonical_shape_vertices"] = pred_canonical_shape_vertices
                    visdict["flame_verts_shape"] = flame_verts_shape
                    visdict["images"] = input_images

                    grid_image = util.visualize_grid(visdict, size=512)

                    self.writer.add_images("train_images/batch_input", np.clip(opdict['images'].detach().cpu(), 0.0, 1.0), self.global_step)
                    self.writer.add_images("train_images/comparison", grid_image[:, :, ::-1], self.global_step, dataformats="HWC")

                    savepath = os.path.join(self.cfg.output_dir, 'train_images/train_' + str(epoch) + '.jpg')
                    cv2.imwrite(savepath, grid_image)

                if self.global_step % self.cfg.train.val_steps == 0:
                    self.validation_step()

                if self.global_step % self.cfg.train.lr_update_step == 0:
                    self.scheduler.step()

                if self.global_step % self.cfg.train.eval_steps == 0:
                    self.evaluation_step()

                if self.global_step % self.cfg.train.checkpoint_steps == 0:
                    self.save_checkpoint(os.path.join(self.cfg.output_dir, 'model' + '.tar'))

                if self.global_step % self.cfg.train.checkpoint_epochs_steps == 0:
                    self.save_checkpoint(os.path.join(self.cfg.output_dir, 'model_' + str(self.global_step) + '.tar'))

            self.epoch += 1

        self.save_checkpoint(os.path.join(self.cfg.output_dir, 'model' + '.tar'))
        logger.info(f'[TRAINER] Fitting has ended!')

    def steps2epochs(self, steps):

        steps_every_epoch = int(len(self.train_dataset) / self.batch_size)
        epochs = int(steps / steps_every_epoch)
        return epochs