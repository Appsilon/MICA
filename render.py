import cv2
import os
import torch
import numpy as np

from configs.config import cfg
from micalib.renderer import MeshShapeRenderer
from models.flame import FLAME
from typing import Optional
from utils import util


renderer = MeshShapeRenderer(obj_filename=cfg.model.topology_path)


def render(images: torch.Tensor, preds: torch.Tensor, truths: Optional[torch.Tensor] = None):
    
    pred_canonical_shape_vertices = torch.empty(0, 3, 512, 512).cuda()
    # flame_verts_shape = torch.empty(0, 3, 512, 512).cuda()
    # deca_images = torch.empty(0, 3, 512, 512).cuda()
    # input_images = torch.empty(0, 3, 224, 224).cuda()
    # L = opdict['pred_canonical_shape_vertices'].shape[0]
    # S = 4 if L > 4 else L
    # for n in np.random.choice(range(L), size=S, replace=False):
    #     rendering = renderer.render_mesh(opdict['pred_canonical_shape_vertices'][n:n + 1, ...])
    #     pred_canonical_shape_vertices = torch.cat([pred_canonical_shape_vertices, rendering])
    #     rendering = renderer.render_mesh(opdict['flame_verts_shape'][n:n + 1, ...])
    #     flame_verts_shape = torch.cat([flame_verts_shape, rendering])
    #     input_images = torch.cat([input_images, opdict['images'][n:n + 1, ...]])

    visdict = {}
    visdict["images"] = images

    for n in range(images.shape[0]):
        rendering = renderer.render_mesh(preds[n:n + 1, ...])
        pred_canonical_shape_vertices = torch.cat([pred_canonical_shape_vertices, rendering])
    
    visdict["pred_canonical_shape_vertices"] = pred_canonical_shape_vertices

    if truths is not None:
        flame_verts_shape = torch.empty(0, 3, 512, 512).cuda()

        for n in range(images.shape[0]):
            rendering = renderer.render_mesh(truths[n:n + 1, ...])
            flame_verts_shape = torch.cat([flame_verts_shape, rendering])

        visdict["flame_verts_shape"] = flame_verts_shape

    grid_image = util.visualize_grid(visdict, size=512)
    return grid_image