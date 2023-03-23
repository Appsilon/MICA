import os
import sys
from time import sleep

import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

from configs.config import get_cfg_defaults
from jobs import train

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

if __name__ == '__main__':

    custom_defaults_file = "configs/scratch.yml"
    default_cfg = get_cfg_defaults()

    custom_defaults_path = str(Path(default_cfg.mica_dir) / custom_defaults_file)
    default_cfg.merge_from_file(custom_defaults_path)

    param_changes = {
        "base": [],
        "K8": ["dataset.K", 8],
        "K16": ["dataset.K", 16],
        "decay1": ["train.weight_decay", 1.],
        "decay01": ["train.weight_decay", .1],
        "decay001": ["train.weight_decay", .01],
        "face10": ["mask_weights.face", 10.],
        "face2": ["mask_weights.face", 2.],
        "humans_only": ["dataset.training_data", ["VHTRAIN"]],
        "meta_only": ["dataset.training_data", ["MHTRAIN"]],
        "mapping5": ["model.mapping_layers", 5],
        "mapping6_skips": ["model.mapping_layers", 6],
        "hidden512": ["model.hidden_layers_size", 512],
        "arcface2": ["model.arcface_unfreeze", 2],
        "chamfer_loss": ["train.loss_keys", ['pred_chamfer_distance', 'penalty_shape_code']],
        "nshape100": ["model.n_shape", 100],
        "nshape200": ["model.n_shape", 200],
    }

    # Create configs
    experiments = {}
    for name, exp in param_changes.items():
        exp_cfg = default_cfg.clone()
        exp_cfg.merge_from_list(exp)
        exp_cfg.output_dir = os.path.join('./param_sensitivity', f"scratch_{name}")
        experiments[name] = exp_cfg

    cudnn.benchmark = False
    cudnn.deterministic = True

    # Run experiments
    for name, exp_cfg in experiments.items():

        print(f"Running experiment: {name}")
        torch.cuda.empty_cache()

        try:
            train(rank=0, world_size=1, cfg=exp_cfg)
        except Exception as e:
            print(e)
        
        # Give processes time to shutdown
        sleep(30)

    # Shutdown once complete
    os.system("sudo shutdown")