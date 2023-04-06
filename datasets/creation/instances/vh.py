import numpy as np
from abc import ABC
from glob import glob
from pathlib import Path

from datasets.creation.instances.instance import Instance
from configs.config import cfg


class VirtualHumans(Instance, ABC):
    def __init__(self, name: str,  src: str, dst: str = cfg.dataset.root, det_score: float = 0.75):
        super(VirtualHumans, self).__init__()
        self.name = name.upper()
        self.src = src
        self.dst = str(Path(dst) / self.name)
        self.det_score = det_score

    def get_min_det_score(self):
        return self.det_score

    def postprocess(self):
        
        paths_file = (Path(cfg.mica_dir) / "datasets" / "image_paths" / self.name).with_suffix(".npy")

        images = self.get_images(arcface=True)
        flame_params = self.get_flame_params()

        metadata = {}
        for subject, imgs in images.items():
            # Hacky way of going from absolute to relative path
            imgs = list(map(lambda x: "/".join(x.split('/')[-2:]), imgs))
            # TODO: Explain why this is needed (no faces found etc.)
            if imgs:
                metadata[subject] = (np.random.choice(imgs, size=len(imgs), replace=False), "none") #, flame_params[subject])

        np.save(paths_file, metadata)

    def get_images(self, arcface=False):

        if arcface:
            img_path = Path(self.get_dst()) / 'arcface_input'
        else:
            img_path = Path(self.get_src()) / 'images'

        images = {}
        for actor in sorted(img_path.glob("*")):
            images[Path(actor).stem] = glob(f'{actor}/*.JPG') + glob(f'{actor}/*.PNG') + glob(f'{actor}/*.jpg') + glob(f'{actor}/*.png')

        return images

    def get_flame_params(self):
        params = {}
        for actor in sorted(glob(str(Path(self.get_src()) / 'FLAME_parameters/*'))):
            param_paths = [sorted(glob(f'{actor}/*.npz'))] 
            params[Path(actor).stem] = param_paths[0] if param_paths else None

        return params

    def get_registrations(self):
        registrations = {}
        for actor in sorted(glob(str(Path(self.get_src()) / 'registrations/*'))):
            registrations[Path(actor).stem] = glob(f'{actor}/*.obj')

        return registrations


if __name__ == "__main__":
    pass