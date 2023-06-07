import numpy as np
from abc import ABC
from glob import glob
from pathlib import Path

from datasets.creation.instances.instance import Instance


class Example(Instance, ABC):
    def __init__(self):
        super(Example, self).__init__()
        self.dst = '/datasets/example/'
        self.src = '/example_dataset/'

    def preprocess(self):
        
        paths_file = (Path(__file__).parents[2] / "image_paths" / self.__class__.__name__.upper()).with_suffix(".npy")

        if not paths_file.exists():
            images = self.get_images()
            flame_params = self.get_flame_params()

            metadata = {}
            for subject, imgs in images.items():
                # Hacky way of going from absolute to relative path
                imgs = list(map(lambda x: "/".join(x.split('/')[-2:]), imgs))
                metadata[subject] = (np.random.choice(imgs, size=len(imgs), replace=False), flame_params[subject])

            np.save(paths_file, metadata)

    def get_images(self):
        images = {}
        for actor in sorted(glob(self.get_src() + 'images/*')):
            images[Path(actor).stem] = glob(f'{actor}/*.jpg')

        return images

    def get_flame_params(self):
        params = {}
        for actor in sorted(glob(self.get_src() + 'FLAME_parameters/*')):
            param_paths = [sorted(glob(f'{actor}/*.npz'))] 
            params[Path(actor).stem] = param_paths[0] if param_paths else None

        return params

    def get_registrations(self):
        registrations = {}
        for actor in sorted(glob(self.get_src() + 'registrations/*')):
            registrations[Path(actor).stem] = [f'{actor}/*.obj']

        return registrations