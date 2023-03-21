import numpy as np
from pathlib import Path

from datasets.creation.instances.vh import VirtualHumans
from datasets.creation.generator import Generator
from configs.config import cfg

np.random.seed(42)


if __name__ == "__main__":
    
    dataset_path = str((Path(cfg.mica_dir) / ".." / "datasets").resolve())

    VhTrain = VirtualHumans(
        name="VHTRAIN",
        src="/mnt/disks/data/APPSILON_Skans_Metahumans/Scans/train",
        dst=dataset_path,
        det_score=0.5
    )

    VhValid = VirtualHumans(
        name="VHVALID",
        src="/mnt/disks/data/APPSILON_Skans_Metahumans/Scans/valid",
        dst=dataset_path,
        det_score=0.5
    )

    MhTrain = VirtualHumans(
        name="MHTRAIN",
        src="/mnt/disks/data/APPSILON_Skans_Metahumans/Metahumans/train",
        dst=dataset_path,
        det_score=0.5
    )

    MhValid = VirtualHumans(
        name="MHVALID",
        src="/mnt/disks/data/APPSILON_Skans_Metahumans/Metahumans/valid",
        dst=dataset_path,
        det_score=0.5
    )

    generator = Generator([VhTrain, VhValid, MhTrain, MhValid])
    
    # Comment out as needed
    generator.copy()
    generator.preprocess()
    generator.arcface()
    generator.postprocess()

