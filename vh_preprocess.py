import numpy as np
import pandas as pd
import shutil
from pathlib import Path
from datasets.creation.instances.vh import VirtualHumans
from datasets.creation.generator import Generator
from configs.config import cfg

np.random.seed(42)

SRC_PATH = Path("/mnt/disks/data/data")
METADATA = pd.read_csv(SRC_PATH / "photos.csv")

METADATA["image_name"] = METADATA["photo_path"].str.rsplit("/", n=1).str.get(-1)

DATASET_ROOT = Path(cfg.dataset.root)
IMG_SRC_DIR = "Photos"
IMG_DST_DIR = "images"
REG_DST_DIR = "registrations"
TRAIN_DIRNAME = "train"
VALID_DIRNAME = "valid"
OVERWRITE = True
VAL_SPLIT = 0.2
ROT_THRESHOLD = None
DET_SCORE = 0.5


def move_data(subject_paths: list[Path], dst: Path):
    
    img_path = dst / IMG_DST_DIR
    reg_path = dst / REG_DST_DIR
    img_path.mkdir(exist_ok=True)
    reg_path.mkdir(exist_ok=True)

    for subject_path in subject_paths:
        
        subject_img_src = subject_path / IMG_SRC_DIR
        subject_img_dst = img_path / subject_path.name
        shutil.copytree(subject_img_src, subject_img_dst)

        subject_reg_dst = reg_path / subject_path.name
        subject_reg_dst.mkdir(exist_ok=True)
        for obj in subject_path.glob("*.obj"):
            shutil.copy(obj, subject_reg_dst / obj.name)


def make_train_valid(data_path: Path, train_dir: str = "train", valid_dir: str = "valid", overwrite: bool = True):
    assert data_path.exists()
    print(f"Making dataset from: {data_path}")

    # Only include subjects with a populated img_src directory and obj file
    subject_paths = [p for p in data_path.glob("*") if (p / IMG_SRC_DIR).exists() and list((p / IMG_SRC_DIR).glob("*")) and list(p.glob("*.obj"))]
    print(f"Total subjects: {len(subject_paths)}")

    # Random train/valid split
    np.random.shuffle(subject_paths)
    split_id = int((1 - VAL_SPLIT) * len(subject_paths))
    train_subjects, val_subjects = subject_paths[:split_id], subject_paths[split_id:]
    print(f"Train: {len(train_subjects)}, valid: {len(val_subjects)}")

    train_path = data_path / train_dir
    valid_path = data_path / valid_dir

    for subset_path, subjects in ((train_path, train_subjects), (valid_path, val_subjects)):

        if subset_path.exists():
            if overwrite:
                shutil.rmtree(subset_path)
            else:
                print(f"Subset path: {subset_path} exists and overwrite is False, skipping data move.")
                continue

        subset_path.mkdir()
        move_data(subjects, subset_path)

    return train_path, valid_path


def purge_over_rotated_images(dataset_path: Path, deg_threshold: float = 5., n_keep: int = 1):

    img_path = dataset_path / IMG_DST_DIR
    purge_imgs = []

    # Loop over all subjects
    for subject_path in img_path.glob("*"):

        print(f"Purging over rotated images for subject: {subject_path.name}")
        subject_metadata = METADATA[METADATA["entity_id"] == subject_path.name]

        # Loop over all subject images
        subject_imgs = list(subject_path.glob("*"))

        n_removed = 0
        for img in subject_imgs:
            
            img_metadata = subject_metadata[subject_metadata["image_name"] == img.name]
            rpy = img_metadata[["roll", "pitch", "yaw"]].abs().to_numpy()

            # Removing images with larger rotation than threshold
            if img_metadata.shape[0] and (np.isnan(rpy).all() or (rpy > deg_threshold)).any():
                
                purge_imgs.append(img)
                n_removed += 1
        
        if len(subject_imgs) - n_removed >= n_keep:
            print(f"Removing {n_removed}/{len(subject_imgs)} ({n_removed/len(subject_imgs):.2f})")
        else:
            raise ValueError(f"Purging for subject {subject_path.name} would remove too many images ({n_removed}/{len(subject_imgs)} > {n_keep})")

    return purge_imgs


def purge_over_rotated_images2(dataset_path: Path, deg_threshold: float = 5., n_keep: int = 1):

    img_path = dataset_path / IMG_DST_DIR
    purge_imgs = []
    rpy_cols = ["roll", "pitch", "yaw"]
    METADATA["purge_image"] = METADATA[rpy_cols].isna().any(axis=0) | METADATA[rpy_cols].max() > deg_threshold

    # Loop over all subjects
    for subject_path in img_path.glob("*"):

        print(f"Purging over rotated images for subject: {subject_path.name}")
        subject_metadata = METADATA[METADATA["entity_id"] == subject_path.name]
        # subject_metadata = subject_metadata.dropna(["roll", "pitch", "yaw"])

        # Loop over all subject images
        subject_imgs = list(subject_path.glob("*"))

        n_removed = 0
        for img in subject_imgs:
            
            img_metadata = subject_metadata[subject_metadata["image_name"] == img.name]
            rpy = img_metadata[["roll", "pitch", "yaw"]].abs().to_numpy()

            # Removing images with larger rotation than threshold
            if img_metadata.shape[0] and (np.isnan(rpy).all() or (rpy > deg_threshold)).any():
                
                purge_imgs.append(img)
                n_removed += 1
        
        if len(subject_imgs) - n_removed >= n_keep:
            print(f"Removing {n_removed}/{len(subject_imgs)} ({n_removed/len(subject_imgs):.2f})")
        else:
            raise ValueError(f"Purging for subject {subject_path.name} would remove too many images ({n_removed}/{len(subject_imgs)} > {n_keep})")

    return purge_imgs


if __name__ == "__main__":
    
    # TODO: Make sure dst are deleted when overwriting too!
    scans_path = SRC_PATH / "Scans"
    meta_path = SRC_PATH / "Metahumans"

    scans_train_path, scans_valid_path = make_train_valid(scans_path, train_dir=TRAIN_DIRNAME, valid_dir=VALID_DIRNAME, overwrite=OVERWRITE)
    meta_train_path, meta_valid_path = make_train_valid(meta_path, train_dir=TRAIN_DIRNAME, valid_dir=VALID_DIRNAME, overwrite=OVERWRITE)

    VhTrain = VirtualHumans(
        name="VHTRAIN",
        src=scans_train_path,
        det_score=DET_SCORE
    )

    VhValid = VirtualHumans(
        name="VHVALID",
        src=scans_valid_path,
        det_score=DET_SCORE
    )

    MhTrain = VirtualHumans(
        name="MHTRAIN",
        src=meta_train_path,
        det_score=DET_SCORE
    )

    MhValid = VirtualHumans(
        name="MHVALID",
        src=meta_valid_path,
        det_score=DET_SCORE
    )

    datasets = [VhTrain, VhValid, MhTrain, MhValid]

    if ROT_THRESHOLD is not None:
        
        purge_imgs = []
        for d in datasets:
            if OVERWRITE and Path(d.dst).exists():
                shutil.rmtree(d.dst)
            
            purge_imgs.extend(purge_over_rotated_images(Path(d.src), ROT_THRESHOLD, n_keep=2))

        for img in purge_imgs:
            img.unlink()

    generator = Generator(datasets)
    
    # Comment out as needed
    generator.copy()
    generator.preprocess()
    generator.arcface()
    generator.postprocess()
