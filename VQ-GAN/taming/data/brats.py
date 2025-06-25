import os
import numpy as np
import pandas
import random
import torch
from torch.utils.data import Dataset
from omegaconf import ListConfig
from monai import transforms
from monai.data import Dataset as MonaiDataset


def get_transforms(phase="train"):
    modalities = ["t1n", "t1c", "t2w", "t2f"]

    if phase == "train":
        train_transforms = transforms.Compose(
            [
                transforms.RandFlipd(keys=modalities, prob=0.1, spatial_axis=0, allow_missing_keys=True),
                transforms.RandFlipd(keys=modalities, prob=0.1, spatial_axis=1, allow_missing_keys=True),
                transforms.RandFlipd(keys=modalities, prob=0.1, spatial_axis=2, allow_missing_keys=True),
            ]
        )
    
    return transforms.Compose(
        [
            transforms.LoadImaged(keys=modalities, allow_missing_keys=True),
            transforms.AddChanneld(keys=modalities, allow_missing_keys=True),
            transforms.Orientationd(keys=modalities, axcodes="RAS", allow_missing_keys=True),
            transforms.EnsureTyped(keys=modalities, allow_missing_keys=True),
            transforms.CropForegroundd(keys=modalities, source_key="t1n", margin=0, allow_missing_keys=True),
            transforms.SpatialPadd(keys=modalities, spatial_size=(144, 192, 144), allow_missing_keys=True),
            transforms.CenterSpatialCropd(keys=modalities, roi_size=(144, 192, 144), allow_missing_keys=True),
            transforms.ScaleIntensityRangePercentilesd(keys=modalities, lower=0.5, upper=99.5, b_min=-1, b_max=1, allow_missing_keys=True),
            train_transforms if phase == "train" else transforms.Compose([])
        ]
    )


def get_brats_dataset(data_paths, csv_path=None, phase="train"):
    transform = get_transforms(phase=phase)
    
    if isinstance(data_paths, ListConfig):
        data_paths = list(data_paths)
    elif not isinstance(data_paths, list):
        data_paths = [data_paths]

    # Determine subject IDs to include
    datalist = []
    if csv_path is not None:
        df = pandas.read_csv(csv_path)
        for sub_id in df["id"].tolist():
            split_list = df[df["id"] == sub_id]["split"].tolist()
            if split_list and split_list[0] == phase:
                datalist.append(sub_id)
    else:
        for data_path in data_paths:
            all_subjects = os.listdir(data_path)
            if phase == "train":
                datalist.extend(all_subjects)
            else:
                datalist.extend(all_subjects[-10:])  # last 10 per path

    # Build dataset entries
    data = []
    for data_path in data_paths:
        for subject in os.listdir(data_path):
            if subject not in datalist:
                continue
            sub_path = os.path.join(data_path, subject)
            if not os.path.exists(sub_path):
                continue

            t1n = os.path.join(sub_path, f"{subject}-t1n.nii.gz")
            t1c = os.path.join(sub_path, f"{subject}-t1c.nii.gz")
            t2w = os.path.join(sub_path, f"{subject}-t2w.nii.gz")
            t2f = os.path.join(sub_path, f"{subject}-t2f.nii.gz")

            data.append({
                "t1n": t1n,
                "t1c": t1c,
                "t2w": t2w,
                "t2f": t2f,
                "subject_id": subject,
                "path": t1n
            })

    print(phase, " num of subject:", len(data))
    return MonaiDataset(data=data, transform=transform)



class Brain3DBase(Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    
    
class BraTSbase(Brain3DBase):
    def __init__(self, source=None, target=None):
        self.modalities = ["t1n", "t1c", "t2w", "t2f"]
        self.source = source
        self.target = target
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = dict(self.data[i])

        if self.source is None:
            target = random.choice(self.modalities)
            source = [m for m in self.modalities if m != target]

        else:
            source, target = self.source, self.target

        x_tar = item[target]
        x_src_1 = item[source[0]]
        x_src_2 = item[source[1]]
        x_src_3 = item[source[2]]
        input=torch.concat([x_src_1,x_src_2,x_src_3],dim=0) #check axis
        item["source"] = input
        item["target"] = x_tar
        item["target_class"] = torch.tensor(self.modalities.index(target))
        item["sources_list"]=source
        
        item["t_list"]=target

        return item


class BraTS2021Train(BraTSbase):
    def __init__(self, data_path, csv_path=None, phase="train"):
        super().__init__()
        self.data = get_brats_dataset(data_path, csv_path, phase)

class BraTS2021Test(BraTSbase):
    def __init__(self, data_path, csv_path=None, phase="test", source=None, target=None):
        super().__init__()
        self.data = get_brats_dataset(data_path, csv_path, phase)