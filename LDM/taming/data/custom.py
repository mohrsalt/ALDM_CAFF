import os
from torch.utils.data import Dataset

from monai import transforms
from monai.data import Dataset as MonaiDataset


brats_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["t1n", "t1c", "t2w", "t2f"], allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["t1n", "t1c", "t2w", "t2f"], allow_missing_keys=True),
        transforms.Lambdad(keys=["t1n", "t1c", "t2w", "t2f"], func=lambda x: x[0, :, :, :]),
        transforms.AddChanneld(keys=["t1n", "t1c", "t2w", "t2f"]),
        transforms.EnsureTyped(keys=["t1n", "t1c", "t2w", "t2f"]),
        transforms.Orientationd(keys=["t1n", "t1c", "t2w", "t2f"], axcodes="RAI", allow_missing_keys=True),
        transforms.CropForegroundd(keys=["t1n", "t1c", "t2w", "t2f"], source_key="t1n", allow_missing_keys=True),
        transforms.SpatialPadd(keys=["t1n", "t1c", "t2w", "t2f"], spatial_size=(160, 160, 128), allow_missing_keys=True),
        transforms.RandSpatialCropd( keys=["t1n", "t1c", "t2w", "t2f"],
             roi_size=(160, 160, 128),
             random_center=True, 
             random_size=False,
         ),
        transforms.ScaleIntensityRangePercentilesd(keys=["t1n", "t1c", "t2w", "t2f"], lower=0, upper=99.75, b_min=0, b_max=1),
    ]
)

def get_brats_dataset(data_paths):
    transform = brats_transforms 
    
    data = []
    for data_path in data_paths:
        
        for subject in os.listdir(data_path):
            sub_path = os.path.join(data_path, subject)
            if os.path.exists(sub_path) == False: continue
            t1n = os.path.join(sub_path, f"{subject}-t1n.nii.gz") 
            t1c = os.path.join(sub_path, f"{subject}-t1c.nii.gz") 
            t2w = os.path.join(sub_path, f"{subject}-t2w.nii.gz") 
            t2f = os.path.join(sub_path, f"{subject}-t2f.nii.gz") 
            seg = os.path.join(sub_path, f"{subject}-seg.nii.gz")

            data.append({"t1n":t1n, "t1c":t1c, "t2w":t2w, "t2f":t2f, "subject_id": subject})
                    
    print("num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)




class CustomBase(Dataset):
    def __init__(self,data_path):
        super().__init__()
        self.data = get_brats_dataset(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class CustomTrain(CustomBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path)


class CustomTest(CustomBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path)