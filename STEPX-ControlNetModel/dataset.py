import torch
import os
import numpy as np
import nibabel as nib
import albumentations as A
import cv2
from torch.utils.data import Dataset, DataLoader
import random
import time
from tqdm import tqdm
import h5py

from collections import defaultdict

__all_phases__ = ["delayed", "noncontrast", "portal-venous"]

def find_duplicate_indices(tuples_list):
    index_dict = defaultdict(list)
    for idx, tpl in enumerate(tuples_list):     # Iterate through the list once and store indices
        index_dict[tpl].append(idx)
    return {key: value for key, value in index_dict.items() if len(value) > 1}  # Extract only duplicates

def varifyh5(filename): # read the h5 file to see if the conversion is finished or not
    try:
        with h5py.File(filename, "r") as hf:   # can read successfully
            pass
        return True
    except OSError:     # transform not complete
        return False


def load_CT_slice(ct_path, slice_idx=None):
    """For AbdomenAtlasPro data: ranging from [-1000, 1000], shape of (H W D) """
    with h5py.File(ct_path, 'r') as hf:
        nii = hf['image']
        z_shape = nii.shape[2]

        # NOTE: take adjacent 3 slices into the 3 RGB channel
        if slice_idx is None:
            slice_idx = random.randint(0, z_shape - 3)   # `random.randint` includes end point
        # while True:
        #     try:    # some slices of some CT are broken
        ct_slice = nii[:, :, slice_idx:slice_idx + 3]   
            #     break
            # except: # if broken, randomly select until select the non-broken slice
            #     print(f"\033[31mBroken slice: {ct_path.split('/')[-2]}, slice {slice_idx}\033[0m")
            #     slice_idx = random.randint(0, z_shape - 3)

    # ct_slice = np.repeat(ct_slice, repeats=3, axis=2)    # (H W 1) -> (H W 3)
            
    # target range: [-1000, 1000] -> [-1, 1]
    ct_slice[ct_slice > 1000.] = 1000.    # clipping range and normalize
    ct_slice[ct_slice < -1000.] = -1000.
    ct_slice = (ct_slice + 1000.) / 2000.       # [-1000, 1000] --> [0, 1]
    return ct_slice  # (H W 3)[0, 1]

class HWCarrayToCHWtensor(A.ImageOnlyTransform):
    """Converts (H, W, C) NumPy array to (C, H, W) PyTorch tensor."""
    def apply(self, img, **kwargs):
        return torch.from_numpy(img).permute(2, 0, 1)  # (H, W, C) → (C, H, W)
    def get_transform_init_args_names(self):
        return ()

class CTPhaseDataset(Dataset):
    def __init__(self, file_paths, data_root, image_transforms=None, tokenizer=None):
        """ (training only)
        Args:
            file_paths (list): List of paths to 3D CT volumes (.nii.gz).
            transform (albumentations.Compose): Transformations to apply to 2D slices.
        NOTE:
            input a phase A ct slice and condition with a phase B ct slice,
            use the text prompt to guide the model to generate pgase ct slice. (cond phase -> input phase)
        """
        self.file_paths = file_paths
        self.data_root = data_root
        self.image_transforms = image_transforms
        self.tokenizer = tokenizer
        self.phases = __all_phases__

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        patient_id, phase_set = self.file_paths[idx]  # random CT

        # choose input-cond pair (can be the same)
        chosen_shape = random.choice(list(phase_set.keys()))    # choose a resolution and its pairs
        chosen_sublist = phase_set[chosen_shape]   # for all same resolution pairs of a patient, choose one
        random.shuffle(chosen_sublist)              # shuffle the same resolution pairs
        chosen_phases_idx = random.choices(chosen_sublist, k=2) # 1st as input, 2nd as cond

        input_ct_phase = self.phases[chosen_phases_idx[0]]
        input_ct_path = os.path.join(self.data_root, f"{patient_id}_{input_ct_phase}", "ct.h5")
        cond_ct_phase = self.phases[chosen_phases_idx[1]]
        cond_ct_path = os.path.join(self.data_root, f"{patient_id}_{cond_ct_phase}", "ct.h5")

        z_shape = eval(chosen_shape)[2] # `eval()` to make string '(512, 512, z_shape)' to tuple (512, 512, z_shape)
        slice_idx = random.randint(0, z_shape - 3)
        # while True:
        #     try:    # make sure extracting the same slice
        input_ct_slice = load_CT_slice(input_ct_path, slice_idx=slice_idx)
        cond_ct_slice = load_CT_slice(cond_ct_path, slice_idx=slice_idx)
            #     break
            # except: # if broken, randomly select until select the non-broken slice
            #     print(f"\033[31mBroken slice: {input_ct_path.split('/')[-2]}, {cond_ct_path.split('/')[-2]}, slice {slice_idx}\033[0m")
            #     slice_idx = random.randint(0, z_shape - 3)

        # ct_slice = load_CT_slice(os.path.join(ct_path, "ct.h5"))    # random slice
        transformed_images = self.image_transforms(image=input_ct_slice)    
        replay_trans = transformed_images["replay"]
        cond_ct_slice = A.ReplayCompose.replay(replay_trans, image=cond_ct_slice)["image"]  
        input_ct_slice = transformed_images["image"]
        text_prompt = f"A {input_ct_phase} CT slice."  # guide the model to generate input phase CT (from cond phase)

        example = dict()
        example["pixel_values"] = input_ct_slice
        example["conditioning_pixel_values"] = cond_ct_slice
        example["input_ids"] = self.tokenize_caption(text_prompt)

        return example  # Shape: (C, H, W)
    
    def tokenize_caption(self, text_prompt, is_train=True):
        captions = text_prompt
        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids 
    


if __name__ == "__main__":
    train_data_dir = "/mnt/T9/AbdomenAtlas/image_mask_h5"
    paths = sorted([entry.path for entry in os.scandir(train_data_dir)
                            if entry.name.startswith("BDMAP_A") or entry.name.startswith("BDMAP_V")])
    paths = [entry.path.replace("ct.h5", "") for path in  paths
                                            for entry in os.scandir(path) if entry.name == "ct.h5"]
    print(len(paths), "CT scans found!")


    train_transforms = A.Compose([
        A.Resize(512, 512, interpolation=cv2.INTER_LINEAR),
        A.RandomResizedCrop((512, 512), scale=(0.75, 1.0), ratio=(1., 1.), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        HWCarrayToCHWtensor(p=1.),
    ])

    train_dataset = CTDataset(paths, transform=train_transforms)

    def collate_fn(examples):
        # pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = torch.stack([example for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        # input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values}#, "input_ids": input_ids}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=1,
        pin_memory=True
    )

    for batch in tqdm(train_dataloader):
        batch = batch["pixel_values"]