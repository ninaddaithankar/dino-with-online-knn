import numpy as np

from datasets import load_dataset
from torch.utils.data import Dataset

from tqdm import tqdm

#credit: https://huggingface.co/datasets/imagenet-1k'
#NOTE: if you are having issues with this dataloader and perms you need to add your HF token
#see these links - https://discuss.huggingface.co/t/imagenet-1k-is-not-available-in-huggingface-dataset-hub/25040 https://huggingface.co/docs/hub/security-tokens
class ImageNetDataset(Dataset):
    def __init__(self, hparams, split, transform, dataset_dir=None, n_samples_per_class=-1):
        self.transform = transform
        split = 'validation' if split in ["valid", "val", "validate"] else split
        self.ds = load_dataset("imagenet-1k", split=split)

        # -- select n samples per class to reduce dataset size as needed
        self.n_samples_per_class = n_samples_per_class
        if n_samples_per_class > 0:
            self.ds = self._filter_by_samples_per_class(self.ds, n_samples_per_class)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        sample = self.ds[idx]
        image = sample['image']
        image_mode = image.mode
        if image_mode == 'L':
            image = image.convert("RGB")
        elif image_mode == 'RGBA':
            image = image.convert("RGB")
        
        transformed_image = self.transform(image)
        return transformed_image, sample['label']
    
    def _filter_by_samples_per_class(self, dataset, samples_per_class):
        print(f"Filtering ImageNet dataset to contain {samples_per_class} samples per class.")

        labels = np.array(dataset["label"])
        selected_indices = []

        for label in tqdm(np.unique(labels), desc="Filtering dataset"):
            label_indices = np.where(labels == label)[0]
            if len(label_indices) > samples_per_class:
                selected_indices.extend(np.random.choice(label_indices, samples_per_class, replace=False))
            else:
                selected_indices.extend(label_indices)

        return dataset.select(selected_indices)


import torch

class ImageNetSequentialClips(ImageNetDataset):
    def __init__(self, hparams, split, transform, dataset_dir=None, n_samples_per_class=-1):
        super().__init__(hparams, split, transform, dataset_dir=dataset_dir, n_samples_per_class=n_samples_per_class)
        self.num_frames = hparams.context_length
        self.num_clips = len(self.ds) // self.num_frames

        # build an index list; this will be reshuffled every epoch via sampler behavior
        self.indices = list(range(len(self.ds)))

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        start = idx * self.num_frames
        end = start + self.num_frames
        frame_indices = self.indices[start:end]

        # get frames using parent class __getitem__
        frames = [super().__getitem__(i)[0] for i in frame_indices]
        clip = torch.stack(frames)   # (T, C, H, W)
        return clip, -1
