import random

import numpy as np
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class CrossSpeciesDataset(Dataset):
    def __init__(self, species_data: dict):
        self.species_data = species_data
        self.species_names = list(species_data.keys())
        
        self.species_lengths = {
            sp: data.shape[0] for sp, (data, labels) in species_data.items()
        }
        self.max_len = max(self.species_lengths.values())

    def __len__(self):
        return self.max_len

    def __getitem__(self, idx):
        batch = {}
        for sp in self.species_names:
            data, labels = self.species_data[sp]

            sp_idx = idx % self.species_lengths[sp]
            batch[sp] = (data[sp_idx], labels[sp_idx])
        return batch


def collate_species_batch(batch):
    collated = {}
    for species in batch[0].keys():
        data = torch.stack([item[species][0] for item in batch])
        labels = torch.stack([item[species][1] for item in batch])
        collated[species] = (data, labels)
    return collated


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    pin_memory: bool = True,
    num_workers: int = 4,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_species_batch,
        pin_memory=pin_memory,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def create_val_loaders(species_data, batch_size: int, num_workers: int = 4):
    loaders = {}
    for sp, (data, labels) in species_data.items():
        ds = TensorDataset(data, labels)
        loaders[sp] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=False
        )
    return loaders
