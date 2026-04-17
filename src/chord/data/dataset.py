import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader


class CrossSpeciesDataset(Dataset):
    def __init__(self, species_data: dict):
        """
        Dataset wrapper for joint multi-species training.

        Args:
            species_data:
                Dictionary mapping species name to a tuple of
                (data, labels).

                Example:
                    {
                        "human": (human_data, human_labels),
                        "mouse": (mouse_data, mouse_labels),
                    }

        Notes:
            - Each species can have a different number of samples.
            - The dataset length is defined as the maximum sample count
              across species.
            - During indexing, species with fewer samples are cycled
              through using modulo indexing.
        """
        self.species_data = species_data
        self.species_names = list(species_data.keys())
        
        # Record the number of samples for each species
        # and use the maximum length as the dataset length
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

            # Cycle through samples for shorter species so that all
            # species can be retrieved for every global index
            sp_idx = idx % self.species_lengths[sp]
            batch[sp] = (data[sp_idx], labels[sp_idx])
        return batch


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    pin_memory: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create a training DataLoader for the cross-species dataset.

    This function uses a custom collate function to assemble batches
    into the format:
        {
            species_name: (batched_data, batched_labels),
            ...
        }

    Args:
        dataset:
            A dataset whose each item is a dictionary mapping species
            name to a (data, label) tuple.
        batch_size:
            Number of samples per batch.
        shuffle:
            Whether to shuffle the global dataset indices.
        pin_memory:
            Whether to pin memory in the DataLoader.
        num_workers:
            Number of worker processes used for data loading.

    Returns:
        A PyTorch DataLoader with a species-aware collate function.
    """
    
    def collate_fn(batch):
        # Collect and stack samples for each species dynamically
        collated = {}

        # Assume every sample in the batch has the same set of species keys
        species_names = batch[0].keys()
        
        for sp in species_names:
            # Stack data tensors and label tensors separately for this species
            sp_data = torch.stack([item[sp][0] for item in batch])
            sp_labels = torch.stack([item[sp][1] for item in batch])
            collated[sp] = (sp_data, sp_labels)
        
        return collated
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=num_workers
    )


def create_val_loaders(species_data, batch_size: int, num_workers: int = 4):
    """
    Create per-species validation DataLoaders without cyclic resampling.

    Unlike the training dataset, this function keeps each species as an
    independent DataLoader containing only its real samples.

    Args:
        species_data:
            Dictionary mapping species name to a tuple of
            (data, labels).
        batch_size:
            Number of samples per validation batch.
        num_workers:
            Number of worker processes used for data loading.

    Returns:
        Dictionary of the form:
            {
                species_name: DataLoader,
                ...
            }
    """
    loaders = {}
    for sp, (data, labels) in species_data.items():
        ds = TensorDataset(data, labels)          # Use only the original samples
        loaders[sp] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,                        # Keep validation order fixed
            pin_memory=True,
            num_workers=num_workers,
            drop_last=False                       # Keep the final incomplete batch
        )
    return loaders