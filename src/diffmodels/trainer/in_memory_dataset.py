from typing import Optional
from tqdm import tqdm

from torch.utils.data import IterableDataset, Dataset

class InMemoryDataset(Dataset):
    def __init__(self, iterable_ds: IterableDataset, use_tqdm : bool = False, device : Optional[str] = None, repeat_dataset : int = 1):
        device = device if device is not None else "cpu"
        it = iterable_ds if not use_tqdm else tqdm(iterable_ds, desc=f"Caching tensors in {device}")
        self.data = []
        for _ in range(repeat_dataset):
            for x in it:
                self.data.append(x.to(device))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def cache_iterable_in_memory(iterable_ds: IterableDataset, use_tqdm : bool = False, device : Optional[str] = None, repeat_dataset : int = 1):
    return InMemoryDataset(iterable_ds, use_tqdm, device, repeat_dataset)
