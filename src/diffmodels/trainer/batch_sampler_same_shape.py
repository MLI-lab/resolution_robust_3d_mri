from torch.utils.data import Sampler
from math import ceil
import random

class BatchSamplerSameShape(Sampler):
    r"""Yield a mini-batch of indices. The sampler will drop the last batch of
            an image size bin if it is not equal to ``batch_size``

    Args:
        examples (dict): List from dataset class.
        batch_size (int): Size of mini-batch.
    """

    def __init__(self, dataset, batch_size, indices=None,
        shuffle=False, group_shape_by='target',
        data_provides_pseudoinverse : bool = False,
        data_provides_measurement : bool = False,
        ):
        self.batch_size = batch_size
        self.data = {}
        self.shuffle = shuffle
        #try:
            #self.indices = range(len(dataset.raw_samples)) if indices is None else indices
        #except AttributeError:
        self.indices = range(len(dataset)) if indices is None else indices

        for idx in self.indices:
            #item = dataset.raw_samples[idx]
            try:
                # for fastMRI datasets this is more efficient
                item = dataset.raw_samples[idx]
                if group_shape_by == 'target':
                    shape = item.metadata['target_shape']
                elif group_shape_by == 'kspace_and_target':
                    shape = item.metadata['kspace_shape'] + item.metadata['target_shape']
                else:
                    raise NotImplementedError(f'group_shape_by = \'{group_shape_by}\' not implemented')
            except:
                item = dataset[idx] # either contains kspace, target as first two, or only target
                target_shape = item.shape if not data_provides_pseudoinverse and not data_provides_measurement else item[1].shape
                obs_shape = item[0].shape if data_provides_measurement else None
                if group_shape_by == 'target':
                    shape = target_shape
                elif group_shape_by == 'kspace_and_target':
                    shape = obs_shape + target_shape
                else:
                    raise NotImplementedError(f'group_shape_by = \'{group_shape_by}\' not implemented')

            if shape in self.data:
                self.data[shape].append(idx)
            else:
                self.data[shape] = [idx]

        self.total = 0
        for shape, indices in self.data.items():
            self.total += ceil(len(indices) / self.batch_size)
            
    def __iter__(self):
        batches = []

        for _, indices in self.data.items():
            if self.shuffle:
                random.shuffle(indices)                
            batch = []
            for idx in indices:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    batches.append(batch)
                    batch = []
            if batch:
                batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return self.total
    
#class DistributedBatchSamplerSameShape(DistributedSampler):
    #def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 #rank: Optional[int] = None, shuffle: bool = True,
                 #seed: int = 0, drop_last: bool = True, batch_size = 1, group_shape_by='target') -> None:
        #super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed,
                         #drop_last=drop_last)
        #self.batch_size = batch_size
        #self.group_shape_by = group_shape_by

    #def __iter__(self):
        #indices = list(super().__iter__())
        #batch_sampler = BatchSamplerSameShape(self.dataset, batch_size=self.batch_size, indices=indices, shuffle=self.shuffle, group_shape_by=self.group_shape_by)
        #return iter(batch_sampler)

    #def __len__(self) -> int:
        #return self.num_samples//self.batch_size # lower bound