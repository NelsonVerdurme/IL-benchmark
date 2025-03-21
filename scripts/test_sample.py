from time import sleep
import torch
import random
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from itertools import cycle

torch.manual_seed(42)
random.seed(42)

dataset = TensorDataset(torch.arange(10))
sampler = RandomSampler(dataset, replacement=True)
dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)


class InfIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iter_obj = iter(dataloader)

    def next_batch(self):
        try:
            return next(self.iter_obj)
        except StopIteration:
            self.iter_obj = iter(self.dataloader)  # Reset iterator
            return next(self.iter_obj)
        
iter_obj = InfIterator(dataloader)

for i in range(100):
    print(iter_obj.next_batch())
    sleep(1)