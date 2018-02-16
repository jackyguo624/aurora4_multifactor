from torch.utils.data import dataset, dataloader
import torch
a = torch.FloatTensor(2,3)

b = torch.FloatTensor(2,3)

c = torch.IntTensor(2,1)


class NDataset(dataset.Dataset):
    def __init__(self, data_tensor, target1_tensor, target2_tensor):
        assert data_tensor.size(0) == target1_tensor.size(0)
        self.data_tensor = data_tensor
        self.target1_tensor = target1_tensor
        self.target2_tensor = target2_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target1_tensor[index], self.target2_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

ndataset = NDataset(a, b, c)
dataloader = dataloader.DataLoader(ndataset)
for x in dataloader:
    print(x[0], x[1], x[2])
