from torch.utils.data.dataset import Dataset


class MultifactorDataset(Dataset):
    '''This Dataset wrap 
       feature: multi-feats
       targets: spk_id, phone_id, clean-feats
    '''
    def __init__(self, feat_tensor, target_tensor, spk_id_tensor, 
                 phone_id_tensor, clean_feats_tensor):
        assert feat_tensor.size(0) == target_tensor.size(0)
        assert feat_tensor.size(0) == spk_id_tensor.size(0)
        assert feat_tensor.size(0) == phone_id_tensor.size(0)
        assert feat_tensor.size(0) == clean_feats_tensor.size(0)
        self.feat_tensor = feat_tensor
        self.target_tensor = target_tensor
        self.spk_id_tensor = spk_id_tensor
        self.phone_id_tensor = phone_id_tensor
        self.clean_feats_tensor = clean_feats_tensor

    def __getitem__(self, index):
        return (self.feat_tensor[index], self.target_tensor[index], 
                self.spk_id_tensor[index], self.phone_id_tensor[index], 
                self.clean_feats_tensor[index])
    def __len__(self):
        return self.feat_tensor.size(0)
