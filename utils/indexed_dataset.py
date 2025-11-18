from torch_geometric.data import Dataset


class IndexedDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 返回数据和对应的索引
        return self.data_list[idx], idx
