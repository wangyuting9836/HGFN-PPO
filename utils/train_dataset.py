from torch_geometric.data import Dataset, Batch


class TrainDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_fn(batch):
    batched_graph = Batch.from_data_list([data[0] for data in batch])
    batched_instance = [data[1] for data in batch]
    return batched_graph, batched_instance
