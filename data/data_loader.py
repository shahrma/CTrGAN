import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt,data_cfgs):
    dataset = None
    if opt.dataset_mode == 'unaligned_sequence':
        from data.unaligned_sequence_dataset import UnalignedSequenceDataset
        dataset = UnalignedSequenceDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt,data_cfgs)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt,data_cfgs):
        BaseDataLoader.initialize(self, opt,data_cfgs)
        self.dataset = CreateDataset(opt,data_cfgs)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data

def CreateDataLoader(opt,data_cfgs):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt,data_cfgs)
    return data_loader
