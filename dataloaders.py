import torch
from torch.utils.data import DataLoader


class MultiScaleRainDataLoader:
    def __init__(self, dataset, batch_size=32, num_workers=4, pin_memory=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn
        )

    @staticmethod
    def collate_fn(batch):
        """
        批次整理函数
        batch: list of dicts, 每个 dict 对应一个 sample
        return: dict of batched tensors
        """
        # ---- 降水路径 ----
        precip_input = torch.stack([sample['precip_input'] for sample in batch], dim=0)  # (B, P, T_pre_in, K)
        precip_time_feat = torch.stack([sample['precip_time_feat'] for sample in batch], dim=0)  # (B, P, T_pre_in, F_time)

        # ---- 外部路径 ----
        meteor_input = torch.stack([sample['meteor_input'] for sample in batch], dim=0)  # (B, P, T_met_in, F)
        meteor_time_feat = torch.stack([sample['meteor_time_feat'] for sample in batch], dim=0)  # (B, P, T_met_in, F_time)

        # ---- 目标 ----
        target = torch.stack([sample['target'] for sample in batch], dim=0)  # (B, P, T_out)

        return {
            'precip_input': precip_input,
            'precip_time_feat': precip_time_feat,
            'meteor_input': meteor_input,
            'meteor_time_feat': meteor_time_feat,
            'target': target
        }

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

