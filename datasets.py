from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiScaleRainDataset(Dataset):
    def __init__(self, precip_series, meteor_series, decision,
                 precip_timestamps, meteor_timestamps, num_spots=6,
                 T_pre_in=180, T_met_in=6, T_out=1, stride=1, pred_offset=60):
        """
        input:
            precip_series: vmd分解后的分钟级降水序列 (P, T_min, K)
            meteor_series: 完整小时级外部特征序列 (P, T_h, F)
            precip_timestamps: pd.DatetimeIndex, length N_pre
            meteor_timestamps: 完整pd.DatetimeIndex, length N_met
            decision: 分钟级决策序列 (P, T_min)

            T_pre_in: 降水序列输入时间窗口
            T_met_in: 外部特征序列输入时间窗口
            T_out: 决策输出时间窗口
            stride: 降水序列输入时间窗口移动步长(min)
        
        """
        super(MultiScaleRainDataset, self).__init__()
        self.precip_series = precip_series
        self.meteor_series = meteor_series
        self.decision = decision

        self.num_spots = num_spots

        self.precip_ts = precip_timestamps
        self.meteor_ts = meteor_timestamps

        self.T_pre_in = T_pre_in
        self.T_met_in = T_met_in
        self.T_out = T_out
        self.stride = stride
        self.pred_offset = pred_offset

        self.N_pre = self.precip_series.shape[1]
        self.N_met = self.meteor_series.shape[1]

        # 所有合法的降雨预测起点 t
        self.valid_t = [
            t for t in range(
                T_pre_in,
                self.N_pre - self.pred_offset - self.T_out,
                self.stride
            )
            if (t // 60) >= T_met_in
        ]

        self.valid_pairs = [
            (t, p)
            for t in self.valid_t
            for p in range(self.num_spots)
        ]

    def __len__(self):
        return len(self.valid_pairs)

    def encode_time_features(self, ts):
        """
        ts: pandas.Timestamp
        return: dict of float features
        """

        # 一天中的分钟
        minute_of_day = ts.dt.hour * 60 + ts.dt.minute
        minute_sin = np.sin(2 * np.pi * minute_of_day / 1440)
        minute_cos = np.cos(2 * np.pi * minute_of_day / 1440)

        # 一年中的第几天
        day_of_year = ts.dt.dayofyear
        days_in_year = np.where(ts.dt.is_leap_year, 366, 365)
        doy_sin = np.sin(2 * np.pi * day_of_year / days_in_year)
        doy_cos = np.cos(2 * np.pi * day_of_year / days_in_year)

        # 周期信息
        weekday = ts.dt.weekday  # 0-6
        weekday_sin = np.sin(2 * np.pi * weekday / 7)
        weekday_cos = np.cos(2 * np.pi * weekday / 7)

        features = np.column_stack([
            minute_sin, minute_cos,
            doy_sin, doy_cos,
            weekday_sin, weekday_cos
        ])

        return features.astype(np.float32)

    def __getitem__(self, idx):
        # 当前降雨 预测起点（min）
        t, p = self.valid_pairs[idx]

        # ====== 分钟级输入（主序列）======
        pre_slice = slice(t - self.T_pre_in, t)

        pre_window = self.precip_series[p, pre_slice, :]  # (T_pre_in, K)

        pre_ts = self.precip_ts.iloc[pre_slice]
        pre_time_feat = self.encode_time_features(pre_ts)

        # ====== 小时级 context（外场）======
        # 当前分钟 t 对应的“小时索引”
        cur_ts = self.precip_ts.iloc[t]
        cur_hour_ts = cur_ts.floor('h')
        hour_idx = self.meteor_ts[self.meteor_ts == cur_hour_ts].index.item() + 1

        met_slice = slice(hour_idx - self.T_met_in, hour_idx)

        met_window = self.meteor_series[p, met_slice, :]  # (T_met_in, F)
        met_ts = self.meteor_ts.iloc[met_slice]

        met_time_feat = self.encode_time_features(met_ts)

        # ====== 预测目标（分钟级 decision）======
        target_t = t + self.pred_offset
        target = self.decision[p, target_t: target_t + self.T_out]

        return {
            # ---- 降水路径 ----
            "precip_input": torch.tensor(pre_window, dtype=torch.float32),  # (T_pre_in, K)
            "precip_time_feat": torch.tensor(pre_time_feat, dtype=torch.float32),  # (T_pre_in, 6)

            # ---- 外部特征路径 ----
            "meteor_input": torch.tensor(met_window, dtype=torch.float32),  # (T_met_in, F)
            "meteor_time_feat": torch.tensor(met_time_feat, dtype=torch.float32),  # (T_met_in, 6)

            # ---- 输出 ----
            "target": torch.tensor(target, dtype=torch.long),  # (T_out,)
        }

    def apply_ratio_resampling(
            self,
            target_ratios,
            seed=42
    ):
        """
        点位级重采样：
        - 样本单位：(t, p)
        - label = decision[p, t + pred_offset]
        - 四类比例由 target_ratios 精确控制
        """
        rng = np.random.RandomState(seed)

        # -----------------------------
        # 1. 按点位级 label 分组
        # -----------------------------
        cls_samples = defaultdict(list)

        for (t, p) in self.valid_pairs:
            label = int(self.decision[p, t + self.pred_offset])
            cls_samples[label].append((t, p))

        total_samples = len(self.valid_pairs)

        # 校验比例
        assert abs(sum(target_ratios.values()) - 1.0) < 1e-6, \
            "target_ratios 必须加和为 1"

        # -----------------------------
        # 2. 按目标比例重采样
        # -----------------------------
        final_pairs = []

        for c, ratio in target_ratios.items():
            target_num = int(total_samples * ratio)
            samples = cls_samples.get(c, [])

            if len(samples) == 0:
                continue

            if len(samples) >= target_num:
                # 欠采样
                chosen = rng.choice(
                    len(samples),
                    size=target_num,
                    replace=False
                )
                final_pairs.extend([samples[i] for i in chosen])
            else:
                # 过采样
                chosen = rng.choice(
                    len(samples),
                    size=target_num,
                    replace=True
                )
                final_pairs.extend([samples[i] for i in chosen])

        # -----------------------------
        # 3. 更新 valid_pairs
        # -----------------------------
        self.valid_pairs = final_pairs

    def get_target_distribution(self):
        """
        点位级 label 分布统计
        """
        decision = self.decision
        if isinstance(decision, torch.Tensor):
            decision = decision.detach().cpu().numpy()

        labels = [
            int(decision[p, t + self.pred_offset])
            for (t, p) in self.valid_pairs
        ]

        return Counter(labels)
