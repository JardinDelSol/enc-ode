import os
import torch
import torch.utils.data as data
import numpy as np
import pickle

# from mujoco_physics import HopperPhysics


class RealData(data.Dataset):
    def __init__(self, dir):
        super(RealData, self).__init__()
        pid_lst = os.listdir(dir)
        self.data = []
        for pid in pid_lst:
            file_dir = os.path.join(dir, pid)
            p_data = np.load(file_dir)
            feature, m_type, time, label = (
                p_data["x"],
                p_data["y"],
                p_data["t"],
                p_data["l"],
            )
            # if len(time) == 17:
            self.data.append([time, m_type, feature, label])

        self.pad = 0

    def set_idx(self, idx_lst):
        new_data = []
        for idx in idx_lst:
            new_data.append(self.data[idx])
        self.data = new_data

    def trainset(self):
        rev_data = []
        for data in self.data:
            time, m_type, feat, label = data
            rev_data.append([time[::-1], m_type[::-1], feat[::-1], label[::-1]])
        self.data += rev_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        return self.data[index]

    def padding(self, seqs, dtype):
        max_len = max(len(seq) for seq in seqs)
        pad = np.ones_like(seqs[0][0, ...]) * self.pad
        batch_seq = np.array([list(seq) + [pad] * (max_len - len(seq)) for seq in seqs])

        return torch.tensor(batch_seq, dtype=dtype)

    def get_unimodal_mask(self, time_seq, type_seq):
        mask = torch.zeros_like(time_seq)  # B, t
        first_flag = torch.zeros((time_seq.size(0), 3)).bool()
        for t in range(time_seq.size(1)):
            ft = torch.clamp_min(type_seq[:, t] - 1, 0).unsqueeze(-1)  # B
            flag = first_flag.gather(1, ft)
            mask[:, t] = torch.where(
                flag, torch.ones_like(flag), torch.zeros_like(flag)
            ).squeeze(-1)
            first_flag.scatter_(1, ft, True)

        return mask

    def collate_fn(self, batch):
        time_seq, type_seq, feat_seq, label_seq = list(zip(*batch))

        time_seq = self.padding(time_seq, torch.float64)  # max_len x t x 1
        type_seq = self.padding(type_seq, torch.int64)
        feat_seq = self.padding(feat_seq, torch.float64).squeeze(2)  # max_len x t x 160 x 3
        label_seq = self.padding(label_seq, torch.int64)

        time_pad_mask = time_seq.eq(self.pad)
        unimodal_mask = self.get_unimodal_mask(time_seq, type_seq)

        return (
            time_seq,  # B x t x1
            type_seq,
            feat_seq,  # B x 160 x 3
            label_seq,
            ~time_pad_mask,  # B x t x 1
            unimodal_mask,
        )
