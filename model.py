from tqdm import tqdm
import torch
import math
import torch.nn as nn
import torch.autograd
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from ODEs import *
from matplotlib import pyplot as plt
from utils import *
import time
import seaborn as sns

plt.switch_backend("agg")


def compute_temporal_embedding(self, time):
    batch_size = time.size(0)
    seq_len = time.size(1)
    pe = torch.zeros(batch_size, seq_len, self.d_time).to(time.device)
    _time = time.unsqueeze(-1)
    div_term = self.div_term.to(time.device)
    pe[..., 0::2] = torch.sin(_time * div_term)
    pe[..., 1::2] = torch.cos(_time * div_term)
    return pe


class ENC_ODE(nn.Module):  # ENC-ODE model
    def __init__(
        self,
        args,
        experimentID,
        tol=1e-8,
        otreg_strength=0.1,
        method="dopri5",
        nlinspace=1,
    ):
        super(ENC_ODE, self).__init__()
        self.feat = args.datatype

        self.batch_size = args.batch_size
        self.hdim = args.hidden_dim
        self.enc_dim = args.enc_dim

        self.method = method
        self.nlinspace = nlinspace

        self.encoder = nn.Sequential(
            nn.Linear(160 + self.enc_dim + self.enc_dim, self.hdim),
            # nn.ReLU(),
            # nn.Linear(self.hdim, self.hdim),
        )  # feature + k, g

        self.decoder = nn.Sequential(
            # nn.Linear(self.hdim, self.hdim),
            # nn.ReLU(),
            nn.Linear(self.hdim, 160),
        )  # Reconstruct

        self.label_emb = nn.Embedding(5, self.enc_dim, padding_idx=0)
        self.type_emb = nn.Embedding(4, self.enc_dim, padding_idx=0)  # 3 + pad
        self.time_emb = nn.Linear(1, 128)

        width = args.model_width
        hidden_state_list = [
            ConcatLinear(self.hdim + self.enc_dim, self.hdim),
            ActNorm(self.hdim),
            ConcatLinear(self.hdim + self.enc_dim, self.hdim),
            ActNorm(self.hdim),
        ]

        self.hidden_state_dynamics = SequentialDiffEq(*hidden_state_list)

        # intensity_odefunc = IntensityODEFunc(self.hdim, self.hidden_state_dynamics)
        intensity_odefunc = IntensityODEFunc(self.hdim, self.hidden_state_dynamics)

        self.ode_solver = TimeVariableODE(
            intensity_odefunc, atol=tol, rtol=tol, energy_regularization=otreg_strength
        )

        # Attn network
        self.Q = nn.Linear(self.hdim + 128, self.hdim)
        self.K = nn.Linear(128 + 16, self.hdim)

        self.reconstruct = args.self_mse

        self.mse = torch.nn.MSELoss(reduction="none")

    def inference(self, result, time_seq, type_seq, feat_seq, time_mask, unimodal_mask):
        B, T, _, _ = result.shape

        T_ = self.time_emb(time_seq.unsqueeze(-1))
        F_ = self.type_emb(type_seq)
        H_ = torch.cat([T_, F_], 2)
        key = self.K(H_)  # B, T-1, H

        R_h = torch.cat(
            [result, T_.unsqueeze(2).expand(-1, -1, T, -1)], 3
        )  # B, T, T, H+144
        query = self.Q(R_h)

        scores = (
            query
            * key.unsqueeze(2)
            / torch.sqrt(torch.tensor(self.hdim, dtype=torch.float32))
        )
        scores = scores.sum(-1)

        attn_mask = torch.tril(torch.ones([T, T]), -1).unsqueeze(0).cuda()
        scores = scores.masked_fill(attn_mask == 0, -1e9)

        attention_weights = scores.softmax(2)

        weighted_result = (
            attention_weights.unsqueeze(-1)
            * self.decoder(result)
            * (~torch.eye(T).unsqueeze(0).unsqueeze(-1).bool().cuda())
        )

        aggr_result = weighted_result.sum(2)[:, 1:]

        # final_result = self.decoder(aggr_result)
        final_result = aggr_result

        #     t1  t2  t3  t4
        # t1   0   0   0   0
        # t2   1   0   0   0
        # t3 0.4 0.6   0   0
        # t4 0.3 0.2 0.5   0

        loss_rmse = (
            self.mse(feat_seq[:, 1:], final_result)
            * time_mask[:, 1:].unsqueeze(-1)
            * unimodal_mask[:, 1:].unsqueeze(-1)
        )
        loss_mae = (
            torch.abs(feat_seq[:, 1:] - final_result)
            * time_mask[:, 1:].unsqueeze(-1)
            * unimodal_mask[:, 1:].unsqueeze(-1)
        )

        total_mask = time_mask[:, 1:] * unimodal_mask[:, 1:]

        event_num = torch.sum(total_mask) * 160

        amy_mask = type_seq[:, 1:] == 1
        fdg_mask = type_seq[:, 1:] == 2
        tau_mask = type_seq[:, 1:] == 3

        total_loss_rmse = torch.sqrt(loss_rmse.sum() / event_num)
        total_loss_mae = loss_mae.sum() / event_num

        amy_loss_rmse = torch.sqrt(
            (loss_rmse * amy_mask.unsqueeze(-1)).sum()
            / ((total_mask * amy_mask).sum() * 160)
        )
        fdg_loss_rmse = torch.sqrt(
            (loss_rmse * fdg_mask.unsqueeze(-1)).sum()
            / ((total_mask * fdg_mask).sum() * 160)
        )
        tau_loss_rmse = torch.sqrt(
            (loss_rmse * tau_mask.unsqueeze(-1)).sum()
            / ((total_mask * tau_mask).sum() * 160)
        )

        amy_loss_mae = (loss_mae * amy_mask.unsqueeze(-1)).sum() / (
            (total_mask * amy_mask).sum() * 160
        )
        fdg_loss_mae = (loss_mae * fdg_mask.unsqueeze(-1)).sum() / (
            (total_mask * fdg_mask).sum() * 160
        )
        tau_loss_mae = (loss_mae * tau_mask.unsqueeze(-1)).sum() / (
            (total_mask * tau_mask).sum() * 160
        )

        return (
            total_loss_rmse,
            amy_loss_rmse,
            fdg_loss_rmse,
            tau_loss_rmse,
            total_loss_mae,
            amy_loss_mae,
            fdg_loss_mae,
            tau_loss_mae,
        )

    def propagate(self, time_seq, type_seq, feat_seq, label_seq, nlinspace=1):
        B, T, F = feat_seq.shape
        result = torch.zeros((B, T, T, self.hdim)).to(time_seq)
        s_e = feat_seq[:, 0]  # B x 160
        s_t = type_seq[:, 0]
        s_l = label_seq[:, 0]

        # age_emb = self.age_emb(s_a)
        event_enc = self.type_emb(s_t)
        label_enc = self.label_emb(s_l)

        obsrv = torch.cat([s_e, label_enc, event_enc], 1)
        h_hat = self.encoder(obsrv).unsqueeze(1)

        result[:, 0, :1, :] = h_hat
        # idx1: which timepoint
        label_stack = label_enc.unsqueeze(1)
        t0 = time_seq[:, 0]

        for i in range(0, len(time_seq[0]) - 1):
            state = (h_hat, label_stack)
            # state = h_hat

            t1 = time_seq[:, i + 1]

            state_traj = self.ode_solver.integrate(
                t0,
                t1,
                state,
                nlinspace=nlinspace,
                method=self.method,
            )
            state = tuple(s[-1] for s in state_traj)
            h_hat, event_enc = state

            e_t1 = feat_seq[:, i + 1]
            t_t1 = type_seq[:, i + 1]
            l_t1 = label_seq[:, i + 1]

            # age_enc = self.age_emb(a_t1)
            event_enc = self.type_emb(t_t1)
            label_enc = self.label_emb(l_t1)

            obsrv = torch.cat([e_t1, label_enc, event_enc], 1)
            h_hat_t1 = self.encoder(obsrv).unsqueeze(1)
            h_hat = torch.cat([h_hat, h_hat_t1], 1)
            label_stack = torch.cat([label_stack, label_enc.unsqueeze(1)], 1)

            result[:, i + 1, : h_hat.size(1)] = h_hat

            t0 = t1

        return result

    def forward(
        self, time_seq, type_seq, feat_seq, label_seq, time_mask, unimodal_mask, epoch
    ):
        type_dict = {"amyloid": 1, "fdg": 2, "tau": 3}

        self.epoch = epoch

        result = self.propagate(
            time_seq,
            type_seq,
            feat_seq,
            label_seq,
            nlinspace=self.nlinspace,
        )

        if self.training:
            # loss = self.loss_fn(
            #     result, time_seq, type_seq, feat_seq, time_mask, unimodal_mask
            # )
            (
                loss_rmse,
                a_loss_rmse,
                f_loss_rmse,
                t_loss_rmse,
                loss_mae,
                a_loss_mae,
                f_loss_mae,
                t_loss_mae,
            ) = self.inference(
                result, time_seq, type_seq, feat_seq, time_mask, unimodal_mask
            )

            if self.feat == "amyloid":
                return a_loss_rmse, a_loss_mae
            elif self.feat == "fdg":
                return f_loss_rmse, f_loss_mae
            else:
                return t_loss_rmse, t_loss_mae

        else:
            (
                loss_rmse,
                a_loss_rmse,
                f_loss_rmse,
                t_loss_rmse,
                loss_mae,
                a_loss_mae,
                f_loss_mae,
                t_loss_mae,
            ) = self.inference(
                result, time_seq, type_seq, feat_seq, time_mask, unimodal_mask
            )

            if self.feat == "amyloid":
                return a_loss_rmse, a_loss_mae
            elif self.feat == "fdg":
                return f_loss_rmse, f_loss_mae
            else:
                return t_loss_rmse, t_loss_mae
