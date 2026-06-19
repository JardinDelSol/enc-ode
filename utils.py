import os
import logging
import pickle
from random import sample
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def save_checkpoint(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, "checkpt-%04d.pth" % epoch)
    torch.save(state, filename)


def get_logger(
    logpath, filepath, package_files=[], displaying=True, saving=True, debug=False
):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="w")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_int_p(p_list, t_list, batch_num, condition_num, data, ID, epoch):
    t_list = torch.stack([t[batch_num].detach().cpu() for t in t_list[::4]])
    for i in range(condition_num):
        x = torch.cat([x[1:, batch_num, i].detach().cpu() for x in p_list[i:]])
        t = t_list[-len(x) :]
        plt.plot(t, x)
    plt.savefig(
        "plot/{}/{}/{}/epoch{}_int_p.png".format(data, *ID.split("-"), epoch), dpi=200
    )
    plt.close()


def plot_int_p10(p_list, t_list, batch_num, condition_num, data, ID, epoch):
    t_list = torch.stack([t[batch_num].detach().cpu() for t in t_list[::4]])
    for i in range(condition_num):
        x = torch.cat([x[1:, batch_num, i].detach().cpu() for x in p_list[i:]])
        t = t_list[-len(x) :]
        plt.plot(t, x)
    plt.savefig("plot/{}/{}/{}/epoch{}_int_p10.png".format(data, *ID.split("-"), epoch))
    plt.close()


def plot_l(
    l_list, t_list, batch_num, event_num, condition_num, nlinspace, data, ID, epoch
):
    t_list = torch.stack([t[batch_num].detach().cpu() for t in t_list[::4]])

    for i in range(condition_num, event_num):
        x = torch.cat(
            [x[1:, batch_num, i].detach().cpu() for x in l_list[i * nlinspace :]]
        )
        t = t_list[-len(x) :]
        plt.plot(t, x)
    plt.savefig(
        "plot/{}/{}/{}/epoch{}_l.png".format(data, *ID.split("-"), epoch), dpi=200
    )
    plt.close()


def plot_p(p_list, t_list, batch_num, condition_num, nlinspace, data, ID, epoch):
    t_list = torch.stack([t[batch_num].cpu().detach() for t in t_list])
    for i in range(condition_num):
        p = torch.stack(
            [x[batch_num, i].detach().cpu() for x in p_list[i * 4 * nlinspace :]]
        )
        t = t_list[-len(p) :]
        plt.plot(t, p)

    plt.savefig(
        "plot/{}/{}/{}/epoch{}_p.png".format(data, *ID.split("-"), epoch), dpi=200
    )
    plt.close()


def plot_p10(p_list, t_list, batch_num, condition_num, nlinspace, data, ID, epoch):
    t_list = torch.stack([t[batch_num].cpu().detach() for t in t_list])
    for i in range(condition_num):
        p = torch.stack(
            [x[batch_num, i].detach().cpu() for x in p_list[i * 4 * nlinspace :]]
        )
        t = t_list[-len(p) :]
        plt.plot(t, p)

    plt.savefig("plot/{}/{}/{}/epoch{}_p10.png".format(data, *ID.split("-"), epoch))
    plt.close()


def plot_t_pred(t_tgt, t_pred, batch_num, infty_num, data, ID, epoch):
    tgt = t_tgt[batch_num]
    pred = t_pred[batch_num]
    for i in range(0, len(tgt) - infty_num, 3):
        plt.axvline(x=tgt[i], color=plt.cm.RdYlBu(i), linewidth=0.6)
        plt.axvline(x=pred[i], color=plt.cm.RdYlBu(i), linewidth=0.6, linestyle="--")
    # plt.savefig("plot/{}/{}/{}/epoch{}_tpred.png".format(data, *ID.split("-"), epoch))
    # plt.close()


class CVaR(nn.Module):
    """Evaluate the sample C-Var of a RV.
    If alpha -> 1, CVAR -> Expectation.
    If alpha -> 0, CVAR -> ESS SUP.
    Parameters
    ----------
    alpha: float
        tail of the distribution (between 0 and 1.).
    learning_rate: float
        learning rate of the internal parameter.
    References
    ----------
    Eq. (27) from Rockafellar, R. T., & Uryasev, S. (2000).
    Optimization of conditional value-at-risk. Journal of risk, 2, 21-42.
    """

    def __init__(
        trainer,
        alpha,
        predict_interval,
        distance_func,
        device,
        running_var,
        learning_rate=1e-2,
    ):
        super(CVaR, trainer).__init__()
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be in [0, 1].")

        trainer._alpha = alpha
        trainer.interval = predict_interval
        trainer.device = device
        trainer.running_var = running_var

        if alpha == 0:
            trainer._var = nn.Parameter(-torch.tensor(1.0), requires_grad=False)
        else:
            trainer._var = nn.Parameter(
                torch.zeros(
                    size=((trainer.interval - 1), 1), device=trainer.device
                ).squeeze(-1),
                requires_grad=True,
            )

        if distance_func == "l2":
            trainer.distance_function = nn.MSELoss(reduction="none")
        elif distance_func == "l1":
            trainer.distance_function == nn.L1Loss(reduction="none")

        trainer._optimizer = torch.optim.Adam(
            [{"params": trainer._var, "lr": learning_rate}]
        )

    def forward(trainer, data, predictions, mask):
        """Execute forward operation of CVaR module.
        output = VaR + 1 / alpha * max(losses - VaR, 0)
        Parameters
        ----------
        losses: torch.tensor
        Returns
        -------
        output: torch.tensor
        """

        losses = trainer.distance_function(data, predictions).mean(-1)[
            :, 1:
        ]  # B x (T-1) # Exclude Initial

        if trainer.running_var:
            CVaR = trainer._var + 1 / (1 - trainer._alpha) * torch.relu(
                losses - trainer._var
            ).mean(
                0
            )  # (T-1)
            loss_CVaR = CVaR.mean()

            losses_sorted = losses.sort(dim=0, descending=True)[0]
            print("Runed VaR : ", trainer._var[:])
            print(
                "Empirical VaR : ",
                losses_sorted[int(data.size(0) * (1 - trainer._alpha)), :],
            )

        else:
            losses_sorted = losses.sort(dim=0, descending=True)[0]  # B x T
            CVaR = losses_sorted[: int(data.size(0) * (1 - trainer._alpha)), :].mean(
                0
            )  # T    (B' x (T-1)   -> (T-1))
            loss_CVaR = CVaR.mean()

        return loss_CVaR

    def zero_grad(trainer):
        """Reset CVaR internal optimizer."""
        if trainer.running_var:
            trainer._optimizer.zero_grad()
        else:
            None

    def step(trainer):
        """Execute a step of the internal CVaR optimizer."""
        if trainer.running_var:
            trainer._optimizer.step()
        else:
            None

    @property
    def var(trainer):
        """Get estimated Value-at-Risk."""
        return trainer._var
