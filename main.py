import os
from tqdm import tqdm

import time
from datetime import datetime
import numpy as np
import argparse
from random import SystemRandom
import sklearn.model_selection as model_selection
from copy import deepcopy

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable

from model import ENC_ODE
from dataloader import RealData

import utils as utils
from utils import *

### Train options
GPU_NUM = 5
LOAD_DIR = None
N_EPOCHS = 1000
parser = argparse.ArgumentParser("temp")
parser.add_argument("-desc", type=str, default="multimodal", help="Description")
parser.add_argument("-gpu_num", type=int, default=GPU_NUM, help="Number of GPU to use.")
parser.add_argument(
    "--load",
    type=str,
    default=LOAD_DIR,
    help="ID of the experiment to load for evaluation. If None, run a new experiment.",
)
parser.add_argument("-n_epochs", type=int, default=N_EPOCHS, help="Number of epochs")
parser.add_argument("--random_seed", type=int, default=2021, help="Random_seed")
parser.add_argument("-lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("-w_decay", type=float, default=1e-5, help="weight decay")
parser.add_argument("-batch_size", type=int, default=1024, help="Batch size")

### Dataset
parser.add_argument("-dataset", type=str, default="adni", help="[adni, oasis]")
parser.add_argument(
    "-datatype", type=str, default="amyloid", help="[tau, amyloid, fdg]"
)

### Dimensionss
parser.add_argument(
    "-hidden_dim", type=int, default=512, help="Dimension of Piecewise layers"
)
parser.add_argument("-enc_dim", type=int, default=16, help="Dimension of experiment")
parser.add_argument("-model_width", type=int, default=1)

### ODEs
parser.add_argument("-tol", type=int, default=1e-6, help="rtal/atol")
parser.add_argument("-method", type=str, default="euler", help="IVP method")
parser.add_argument("-nlinspace", type=int, default=4, help="number of linspace")
parser.add_argument("-reverse", type=bool, default=False, help="reverse learning?")

### Etc.
parser.add_argument(
    "-self_mse", type=bool, default=True, help="mse on self reconstruction"
)

args = parser.parse_args()

load_PATH = "chpt/{}/experiment_{}.ckpt".format(args.dataset, args.load)
save_PATH = "chpt/{}".format(args.dataset)
utils.makedirs(save_PATH)


device = "cuda:{}".format(args.gpu_num)
torch.cuda.set_device(device)
torch.autograd.set_detect_anomaly(True)


def get_dataset(args):
    data_dir = os.path.join("../datasets", args.dataset)
    dataset = RealData(data_dir)

    num_data = len(dataset)
    train_idx, test_idx = model_selection.train_test_split(
        list(range(num_data)), test_size=0.2, random_state=42
    )
    train_dataset = deepcopy(dataset)
    train_dataset.set_idx(train_idx)
    if args.reverse:
        train_dataset.trainset()

    test_dataset = deepcopy(dataset)
    test_dataset.set_idx(test_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=False,
    )

    return (
        train_loader,
        test_loader,
    )


def run(model, time_seq, type_seq, event_seq, label_seq, time_mask, unimodal_mask, epoch):
    time_seq, type_seq, event_seq, label_seq, time_mask, unimodal_mask = (
        time_seq.cuda(),
        type_seq.cuda(),
        event_seq.cuda(),
        label_seq.cuda(),
        time_mask.cuda(),
        unimodal_mask.cuda(),
    )
    return model(
        time_seq, type_seq, event_seq, label_seq, time_mask, unimodal_mask, epoch
    )


def run_epoch(model, train_loader, optimizer, logger, epoch):
    model.train()

    losses_rmse = 0
    losses_mae = 0
    train_time = 0

    for i, (
        time_seq,
        type_seq,
        event_seq,
        label_seq,
        time_mask,
        unimodal_mask,
    ) in enumerate(train_loader):
        start = time.time()
        rmse, mae = run(
            model,
            time_seq,
            type_seq,
            event_seq,
            label_seq,
            time_mask,
            unimodal_mask,
            epoch,
        )

        optimizer.zero_grad()
        rmse.backward()
        optimizer.step()

        tt = time.time() - start
        # print("training :", tt)
        train_time += tt

        losses_rmse += rmse.item()
        losses_mae += mae.item()

    event_loss_rmse = losses_rmse / len(train_loader)
    event_loss_mae = losses_mae / len(train_loader)
    message = "Epoch : [%d / %d], RMSE Loss : %.4f MAE Loss : %.4f" % (
        epoch + 1,
        args.n_epochs,
        event_loss_rmse,
        event_loss_mae
    )
    logger.info(message)
    # print("Avg. time : ", train_time / len(train_loader))

    return event_loss_rmse, event_loss_mae


def test_epoch(model, test_loader, optimizer, logger, epoch):
    model.eval()
    with torch.no_grad():
        losses_rmse = 0
        losses_mae = 0


        train_time = 0

        for i, (
            time_seq,
            type_seq,
            event_seq,
            label_seq,
            time_mask,
            unimodal_mask,
        ) in enumerate(test_loader):
            start = time.time()
            rmse, mae = run(
                model,
                time_seq,
                type_seq,
                event_seq,
                label_seq,
                time_mask,
                unimodal_mask,
                epoch,
            )

            # msg = "Iter = [%d / %d]| RMSE %f MAE" % (i + 1, len(test_loader), loss_rmse.item(), loss_mae.item())
            # logger.info(msg)

            tt = time.time() - start
            # print("testing :", tt)
            train_time += tt

            losses_rmse += rmse.item()
            losses_mae += mae.item()
            # a_losses += a_loss.item()
            # f_losses += f_loss.item()
            # t_losses += t_loss.item()

        event_loss_rmse = losses_rmse / len(test_loader)
        event_loss_mae = losses_mae / len(test_loader)
        # amy_loss = a_losses / len(test_loader)
        # fdg_loss = f_losses / len(test_loader)
        # tau_loss = t_losses / len(test_loader)
        message = "Test Epoch : [%d / %d], %s RMSE : %.4f MAE : %.4f" % (
            epoch + 1,
            args.n_epochs,
            args.datatype,
            event_loss_rmse,
            event_loss_mae
        )
        logger.info(message)
        # print("Avg. time : ", train_time / len(train_loader))

    return event_loss_rmse, event_loss_mae


def main(args):
    torch.manual_seed(args.random_seed)

    experimentID = args.load

    if experimentID is None:
        now = datetime.now()
        experimentID = now.strftime("%b%d-%H%M%S")
        ckpt_path = os.path.join(save_PATH, "experiment_" + str(experimentID) + ".ckpt")
    else:
        ckpt_path = os.path.join(
            save_PATH, "experiment_" + str(experimentID) + "_cont.ckpt"
        )
        checkpoint = torch.load(load_PATH)
        args = checkpoint["args"]
        args.gpu_num = GPU_NUM
        args.load = LOAD_DIR
        args.n_epochs = N_EPOCHS

    (
        train_loader,
        # val_loader,
        test_loader,
    ) = get_dataset(args)

    start_epoch = 0

    model = ENC_ODE(
        args,
        experimentID,
        tol=args.tol,
        otreg_strength=0.1,
        method=args.method,
        nlinspace=args.nlinspace,
    ).cuda()
    model.double()

    params = []
    for p in model.parameters():
        params.append(p)

    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.w_decay)

    if args.load != None:
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        log_path = "logs/{}/{}/{}_cont.log".format(
            args.dataset, *experimentID.split("-")
        )
    else:
        utils.makedirs("logs/{}/{}/".format(args.dataset, experimentID.split("-")[0]))
        log_path = "logs/{}/{}/{}.log".format(args.dataset, *experimentID.split("-"))

    num_params = utils.count_parameters(model)

    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath("__file__"))
    logger.info(args)

    logger.info("Num of parameters:" + str(num_params))

    ### save dir ###
    logger.info("Experiments " + str(experimentID))
    logger.info(args.dataset)

    best_val_loss_rmse = 1e9
    best_val_loss_mae = 1e9
    early_stop = 0

    for epoch in tqdm(range(start_epoch, args.n_epochs)):
        logger.info("Experiments " + str(experimentID))
        logger.info(args.dataset)

        logger.info("--- Training ---")
        train_loss_rmse, train_loss_mae = run_epoch(model, train_loader, optimizer, logger, epoch)

        # logger.info("--- Validating ---")
        # test_epoch(model, train_loader, optimizer, logger, epoch)

        logger.info("--- Testing ---")
        val_loss_rmse, val_loss_mae = test_epoch(model, test_loader, optimizer, logger, epoch)

        if val_loss_rmse < best_val_loss_rmse:
            best_val_loss_rmse = val_loss_rmse
            best_val_loss_mae = val_loss_mae
            torch.save(
                {
                    "epoch": epoch + 1,
                    "args": args,
                    "best_val_loss_rmse": best_val_loss_rmse,
                    "best_val_loss_mae": best_val_loss_mae,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                ckpt_path,
            )
            logger.info("Checkpoint updated! \nVal Loss : {:.4f}".format(val_loss_rmse))
            early_stop = 0
        else:
            early_stop += 1

    print(f"best_val_loss_rmse:{best_val_loss_rmse} best_val_loss_mae:{best_val_loss_mae}")


if __name__ == "__main__":
    main(args)
