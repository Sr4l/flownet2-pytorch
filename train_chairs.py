#!/usr/bin/env python3
"""Download ~N pairs from Kaggle FlyingChairs, then train FlowNet2.

Kaggle uses: 00001_img1.ppm, 00001_img2.ppm, 00001_flow.flo
FlyingChairs loader expects: *-img0.ppm, *-img1.ppm, *-flow.flo
-> symlinks fix the naming.

Requires kagglehub (installed in .venv) + Kaggle API token:
    python -m kagglehub login
    or set KAGGLE_USERNAME / KAGGLE_KEY env vars

Usage:
    python train_chairs.py --num_pairs 100     # download + train
    python train_chairs.py --only_download     # download 100 pairs only
    python train_chairs.py --train_only        # train on existing download
"""

import argparse
import os
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath("."))

import kagglesdk

import inspect
import torch
from tqdm import tqdm

from utils import tools
import models, losses, datasets as datasets_module

OWNER, SLUG = "craljimenez", "flyingchairs"


# ─── download ──────────────────────────────────────────────────────────────

def _api_client():
    return kagglesdk.KaggleClient()


def download_one(remote_path, local_path, client=None):
    """Download one file from Kaggle dataset.

    The kagglesdk download_dataset_raw returns a requests.Response
    directly (follows redirect), so we just write the content.
    """
    from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRawRequest
    if client is None:
        client = _api_client()
    r = ApiDownloadDatasetRawRequest()
    r.owner_slug = OWNER
    r.dataset_slug = SLUG
    r.file_name = remote_path

    resp = client.datasets.dataset_api_client.download_dataset_raw(r)
    # resp is requests.Response (not HttpRedirect)
    resp.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(1 << 20):
            f.write(chunk)


def download_subset(data_dir, num_pairs):
    """Download num_pairs image/flow triplets as raw files, then symlink."""
    raw_dir = os.path.join(data_dir, ".raw")
    link_dir = os.path.join(data_dir, "links")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(link_dir, exist_ok=True)

    have = len([f for f in os.listdir(link_dir) if f.endswith("-flow.flo")])
    left = num_pairs - have
    if left <= 0:
        print(f"[download] {have} pairs already present.")
        return link_dir

    print(f"[download] Fetching {left} pairs (~{left * 2.7:.0f} MB) ...")

    client = _api_client()

    for i in tqdm(range(have, num_pairs), desc="Pairs"):
        pid = f"{i + 1:05d}"          # Kaggle id (1-indexed: 00001, 00002, ...)
        lid = f"{i:07d}"              # symlink id (0-indexed, 7-digit)
        files = [
            (f"FlyingChairs_release/data/{pid}_img1.ppm", lid + "-img0.ppm"),
            (f"FlyingChairs_release/data/{pid}_img2.ppm", lid + "-img1.ppm"),
            (f"FlyingChairs_release/data/{pid}_flow.flo", lid + "-flow.flo"),
        ]
        ok = True
        for remote, link_name in files:
            raw = os.path.join(raw_dir, os.path.basename(remote))
            link = os.path.join(link_dir, link_name)
            if not os.path.exists(raw):
                try:
                    download_one(remote, raw, client)
                except Exception as e:
                    print(f"\n  FAIL {pid} {link_name}: {e}")
                    ok = False
                    break
                time.sleep(0.3)  # rate-limit
            if ok and not os.path.exists(link):
                rel_target = os.path.relpath(raw, link_dir)
                os.symlink(rel_target, link)
        if not ok:
            print("  (re-run to resume)")
            break

    total = len([f for f in os.listdir(link_dir) if f.endswith("-flow.flo")])
    print(f"[download] {total}/{num_pairs} pairs ready in {link_dir}")
    return link_dir


# ─── train ─────────────────────────────────────────────────────────────────

def train(args, data_dir):
    from os.path import exists, join
    from torch.utils.data import DataLoader
    import tensorboardX

    torch.set_num_threads(2)

    args.model_class = tools.module_to_dict(models)[args.model]
    args.loss_class = tools.module_to_dict(losses)[args.loss]
    args.optimizer_class = tools.module_to_dict(__import__("torch.optim", fromlist=["Adam"]))[args.optimizer]

    if args.fp16:
        raise ValueError("--fp16 training is not supported by this script; use main.py instead")

    assert exists(data_dir), f"Data dir {data_dir} missing"

    cuda = torch.cuda.is_available() and args.number_gpus > 0
    # mirror main.py: parallel DataLoader args only when CUDA is used
    gpuargs = {'num_workers': args.number_workers * args.number_gpus,
               'pin_memory': True,
               'drop_last': True} if cuda else {}

    train_ds = datasets_module.FlyingChairs(
        args, True, root=data_dir,
        **{k[len("training_dataset_"):]: v for k, v in vars(args).items()
           if k.startswith("training_dataset_")}
    )
    train_dl = DataLoader(train_ds, batch_size=args.effective_batch_size,
                          shuffle=True, **gpuargs)
    print(f"  Train samples: {len(train_ds)}")

    val_ds = datasets_module.FlyingChairs(
        args, False, root=data_dir,
        **{k[len("validation_dataset_"):]: v for k, v in vars(args).items()
           if k.startswith("validation_dataset_")}
    )
    val_dl = DataLoader(val_ds, batch_size=args.effective_batch_size, shuffle=False, **gpuargs)

    model = args.model_class(args)
    if cuda:
        model = model.cuda()
        # mirror main.py: DataParallel whenever CUDA GPUs are used (even a single one)
        model = torch.nn.DataParallel(model, device_ids=list(range(args.number_gpus)))
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    model_inner = model.module if hasattr(model, "module") else model

    start_epoch, best_EPE = 0, 1e8
    if args.resume and exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        start_epoch = ckpt["epoch"]
        best_EPE = ckpt["best_EPE"]
        model_inner.load_state_dict(ckpt["state_dict"])
        print(f"  Resumed epoch {start_epoch}, best EPE {best_EPE:.4f}")

    opt_kwargs = {k[len("optimizer_"):]: v for k, v in vars(args).items()
               if k.startswith("optimizer_") and k != "optimizer_class"}
    init_params = set(inspect.signature(args.optimizer_class.__init__).parameters)
    opt_kwargs = {k: v for k, v in opt_kwargs.items() if k in init_params}
    optimizer = args.optimizer_class(model.parameters(), **opt_kwargs)
    loss_kwargs = {k[len("loss_"):]: v for k, v in vars(args).items()
               if k.startswith("loss_") and k != "loss_class"}
    # Only pass kwargs that the loss init actually accepts
    init_params = set(inspect.signature(args.loss_class.__init__).parameters)
    loss_kwargs = {k: v for k, v in loss_kwargs.items() if k in init_params}
    loss_fn = args.loss_class(args, **loss_kwargs)

    os.makedirs(join(args.save, "train"), exist_ok=True)
    os.makedirs(join(args.save, "validation"), exist_ok=True)
    tlog = tensorboardX.SummaryWriter(join(args.save, "train"))
    vlog = tensorboardX.SummaryWriter(join(args.save, "validation"))

    iteration = 0

    def evaluate():
        was = model.training
        model.eval()
        s, nb = 0.0, 0
        with torch.no_grad():
            for b in tqdm(val_dl, desc="Eval"):
                d, t = b
                d0 = d[0]  # dataset already gives [B, 3, 2, H, W] as the model expects
                t0 = t[0]
                if cuda:
                    d0 = d0.cuda()
                    t0 = t0.cuda()
                lv = loss_fn(model(d0), t0)
                s += lv[0].item()  # primary loss only, same metric as main.py's validation_loss
                nb += 1
        model.train(was)
        return s / max(nb, 1)

    for epoch in range(start_epoch, args.total_epochs):
        model.train()
        ls_total, n, ls_buf = 0.0, 0, 0.0

        for b in tqdm(train_dl, desc=f"Ep {epoch}"):
            d, t = b
            # Dataset gives images as [B, 3, 2, H, W] (channels in dim 1) — what the model expects
            d0 = d[0]
            t0 = t[0]  # flow is [B, 2, H, W]
            if cuda:
                d0 = d0.cuda()
                t0 = t0.cuda()
            optimizer.zero_grad()
            model_out = model(d0)  # flow tensor(s), possibly multi-scale
            loss_values = loss_fn(model_out, t0)  # [loss, epe]
            loss_values = [torch.mean(lv) for lv in loss_values]
            loss_val = loss_values[0]  # first loss drives the weight update (as in main.py)
            loss_vals = [lv.item() for lv in loss_values]  # keep for logging
            loss_val.backward()
            if args.gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()

            # LR schedule, per iteration (as in main.py); self-guards on schedule_lr_frequency > 0
            tools.update_hyperparameter_schedule(args, epoch, iteration, optimizer)

            iteration += 1
            bl = sum(loss_vals)
            ls_total += bl
            ls_buf += bl
            n += 1

            if iteration % 10 == 0:
                for j, lv in enumerate(loss_vals):
                    tlog.add_scalar(f"loss_{j}", lv, iteration)
                tlog.add_scalar("sum", ls_buf / 10, iteration)
                tlog.add_scalar("lr", optimizer.param_groups[0]["lr"], iteration)
                ls_buf = 0.0

        print(f"  [{epoch}] loss={ls_total / max(n, 1):.4f}")

        if (epoch + 1) % args.validation_frequency == 0:
            vl = evaluate()
            vlog.add_scalar("val_loss", vl, epoch + 1)
            # compare BEFORE updating best (as in main.py), otherwise is_best is never True
            is_best = vl < best_EPE
            if is_best:
                best_EPE = vl
            tools.save_checkpoint(
                {"arch": args.model, "epoch": epoch + 1, "state_dict": model_inner.state_dict(),
                 "best_EPE": best_EPE},
                is_best, args.save, args.model)
            print(f"  val_loss={vl:.4f}  best={best_EPE:.4f}")


# ─── main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_pairs", type=int, default=200, help="Pairs to download (~2.7 MB each)")
    p.add_argument("--data_dir", type=str, default="./work/chairs_data", help="Data output dir")
    p.add_argument("--only_download", action="store_true")
    p.add_argument("--train_only", action="store_true")

    g = p.add_argument_group("training")
    g.add_argument("-b", "--batch_size", type=int, default=4)
    g.add_argument("--total_epochs", type=int, default=500)
    g.add_argument("--crop_size", type=int, nargs=2, default=[256, 256])
    g.add_argument("--number_workers", type=int, default=4)
    g.add_argument("--number_gpus", type=int, default=-1)
    g.add_argument("--rgb_max", type=float, default=255.0)
    g.add_argument("--save", type=str, default="./work")
    g.add_argument("--resume", type=str, default=None)
    g.add_argument("--validation_frequency", type=int, default=10)
    g.add_argument("--schedule_lr_frequency", type=int, default=0,
                   help="in number of iterations (0 for no schedule)")
    g.add_argument("--schedule_lr_fraction", type=float, default=10)
    g.add_argument("--fp16", action="store_true")
    g.add_argument("--gradient_clip", type=float, default=None)
    g.add_argument("--seed", type=int, default=1)

    mg = p.add_argument_group("model")
    mg.add_argument("--model", type=str, default="FlowNet2",
                    choices=["FlowNet2", "FlowNet2C", "FlowNet2S", "FlowNet2SD", "FlowNet2CS", "FlowNet2CSS"])
    mg.add_argument("--model_batchNorm", action="store_true")
    mg.add_argument("--model_div_flow", type=float, default=20.0)

    lg = p.add_argument_group("loss")
    lg.add_argument("--loss", type=str, default="L1Loss", choices=["L1Loss", "L2Loss", "MultiScale"])
    lg.add_argument("--loss_startScale", type=int, default=4)
    lg.add_argument("--loss_numScales", type=int, default=5)
    lg.add_argument("--loss_l_weight", type=float, default=0.32)
    lg.add_argument("--loss_norm", type=str, default="L1")

    og = p.add_argument_group("optimizer")
    og.add_argument("--optimizer", type=str, default="Adam",
                    choices=["Adam", "ASGD", "Adamax", "Adadelta", "Adagrad", "AdamW", "RAdam", "RMSprop", "SGD"])
    og.add_argument("--optimizer_lr", type=float, default=1e-4)
    og.add_argument("--optimizer_weight_decay", type=float, default=0.0)
    og.add_argument("--optimizer_betas", type=float, nargs=2, default=[0.9, 0.999])

    p.add_argument("--training_dataset_replicates", type=int, default=1)
    p.add_argument("--validation_dataset_replicates", type=int, default=1)

    args = p.parse_args()
    gc = torch.cuda.device_count()
    args.number_gpus = gc if args.number_gpus < 0 else args.number_gpus
    args.effective_batch_size = args.batch_size * max(args.number_gpus, 1)
    args.inference_size = [-1, -1]

    if not args.train_only:
        print("=== Download ===")
        data = download_subset(args.data_dir, args.num_pairs)
        if args.only_download:
            return
    else:
        data = os.path.join(args.data_dir, "links") if os.path.isdir(os.path.join(args.data_dir, "links")) else args.data_dir

    print("\n=== Train ===")
    train(args, data)


if __name__ == "__main__":
    main()