"""Controlled tests of sensor -> intermediate response -> label dependence.

Importing this module never starts training. See HIERARCHY_EXPERIMENT.md.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
from PIL import Image
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.nn import functional as F


BASE = Path(__file__).resolve().parent
ELECTRO = ["meas_v", "meas_t1", "meas_t2", "atmp_t", "thermal_t", "indu_t"]
GAS = ["meas_p", "meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl",
       "meas_hf", "meas_n2o", "meas_no", "meas_no2", "meas_so2"]
NUMERIC = ELECTRO + GAS
MODES = ["flat", "conditional", "hierarchy_noaux", "hierarchy_supervised", "hierarchy_shuffled"]
HIERARCHY = set(MODES[2:])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def split_frames(root, split_seed, dev_fraction, smoke=False):
    train = pd.read_csv(root / "train/train.csv").reset_index(drop=True)
    validation = pd.read_csv(root / "validation/validation.csv").reset_index(drop=True)
    for name, frame in (("train", train), ("validation", validation)):
        if not frame.sample_id.is_unique:
            raise ValueError(f"Duplicate sample_id in {name}")
        for column, allowed in (("label", range(1, 7)), ("bat_class", [1, 2, 3]),
                                ("tr_method", [1, 2]), ("soc_status", [1, 2])):
            if not frame[column].isin(allowed).all():
                raise ValueError(f"Invalid {column} in {name}")
    if set(train.experiment_id) & set(validation.experiment_id):
        raise ValueError("Train/validation experiment leakage")
    if set(train.sample_id) & set(validation.sample_id):
        raise ValueError("Train/validation sample leakage")
    splitter = GroupShuffleSplit(n_splits=1, test_size=dev_fraction, random_state=split_seed)
    fit_ids, dev_ids = next(splitter.split(train, groups=train.experiment_id))
    frames = {"fit": train.iloc[fit_ids].copy(), "dev": train.iloc[dev_ids].copy(),
              "validation": validation.copy()}
    for name, frame in frames.items():
        if smoke:
            frame = frame.sample(min(len(frame), 384 if name == "fit" else 192), random_state=split_seed)
        frames[name] = frame.reset_index(drop=True)
        if set(frames[name].label) != set(range(1, 7)):
            raise ValueError(f"{name} is missing a class; choose a suitable group split explicitly")
    groups = [set(frame.experiment_id) for frame in frames.values()]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Experiment overlap between fit/dev/validation")
    return frames


class Preprocessor:
    """Only fit rows determine imputation, normalization, and class weights."""

    def __init__(self, fit):
        numeric = self.numeric(fit)
        self.median = numeric.median().fillna(0.0)
        filled = numeric.fillna(self.median)
        self.mean = filled.mean()
        self.std = filled.std(ddof=0).clip(lower=1e-6)
        self.cap_mean = float(fit.bat_cc.mean())
        self.cap_std = max(float(fit.bat_cc.std(ddof=0)), 1e-6)
        counts = np.bincount(fit.label.to_numpy(int) - 1, minlength=6)
        if (counts == 0).any():
            raise ValueError("Every class must occur in fit")
        weights = counts.astype(np.float32) ** -0.5
        self.class_weights = weights / weights.mean()
        observed_mean = numeric[GAS].mean().fillna(self.median[GAS])
        self.gas_mean = ((observed_mean - self.mean[GAS]) / self.std[GAS]).to_numpy(np.float32)

    @staticmethod
    def numeric(frame):
        return frame[NUMERIC].astype(float).replace([np.inf, -np.inf], np.nan)

    def transform(self, frame):
        num = self.numeric(frame)
        missing = num.isna().to_numpy(np.float32)
        values = ((num.fillna(self.median) - self.mean) / self.std).to_numpy(np.float32)
        static = np.concatenate([
            np.eye(3, dtype=np.float32)[frame.bat_class.to_numpy(int) - 1],
            np.eye(2, dtype=np.float32)[frame.tr_method.to_numpy(int) - 1],
            ((frame.bat_cc.to_numpy(np.float32) - self.cap_mean) / self.cap_std)[:, None],
            (frame.soc_status.to_numpy(np.float32) * 0.5)[:, None],
        ], axis=1)
        q = np.concatenate([values[:, :6], missing[:, :6], static], axis=1)
        gas = np.stack([values[:, 6:], missing[:, 6:]], axis=-1)
        return q.astype(np.float32), gas.astype(np.float32), frame.label.to_numpy(np.int64) - 1

    def metadata(self):
        return {"numeric_columns": NUMERIC, "median": self.median.tolist(),
                "mean": self.mean.tolist(), "std": self.std.tolist(),
                "capacity_mean": self.cap_mean, "capacity_std": self.cap_std,
                "class_weights": self.class_weights.tolist(), "gas_mean_normalized": self.gas_mean.tolist()}


def image_descriptor(rgb, mask, grid):
    """Fixed, padding-masked RGB means in a grid; no trainable target encoder."""
    size = rgb.shape[-1]
    cell = size // grid
    weights = mask.astype(np.float32)
    counts = weights.reshape(grid, cell, grid, cell).sum(axis=(1, 3))
    summed = (rgb.astype(np.float32) / 255.0 * weights[None]).reshape(3, grid, cell, grid, cell).sum(axis=(2, 4))
    means = summed / np.maximum(counts[None], 1.0)
    valid = np.broadcast_to(counts[None] > 0, means.shape)
    return means.reshape(-1), valid.reshape(-1).copy()


def read_image(path, size, grid):
    with Image.open(path) as source:
        im = source.convert("RGB")
        ratio = min(size / im.width, size / im.height)
        w, h = max(1, min(size, round(im.width * ratio))), max(1, min(size, round(im.height * ratio)))
        pixels = np.asarray(im.resize((w, h), Image.Resampling.BILINEAR), dtype=np.uint8)
    packed = np.zeros((4, size, size), dtype=np.uint8)
    x, y = (size - w) // 2, (size - h) // 2
    packed[:3, y:y+h, x:x+w] = pixels.transpose(2, 0, 1)
    packed[3, y:y+h, x:x+w] = 1
    descriptor, valid = image_descriptor(packed[:3], packed[3], grid)
    return packed, descriptor, valid


def cache_images(root, frame, source_split, cache_root, size, grid, workers):
    paths = [root / source_split / str(p).replace("\\", "/") for p in frame.image_path]
    fingerprint = hashlib.sha256(f"image-cache-v1:{size}:{grid}".encode())
    for sample, path in zip(frame.sample_id, paths):
        stat = path.stat()
        fingerprint.update(f"{sample}:{path}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    folder = cache_root / fingerprint.hexdigest()[:20]
    image_path, target_path = folder / "images.npy", folder / "targets.npz"
    if not (image_path.exists() and target_path.exists()):
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / "images.partial.npy"
        images = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8,
                                          shape=(len(frame), 4, size, size))
        targets = np.empty((len(frame), 3 * grid * grid), dtype=np.float32)
        valid = np.empty_like(targets, dtype=bool)
        print(f"Caching {source_split}: {len(frame)} images -> {folder}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, (image, target, mask) in enumerate(pool.map(lambda p: read_image(p, size, grid), paths)):
                images[i], targets[i], valid[i] = image, target, mask
                if (i + 1) % 4000 == 0:
                    print(f"  cached {i + 1}/{len(frame)}", flush=True)
        images.flush()
        del images
        temporary.replace(image_path)
        np.savez(target_path, targets=targets, valid=valid)
    loaded = np.load(target_path)
    return np.load(image_path, mmap_mode="c"), loaded["targets"], loaded["valid"]


def fit_image_normalizer(targets, valid):
    counts = valid.sum(axis=0).clip(min=1)
    mean = (targets * valid).sum(axis=0) / counts
    var = (((targets - mean) ** 2) * valid).sum(axis=0) / counts
    return mean.astype(np.float32), np.maximum(np.sqrt(var), 1e-3).astype(np.float32)


@dataclass
class PreparedSplit:
    frame: pd.DataFrame
    images: torch.Tensor
    q: torch.Tensor
    gas: torch.Tensor
    y: torch.Tensor
    image_target: torch.Tensor
    image_valid: torch.Tensor

    def __len__(self):
        return len(self.frame)

    def batch(self, ids, device):
        ids = ids.to(self.q.device)
        images = self.images.index_select(0, ids)
        return (images[:, :3].to(device, dtype=torch.float32).div_(255.0),
                images[:, 3:4].to(device), self.q.index_select(0, ids).to(device),
                self.gas.index_select(0, ids).to(device), self.y.index_select(0, ids).to(device),
                self.image_target.index_select(0, ids).to(device), self.image_valid.index_select(0, ids).to(device))


def prepare_data(args, frames, preprocessor, device, out):
    cached = {}
    for name, frame in frames.items():
        cached[name] = cache_images(args.data_root, frame, "train" if name != "validation" else "validation",
                                    args.cache_dir, args.image_size, args.image_grid, args.cache_workers)
    cache_manifest = {name: str(images.filename) for name, (images, _, _) in cached.items()}
    cache_manifest_path = out / "image_cache_manifest.json"
    if cache_manifest_path.exists() and json.loads(cache_manifest_path.read_text()) != cache_manifest:
        raise ValueError("Image source files changed since this run; start a new output directory")
    write_json(cache_manifest_path, cache_manifest)
    image_mean, image_std = fit_image_normalizer(cached["fit"][1], cached["fit"][2])
    metadata = preprocessor.metadata()
    metadata.update(image_grid=args.image_grid, image_size=args.image_size,
                    image_mean=image_mean.tolist(), image_std=image_std.tolist(),
                    image_target="padding-masked RGB means, channel-major grid order")
    write_json(out / "preprocessing.json", metadata)
    cache_device = torch.device("cpu")
    required = sum(array.nbytes for array, _, _ in cached.values())
    if args.cache_device == "cuda" or (args.cache_device == "auto" and device.type == "cuda"
                                        and torch.cuda.mem_get_info(device)[0] > required + 3 * 1024 ** 3):
        if device.type != "cuda":
            raise ValueError("CUDA cache requested without CUDA compute")
        cache_device = device
    print(f"Image cache: {required / 1024**3:.2f} GiB, resident on {cache_device}", flush=True)
    prepared = {}
    for name, frame in frames.items():
        images, target, valid = cached.pop(name)
        q, gas, y = preprocessor.transform(frame)
        standardized = np.where(valid, (target - image_mean) / image_std, 0).astype(np.float32)
        tensors = [torch.from_numpy(array).to(cache_device) for array in (images, q, gas, y, standardized, valid)]
        prepared[name] = PreparedSplit(frame, *tensors)
    return prepared


def masked_mean(tokens, valid):
    weights = valid.to(tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(1) / weights.sum(1).clamp_min(1.0)


class Readout(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.query, self.key, self.value, self.output = [nn.Linear(d, d) for _ in range(4)]
        self.fixed_query = nn.Parameter(torch.randn(1, d) * 0.02)
        self.ffn = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, tokens, q, valid, mode):
        if mode == "flat":
            attention = valid.to(tokens.dtype)
        else:
            scores = (self.key(tokens) * self.query(q).unsqueeze(1)).sum(-1) / math.sqrt(tokens.size(-1))
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            attention = scores.softmax(-1) * valid.to(scores.dtype)
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        z = self.output((attention.unsqueeze(-1) * self.value(tokens)).sum(1))
        return z + self.ffn(z)


class BatteryHierarchyModel(nn.Module):
    def __init__(self, mode, q_dim=19, d=64, image_target_dim=48):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.q_encoder = nn.Sequential(nn.Linear(q_dim, d), nn.ReLU(), nn.Linear(d, d))
        self.gas_id = nn.Embedding(len(GAS), 8)
        self.register_buffer("gas_indices", torch.arange(len(GAS)))
        self.gas_encoder = nn.Sequential(nn.Linear(10, d), nn.ReLU(), nn.Linear(d, d))
        layers = []
        for cin, cout in [(3, 16), (16, 32), (32, 64), (64, d)]:
            layers.extend([nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                           nn.GroupNorm(8, cout), nn.ReLU()])
        self.cnn = nn.Sequential(*layers)
        self.position = nn.Linear(2, d, bias=False)
        self.gas_readout, self.image_readout = Readout(d), Readout(d)
        self.head = nn.Sequential(nn.Linear(d * 5, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 6))
        # Added after the original modules so h3's common initialization is preserved.
        self.gas_predictor = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, len(GAS)))
        self.image_predictor = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, image_target_dim))
        self.gas_prior_encoder = nn.Sequential(nn.Linear(len(GAS), d), nn.ReLU(), nn.Linear(d, d))
        self.image_prior_encoder = nn.Sequential(nn.Linear(image_target_dim, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, image, image_mask, q_input, gas_input):
        q = self.q_encoder(q_input)
        gas_prediction, image_prediction = self.gas_predictor(q), self.image_predictor(q)
        identity = self.gas_id(self.gas_indices).unsqueeze(0).expand(len(q), -1, -1)
        gas_tokens = self.gas_encoder(torch.cat([gas_input, identity], dim=-1))
        gas_valid = gas_input[..., 1] < 0.5
        feature = self.cnn(image)
        h, w = feature.shape[-2:]
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=image.device),
                                torch.linspace(-1, 1, w, device=image.device), indexing="ij")
        xy = torch.stack([xx, yy], dim=-1).reshape(h * w, 2)
        image_tokens = feature.flatten(2).transpose(1, 2) + self.position(xy).unsqueeze(0)
        image_valid = F.adaptive_max_pool2d(image_mask.float(), (h, w)).flatten(1) > 0
        if self.mode in HIERARCHY:
            gas_q = self.gas_prior_encoder(gas_prediction)
            image_q = self.image_prior_encoder(image_prediction)
            # All sensor-to-label paths go through the explicitly predicted responses.
            context, readout_mode = (gas_q + image_q) * 0.5, "conditional"
        else:
            context, gas_q, image_q, readout_mode = q, q, q, self.mode
        combined = torch.cat([context, masked_mean(gas_tokens, gas_valid), masked_mean(image_tokens, image_valid),
                              self.gas_readout(gas_tokens, gas_q, gas_valid, readout_mode),
                              self.image_readout(image_tokens, image_q, image_valid, readout_mode)], dim=-1)
        return self.head(combined), gas_prediction, image_prediction


def masked_squared_error(prediction, target, valid):
    # torch.where also prevents an excluded NaN target from poisoning the loss.
    residual = torch.where(valid, prediction - target, torch.zeros_like(prediction))
    return residual.square().sum(), valid.sum()


def auxiliary_loss(gas_prediction, image_prediction, gas, image_target, image_valid):
    gas_sse, gas_count = masked_squared_error(gas_prediction, gas[..., 0], gas[..., 1] < 0.5)
    image_sse, image_count = masked_squared_error(image_prediction, image_target, image_valid)
    return gas_sse / gas_count.clamp_min(1), image_sse / image_count.clamp_min(1)


@torch.no_grad()
def evaluate(model, data, args, device, gas_mean, collect=False):
    model.eval()
    probabilities, gas_outputs, image_outputs = [], [], []
    # SSE and train-mean baseline SSE, with explicit missing-value masks.
    totals = torch.zeros(6, device=device, dtype=torch.float64)
    feature_sse = torch.zeros(len(GAS) + 3 * args.image_grid ** 2, device=device, dtype=torch.float64)
    feature_baseline, feature_count = feature_sse.clone(), feature_sse.clone()
    gas_mean = torch.as_tensor(gas_mean, device=device)
    for start in range(0, len(data), args.eval_batch_size):
        ids = torch.arange(start, min(start + args.eval_batch_size, len(data)))
        image, mask, q, gas, _, image_target, image_valid = data.batch(ids, device)
        logits, gas_prediction, image_prediction = model(image, mask, q, gas)
        probabilities.append(logits.softmax(-1).cpu().numpy())
        if model.mode in HIERARCHY:
            gvalid = gas[..., 1] < 0.5
            gres = torch.where(gvalid, gas_prediction - gas[..., 0], 0).square()
            gbase = torch.where(gvalid, gas[..., 0] - gas_mean, 0).square()
            ires = torch.where(image_valid, image_prediction - image_target, 0).square()
            ibase = torch.where(image_valid, image_target, 0).square()
            totals += torch.stack([gres.sum(), gvalid.sum(), gbase.sum(), ires.sum(), image_valid.sum(), ibase.sum()])
            feature_sse += torch.cat([gres.sum(0), ires.sum(0)])
            feature_baseline += torch.cat([gbase.sum(0), ibase.sum(0)])
            feature_count += torch.cat([gvalid.sum(0), image_valid.sum(0)])
            if collect:
                gas_outputs.append(gas_prediction.cpu().numpy())
                image_outputs.append(image_prediction.cpu().numpy())
    prob = np.concatenate(probabilities)
    y = data.frame.label.to_numpy(int) - 1
    pred = prob.argmax(1)
    result = {"macro_f1": float(f1_score(y, pred, labels=np.arange(6), average="macro", zero_division=0)),
              "accuracy": float((y == pred).mean()),
              **{f"F1_{i+1}": float(v) for i, v in enumerate(f1_score(y, pred, labels=np.arange(6), average=None, zero_division=0))}}
    if model.mode in HIERARCHY:
        gs, gn, gb, ims, imn, imb = totals.cpu().tolist()
        result.update(gas_mse=gs / max(gn, 1), gas_mean_baseline_mse=gb / max(gn, 1),
                      gas_skill_vs_mean=1 - gs / gb if gb > 0 else None,
                      image_mse=ims / max(imn, 1), image_mean_baseline_mse=imb / max(imn, 1),
                      image_skill_vs_mean=1 - ims / imb if imb > 0 else None)
    details = {"y": y, "pred": pred, "prob": prob}
    if collect and model.mode in HIERARCHY:
        details.update(gas_prediction=np.concatenate(gas_outputs), image_prediction=np.concatenate(image_outputs),
                       feature_sse=feature_sse.cpu().numpy(), feature_baseline=feature_baseline.cpu().numpy(),
                       feature_count=feature_count.cpu().numpy())
    return result, details


def save_evaluation(folder, data, metrics, details, args):
    y, pred = details["y"], details["pred"]
    predictions = data.frame[["sample_id", "experiment_id", "label"]].copy()
    predictions["pred"] = pred + 1
    for i in range(6):
        predictions[f"p{i+1}"] = details["prob"][:, i]
    predictions.to_csv(folder / "val_predictions.csv", index=False)
    pd.DataFrame(classification_report(y, pred, labels=np.arange(6), target_names=[str(i) for i in range(1, 7)],
                                       output_dict=True, zero_division=0)).T.to_csv(folder / "class_report.csv")
    pd.DataFrame(confusion_matrix(y, pred, labels=np.arange(6)),
                 index=[f"true_{i}" for i in range(1, 7)],
                 columns=[f"pred_{i}" for i in range(1, 7)]).to_csv(folder / "confusion_matrix.csv")
    if "gas_prediction" in details:
        np.savez_compressed(folder / "intermediate_predictions.npz", gas_prediction=details["gas_prediction"],
                            image_prediction=details["image_prediction"], gas_target=data.gas[:, :, 0].cpu().numpy(),
                            gas_valid=(data.gas[:, :, 1] < 0.5).cpu().numpy(),
                            image_target=data.image_target.cpu().numpy(), image_valid=data.image_valid.cpu().numpy())
        names = GAS + [f"image_{channel}_{row}_{col}" for channel in "RGB"
                       for row in range(args.image_grid) for col in range(args.image_grid)]
        count = details["feature_count"].clip(min=1)
        feature = pd.DataFrame({"feature": names, "observations": details["feature_count"],
                                "mse": details["feature_sse"] / count,
                                "train_mean_baseline_mse": details["feature_baseline"] / count})
        feature["skill_vs_mean"] = 1 - feature.mse / feature.train_mean_baseline_mse.replace(0, np.nan)
        feature.to_csv(folder / "intermediate_features.csv", index=False)


def run_one(mode, seed, initial_state, datasets, preprocessor, args, device, out):
    folder = out / f"seed_{seed}" / mode
    folder.mkdir(parents=True, exist_ok=True)
    complete = folder / "metrics.json"
    if args.resume and complete.exists():
        print(f"Resuming: already completed seed={seed} {mode}", flush=True)
        return json.loads(complete.read_text())
    seed_everything(seed)
    model = BatteryHierarchyModel(mode, d=args.feature_dim, image_target_dim=3 * args.image_grid ** 2)
    model.load_state_dict(initial_state)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(preprocessor.class_weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    fit = datasets["fit"]
    shuffle_generator = torch.Generator().manual_seed(seed)
    # Independent RNG: this control does not change minibatches, initialization, or dropout.
    permutation = torch.from_numpy(np.random.default_rng(seed + 17001).permutation(len(fit))).to(fit.q.device)
    best, best_epoch, history, active_parameters = -1.0, 0, [], None
    start_time = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(fit), generator=shuffle_generator)
        totals = torch.zeros(4, device=device, dtype=torch.float64)
        for start in range(0, len(fit), args.batch_size):
            ids = order[start:start + args.batch_size]
            image, mask, q, gas, y, image_target, image_valid = fit.batch(ids, device)
            optimizer.zero_grad(set_to_none=True)
            logits, gas_prediction, image_prediction = model(image, mask, q, gas)
            ce = criterion(logits, y)
            if mode == "hierarchy_shuffled":
                target_ids = permutation.index_select(0, ids.to(permutation.device))
                gas_target = fit.gas.index_select(0, target_ids).to(device)
                image_target = fit.image_target.index_select(0, target_ids).to(device)
                image_valid = fit.image_valid.index_select(0, target_ids).to(device)
            else:
                gas_target = gas
            gas_loss, image_loss = auxiliary_loss(gas_prediction, image_prediction, gas_target, image_target, image_valid)
            loss = ce
            if mode in {"hierarchy_supervised", "hierarchy_shuffled"}:
                loss = loss + args.gas_aux_weight * gas_loss + args.image_aux_weight * image_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss: {seed=} {mode=} {epoch=}")
            loss.backward()
            if active_parameters is None:
                active_parameters = sum(p.numel() for p in model.parameters() if p.grad is not None)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals += torch.stack([ce.detach(), gas_loss.detach(), image_loss.detach(), loss.detach()]) * len(y)
        averages = (totals / len(fit)).cpu().tolist()
        dev, _ = evaluate(model, datasets["dev"], args, device, preprocessor.gas_mean)
        if dev["macro_f1"] > best:
            best, best_epoch = dev["macro_f1"], epoch
            torch.save({"model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "mode": mode, "seed": seed, "best_dev_epoch": epoch, "best_dev_f1": best,
                        "feature_dim": args.feature_dim, "image_grid": args.image_grid}, folder / "best.pt")
        history.append({"epoch": epoch, "train_ce": averages[0], "train_gas_mse": averages[1],
                        "train_image_mse": averages[2], "train_objective": averages[3], "dev_macro_f1": dev["macro_f1"]})
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        print(f"seed {seed} | {mode:21s} | {epoch:02d}/{args.epochs} | CE {averages[0]:.4f} | "
              f"dev F1 {dev['macro_f1']:.4f} | best {best:.4f}", flush=True)
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    metrics, details = evaluate(model, datasets["validation"], args, device, preprocessor.gas_mean, collect=True)
    metrics.update(mode=mode, seed=seed, best_dev_epoch=best_epoch, best_dev_f1=best,
                   minutes=round((time.monotonic() - start_time) / 60, 3),
                   parameters_total=sum(p.numel() for p in model.parameters()), parameters_active=active_parameters)
    save_evaluation(folder, datasets["validation"], metrics, details, args)
    write_json(complete, metrics)
    print(f"FINAL seed {seed} | {mode}: held-out validation macro-F1 {metrics['macro_f1']:.4f}", flush=True)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def macro_from_confusions(cms):
    diagonal = np.diagonal(cms, axis1=-2, axis2=-1)
    denom = cms.sum(-1) + cms.sum(-2)
    return np.divide(2 * diagonal, denom, out=np.zeros_like(diagonal, dtype=float), where=denom > 0).mean(-1)


def summarize(out, results, args, frames):
    summary = pd.DataFrame(results)
    summary.to_csv(out / "summary.csv", index=False)
    columns = ["macro_f1", "accuracy"] + [f"F1_{i}" for i in range(1, 7)] + ["gas_skill_vs_mean", "image_skill_vs_mean"]
    columns = [column for column in columns if column in summary]
    grouped = summary.groupby("mode", sort=False)
    aggregate = grouped[columns].mean().add_suffix("_mean")
    aggregate["seeds"] = grouped.size()
    aggregate["macro_f1_std"] = grouped.macro_f1.std(ddof=1).fillna(0)
    aggregate.to_csv(out / "aggregate.csv")
    frame = frames["validation"]
    experiments = sorted(frame.experiment_id.unique())
    confusion = {}
    for mode in args.modes:
        per_seed = []
        for seed in args.seeds:
            prediction = pd.read_csv(out / f"seed_{seed}" / mode / "val_predictions.csv")
            if prediction.sample_id.tolist() != frame.sample_id.tolist():
                raise ValueError("Prediction row mismatch")
            per_seed.append(np.stack([confusion_matrix(group.label - 1, group.pred - 1, labels=np.arange(6))
                                      for _, group in prediction.groupby("experiment_id", sort=True)]))
        confusion[mode] = np.stack(per_seed)
    rng = np.random.default_rng(args.split_seed + 971)
    weights = rng.multinomial(len(experiments), np.full(len(experiments), 1 / len(experiments)), size=args.bootstrap)
    boot = {mode: macro_from_confusions(np.einsum("be,semn->bsmn", weights, cms)).mean(1)
            for mode, cms in confusion.items()}
    comparisons = []
    pairs = [("conditional", "flat")] + [("hierarchy_supervised", mode)
                                            for mode in ["hierarchy_noaux", "hierarchy_shuffled", "flat", "conditional"]]
    for candidate, reference in pairs:
        if candidate not in confusion or reference not in confusion:
            continue
        delta = float(aggregate.loc[candidate, "macro_f1_mean"] - aggregate.loc[reference, "macro_f1_mean"])
        lower, upper = np.quantile(boot[candidate] - boot[reference], [0.025, 0.975])
        comparisons.append({"candidate": candidate, "reference": reference, "mean_delta": delta,
                            "experiment_bootstrap_95_low": float(lower), "experiment_bootstrap_95_high": float(upper)})
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(out / "comparisons.csv", index=False)
    lines = ["# 센서 → 중간 반응 → label 비교 실험", "",
             f"Smoke test: {args.smoke}. Seeds: {args.seeds}. Epochs: {args.epochs}.", "",
             "## 평가 방법", "",
             "- 제공 train을 experiment_id 단위로 fit/dev로 분리. 전처리와 가중치는 fit에서만 계산.",
             "- checkpoint는 dev macro-F1로만 선택. 제공 validation은 최종 평가에만 사용.",
             "- 모든 모델에 같은 초기 공통 가중치, 입력, 배치 순서, optimizer와 학습 횟수 적용.",
             "- 세 hierarchy 모델은 구조·파라미터 수가 같고 중간 목표의 감독 방식만 다름.",
             "- 이미지 목표는 고정된 4×4 RGB 영역 평균(기본값). 전체 이미지 생성이나 의미 특징 예측은 아님.", "",
             "## 검증 macro-F1", "", "| 모델 | 평균 | seed 표준편차 |", "|---|---:|---:|"]
    for mode, row in aggregate.iterrows():
        lines.append(f"| {mode} | {row.macro_f1_mean:.4f} | {row.macro_f1_std:.4f} |")
    lines += ["", "## 쌍별 비교", "", "차이는 candidate − reference. 95% 구간은 검증 실험 단위 paired bootstrap이며, seed별 점수 차이를 평균함.",
              "", "| candidate | reference | 평균 차이 | 95% 구간 |", "|---|---|---:|---:|"]
    for row in comparisons:
        lines.append(f"| {row['candidate']} | {row['reference']} | {row['mean_delta']:+.4f} | "
                     f"[{row['experiment_bootstrap_95_low']:+.4f}, {row['experiment_bootstrap_95_high']:+.4f}] |")
    lines += ["", "## 중간 반응 예측", "", "skill = 1 − 예측 SSE / fit 평균값 예측 SSE. 양수면 train 평균 기준보다 잘 예측함.",
              "", "| 모델 | 가스 skill | 이미지 skill |", "|---|---:|---:|"]
    for mode in args.modes:
        if mode in HIERARCHY:
            row = aggregate.loc[mode]
            lines.append(f"| {mode} | {row.gas_skill_vs_mean_mean:.4f} | {row.image_skill_vs_mean_mean:.4f} |")
    primary = [r for r in comparisons if r["candidate"] == "hierarchy_supervised"
               and r["reference"] in {"hierarchy_noaux", "hierarchy_shuffled"}]
    if "hierarchy_supervised" in aggregate.index:
        row = aggregate.loc["hierarchy_supervised"]
        predictive = row.gas_skill_vs_mean_mean > 0 and row.image_skill_vs_mean_mean > 0
        supported = len(primary) == 2 and all(r["experiment_bootstrap_95_low"] > 0 for r in primary) and predictive
        conclusion = ("이 설정에서는 중간 반응 예측 가능성과 올바른 중간 감독의 label 성능 개선을 함께 지지함."
                      if supported else "이 설정에서는 파생 구조 가설을 충분히 뒷받침하지 못함. 중간 예측 성능과 label 개선을 나누어 해석해야 함.")
        lines += ["", "## 해석", "", conclusion]
    lines += ["", "## 범위와 한계", "",
              "- 동시점 관측의 예측적 의존관계를 검사함. 물리적 인과 방향이나 완전한 매개 관계를 입증하지 않음.",
              "- hierarchy도 실제 가스·이미지를 입력으로 사용하며, 센서만으로 두 모달리티를 대체하는 실험은 아님.",
              "- bootstrap은 고정된 학습 결과에 대한 검증 실험 표본 불확실성임. 새로운 학습 데이터로 재학습한 불확실성까지 포함하지 않음.",
              "- 제공 test에는 label이 없으므로 test 점수는 계산하지 않음.",
              "- 기존 h3 결과는 validation으로 checkpoint를 골랐으므로 이 실험 수치와 직접 비교하지 않음.",
              "- 이미지 요약 목표, 보조 손실 가중치, 15 epoch 설정에 대한 한 번의 사전 지정 비교임.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + aggregate[["macro_f1_mean", "macro_f1_std", "seeds"]].round(4).to_string(), flush=True)
    print(f"Report: {out / 'report.md'}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=BASE.parent / "data/battery_data_light")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=BASE / ".hierarchy_cache")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--cache-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gas-aux-weight", type=float, default=0.1)
    parser.add_argument("--image-aux-weight", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--image-grid", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--split-seed", type=int, default=20261114)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--cache-workers", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="1 epoch, 1 seed, small samples; not evidence")
    args = parser.parse_args()
    if args.smoke:
        args.epochs, args.seeds, args.bootstrap = 1, [args.seeds[0]], 100
    if args.image_size % args.image_grid or args.feature_dim % 8:
        parser.error("image-size must divide by image-grid; feature-dim must divide by 8")
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.bootstrap, args.cache_workers, args.cpu_threads) <= 0:
        parser.error("Counts must be positive")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.modes)) != len(args.modes):
        parser.error("Duplicate seeds or modes")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Run on the GPU host, or explicitly use --device cpu.")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    out = args.output or BASE / "hierarchy_runs" / datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    configuration = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
                     if k not in {"output", "resume"}}
    configuration.update(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         actual_device=str(device), torch_version=str(torch.__version__),
                         gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
    configuration["csv_sha256"] = {
        split: hashlib.sha256((args.data_root / split / f"{split}.csv").read_bytes()).hexdigest()
        for split in ("train", "validation")}
    config_path = out / "config.json"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != configuration:
            raise ValueError("Output already exists or resume configuration/code differs; use a new output directory")
    else:
        write_json(config_path, configuration)
        (out / "source.py").write_bytes(Path(__file__).read_bytes())
    frames = split_frames(args.data_root, args.split_seed, args.dev_fraction, args.smoke)
    print(f"Device: {device}; output: {out}", flush=True)
    for name, frame in frames.items():
        print(f"{name}: {len(frame)} rows, {frame.experiment_id.nunique()} experiments, labels {frame.label.value_counts().sort_index().to_dict()}", flush=True)
    manifest = pd.concat([frame[["sample_id", "experiment_id", "label"]].assign(split=name)
                          for name, frame in frames.items()], ignore_index=True)
    manifest.to_csv(out / "split_manifest.csv", index=False)
    preprocessor = Preprocessor(frames["fit"])
    datasets = prepare_data(args, frames, preprocessor, device, out)
    results = []
    for seed in args.seeds:
        seed_everything(seed)
        initial = copy.deepcopy(BatteryHierarchyModel("flat", d=args.feature_dim,
                                                     image_target_dim=3 * args.image_grid ** 2).state_dict())
        for mode in args.modes:
            results.append(run_one(mode, seed, initial, datasets, preprocessor, args, device, out))
            pd.DataFrame(results).to_csv(out / "summary.csv", index=False)
    summarize(out, results, args, frames)


if __name__ == "__main__":
    main()
