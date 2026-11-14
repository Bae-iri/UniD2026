import os, random, copy, json, time, math
from pathlib import Path
from datetime import datetime

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent / "data" / "battery_data_light"

# 결과 저장 폴더
OUT = BASE_DIR / "battery_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")

SEED = 42
IMAGE_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 15
LR = 3e-4
NUM_WORKERS = 2
D = 64

# 우선 A와 B만 실행
MODES = ["flat", "conditional"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT.mkdir(parents=True, exist_ok=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Query를 만드는 현재 전기·열·환경 정보
ELECTRO = [
    "meas_v",
    "meas_t1",
    "meas_t2",
    "atmp_t",
    "thermal_t",
    "indu_t",
]

# Query가 참고할 가스·압력 정보
GAS = [
    "meas_p",
    "meas_ch4",
    "meas_co",
    "meas_co2",
    "meas_hcn",
    "meas_hcl",
    "meas_hf",
    "meas_n2o",
    "meas_no",
    "meas_no2",
    "meas_so2",
]

NUMERIC = ELECTRO + GAS

train_df = pd.read_csv(ROOT / "train/train.csv").reset_index(drop=True)
val_df = pd.read_csv(ROOT / "validation/validation.csv").reset_index(drop=True)

# 제공된 실험 단위 분리 유지
assert set(train_df.experiment_id).isdisjoint(set(val_df.experiment_id))

for df in [train_df, val_df]:
    assert df.sample_id.is_unique
    assert df.label.isin(range(1, 7)).all()
    assert df.bat_class.isin([1, 2, 3]).all()
    assert df.tr_method.isin([1, 2]).all()
    assert df.soc_status.isin([1, 2]).all()


def numeric_frame(df):
    return (
        df[NUMERIC]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )


# 결측 대체값·평균·표준편차는 train에서만 계산
train_num = numeric_frame(train_df)

median = train_num.median().fillna(0.0)
filled = train_num.fillna(median)

mean = filled.mean()
std = filled.std(ddof=0).clip(lower=1e-6)

cap_mean = float(train_df.bat_cc.mean())
cap_std = max(float(train_df.bat_cc.std(ddof=0)), 1e-6)


def make_arrays(df):
    num = numeric_frame(df)

    missing = num.isna().to_numpy(dtype=np.float32)
    values = (
        (num.fillna(median) - mean) / std
    ).to_numpy(dtype=np.float32)

    # 정적 조건:
    # 배터리 형태 3개 + 유도 방식 2개 + 용량 1개 + SOC 1개
    static = np.concatenate([
        np.eye(3, dtype=np.float32)[
            df.bat_class.to_numpy(dtype=int) - 1
        ],
        np.eye(2, dtype=np.float32)[
            df.tr_method.to_numpy(dtype=int) - 1
        ],
        (
            (df.bat_cc.to_numpy(dtype=np.float32) - cap_mean)
            / cap_std
        )[:, None],
        (
            df.soc_status.to_numpy(dtype=np.float32) * 0.5
        )[:, None],
    ], axis=1)

    # Query 입력:
    # 현재 전기·열 6개 + 결측 표시 6개 + 정적 조건 7개
    q = np.concatenate([
        values[:, :6],
        missing[:, :6],
        static,
    ], axis=1)

    # 가스·압력 변수마다 [값, 결측 표시]를 유지
    # shape: [샘플 수, 변수 수 11, 2]
    gas = np.stack([
        values[:, 6:],
        missing[:, 6:],
    ], axis=-1)

    # CrossEntropyLoss용 0~5 라벨
    y = df.label.to_numpy(dtype=np.int64) - 1

    return q.astype(np.float32), gas.astype(np.float32), y


train_arrays = make_arrays(train_df)
val_arrays = make_arrays(val_df)

Q_DIM = train_arrays[0].shape[1]

# 두 모델에 동일하게 적용할 클래스 가중치
counts = np.bincount(train_arrays[2], minlength=6)
assert (counts > 0).all()

weights = counts.astype(np.float32) ** -0.5
CLASS_WEIGHTS = torch.tensor(
    weights / weights.mean(),
    device=DEVICE,
)

preprocessing = {
    "numeric_columns": NUMERIC,
    "median": median.tolist(),
    "mean": mean.tolist(),
    "std": std.tolist(),
    "capacity_mean": cap_mean,
    "capacity_std": cap_std,
    "image_size": IMAGE_SIZE,
    "seed": SEED,
    "feature_dim": D,
}

(OUT / "preprocessing.json").write_text(
    json.dumps(preprocessing, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("device:", DEVICE)
print("train:", len(train_df), "| validation:", len(val_df))
print("Query 입력 차원:", Q_DIM)
print("결과 저장 위치:", OUT)

class BatteryDataset(Dataset):
    def __init__(self, df, split, arrays):
        self.paths = [
            ROOT / split / str(p).replace("\\", "/")
            for p in df.image_path
        ]

        self.q, self.gas, self.y = [
            torch.from_numpy(a)
            for a in arrays
        ]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as source:
            im = source.convert("RGB")

            # 종횡비 유지
            ratio = min(
                IMAGE_SIZE / im.width,
                IMAGE_SIZE / im.height,
            )

            w = max(1, min(IMAGE_SIZE, round(im.width * ratio)))
            h = max(1, min(IMAGE_SIZE, round(im.height * ratio)))

            im = im.resize((w, h), Image.Resampling.BILINEAR)
            pixels = np.asarray(im, dtype=np.float32) / 255.0

        image = np.zeros(
            (IMAGE_SIZE, IMAGE_SIZE, 3),
            dtype=np.float32,
        )
        mask = np.zeros(
            (1, IMAGE_SIZE, IMAGE_SIZE),
            dtype=np.float32,
        )

        x = (IMAGE_SIZE - w) // 2
        y = (IMAGE_SIZE - h) // 2

        image[y:y+h, x:x+w] = pixels
        mask[:, y:y+h, x:x+w] = 1.0

        image = torch.from_numpy(
            image.transpose(2, 0, 1).copy()
        )

        return (
            image,
            torch.from_numpy(mask),
            self.q[i],
            self.gas[i],
            self.y[i],
        )


def masked_mean(tokens, valid):
    weight = valid.to(tokens.dtype).unsqueeze(-1)

    return (
        (tokens * weight).sum(1)
        / weight.sum(1).clamp_min(1.0)
    )


class Readout(nn.Module):
    """
    flat:
        유효 token을 균등하게 요약

    conditional:
        현재 상황 Query에 따라 token 선택

    fixed:
        모든 샘플이 공유하는 학습 가능한 Query로 token 선택
    """

    def __init__(self, d):
        super().__init__()

        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.output = nn.Linear(d, d)

        self.fixed_query = nn.Parameter(
            torch.randn(1, d) * 0.02
        )

        self.ffn = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

    def forward(self, tokens, q, valid, mode):
        if mode == "flat":
            # A: 모든 유효 token을 같은 비중으로 사용
            attention = valid.to(tokens.dtype)

        else:
            if mode == "conditional":
                # B: 현재 상황에 따라 Query가 바뀜
                selection_q = q
            else:
                # C: 정보 선택용 Query만 상수로 교체
                selection_q = self.fixed_query.expand(
                    q.size(0), -1
                )

            score = (
                self.key(tokens)
                * self.query(selection_q).unsqueeze(1)
            ).sum(-1) / math.sqrt(tokens.size(-1))

            score = score.masked_fill(
                ~valid,
                torch.finfo(score.dtype).min,
            )

            attention = (
                score.softmax(dim=-1)
                * valid.to(score.dtype)
            )

        # 전부 결측인 경우에도 NaN 방지
        attention = (
            attention
            / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        )

        z = (
            attention.unsqueeze(-1)
            * self.value(tokens)
        ).sum(1)

        z = self.output(z)

        return z + self.ffn(z)


class BatteryModel(nn.Module):
    def __init__(self, mode):
        super().__init__()

        assert mode in {"flat", "conditional", "fixed"}
        self.mode = mode

        # 정적 조건 + 현재 전기·열 → 상황 Query
        self.q_encoder = nn.Sequential(
            nn.Linear(Q_DIM, D),
            nn.ReLU(),
            nn.Linear(D, D),
        )

        # 가스 변수의 종류를 구분하는 embedding
        self.gas_id = nn.Embedding(len(GAS), 8)
        self.register_buffer(
            "gas_indices",
            torch.arange(len(GAS)),
        )

        # 각 가스 token = [측정값, 결측 표시, 변수 embedding]
        self.gas_encoder = nn.Sequential(
            nn.Linear(2 + 8, D),
            nn.ReLU(),
            nn.Linear(D, D),
        )

        # 현재 이미지 → 공간별 특징
        layers = []

        for cin, cout in [
            (3, 16),
            (16, 32),
            (32, 64),
            (64, D),
        ]:
            layers += [
                nn.Conv2d(
                    cin, cout,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(8, cout),
                nn.ReLU(),
            ]

        self.cnn = nn.Sequential(*layers)

        # 이미지 token에 공간 위치 정보 추가
        self.position = nn.Linear(2, D, bias=False)

        self.gas_readout = Readout(D)
        self.image_readout = Readout(D)

        # q + 원래 가스 + 원래 이미지 + 선택된 가스 + 선택된 이미지
        self.head = nn.Sequential(
            nn.Linear(D * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 6),
        )

    def forward(self, image, image_mask, q_input, gas_input):
        q = self.q_encoder(q_input)

        # ---------- 가스 token ----------
        identity = self.gas_id(
            self.gas_indices
        ).unsqueeze(0)

        identity = identity.expand(
            gas_input.size(0), -1, -1
        )

        gas_tokens = self.gas_encoder(
            torch.cat([gas_input, identity], dim=-1)
        )

        gas_valid = gas_input[..., 1] < 0.5

        # ---------- 이미지 공간 token ----------
        feature = self.cnn(image)
        _, _, h, w = feature.shape

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=image.device),
            torch.linspace(-1, 1, w, device=image.device),
            indexing="ij",
        )

        xy = torch.stack(
            [xx, yy], dim=-1
        ).reshape(h * w, 2)

        image_tokens = (
            feature.flatten(2).transpose(1, 2)
            + self.position(xy).unsqueeze(0)
        )

        image_valid = (
            F.adaptive_max_pool2d(
                image_mask, (h, w)
            ).flatten(1) > 0
        )

        # ---------- 모든 모델의 공통 직접 연결 ----------
        gas_base = masked_mean(gas_tokens, gas_valid)
        image_base = masked_mean(image_tokens, image_valid)

        # ---------- 모드별 정보 선택 ----------
        gas_read = self.gas_readout(
            gas_tokens, q, gas_valid, self.mode
        )
        image_read = self.image_readout(
            image_tokens, q, image_valid, self.mode
        )

        # fixed에서도 실제 현재 상황 q는 최종 판단에 계속 사용
        combined = torch.cat([
            q,
            gas_base,
            image_base,
            gas_read,
            image_read,
        ], dim=-1)

        return self.head(combined)


train_ds = BatteryDataset(
    train_df, "train", train_arrays
)
val_ds = BatteryDataset(
    val_df, "validation", val_arrays
)

# 모든 모드가 같은 초기 가중치에서 시작
seed_everything(SEED)

INITIAL_STATE = copy.deepcopy(
    BatteryModel("flat").state_dict()
)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset, shuffle):
    generator = torch.Generator().manual_seed(SEED)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=generator,
    )


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    labels = []
    probabilities = []

    for batch in loader:
        image, mask, q, gas, y = [
            t.to(DEVICE, non_blocking=True)
            for t in batch
        ]

        logits = model(image, mask, q, gas)

        probabilities.append(
            logits.softmax(-1).cpu().numpy()
        )
        labels.append(y.cpu().numpy())

    y = np.concatenate(labels)
    prob = np.concatenate(probabilities)
    pred = prob.argmax(axis=1)

    score = f1_score(
        y,
        pred,
        labels=np.arange(6),
        average="macro",
        zero_division=0,
    )

    return float(score), y, pred, prob


def run_one(mode):
    seed_everything(SEED)

    model = BatteryModel(mode)
    model.load_state_dict(INITIAL_STATE)
    model = model.to(DEVICE)

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False)

    criterion = nn.CrossEntropyLoss(
        weight=CLASS_WEIGHTS
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    folder = OUT / mode
    folder.mkdir(parents=True, exist_ok=True)

    checkpoint = folder / "best.pt"
    best = -1.0
    best_epoch = 0
    history = []
    started = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()

        loss_sum = 0.0
        weight_sum = 0.0

        for batch in train_loader:
            image, mask, q, gas, y = [
                t.to(DEVICE, non_blocking=True)
                for t in batch
            ]

            optimizer.zero_grad(set_to_none=True)

            # softmax 이전 logits를 CrossEntropyLoss에 전달
            logits = model(image, mask, q, gas)
            loss = criterion(logits, y)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"{mode}: non-finite loss"
                )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(), 5.0
            )

            optimizer.step()

            batch_weight = CLASS_WEIGHTS[y].sum().item()
            loss_sum += loss.item() * batch_weight
            weight_sum += batch_weight

        score, _, _, _ = evaluate(model, val_loader)

        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / weight_sum,
            "val_macro_f1": score,
        })

        if score > best:
            best = score
            best_epoch = epoch

            torch.save(
                {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                },
                checkpoint,
            )

        print(
            f"{mode:11s} | "
            f"epoch {epoch:02d}/{EPOCHS} | "
            f"loss {loss_sum / weight_sum:.4f} | "
            f"val macro-F1 {score:.4f} | "
            f"best {best:.4f}",
            flush=True,
        )

    # 최고 성능 모델을 다시 불러와 최종 평가
    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=DEVICE,
            weights_only=True,
        )
    )

    score, y, pred, prob = evaluate(
        model, val_loader
    )

    report = pd.DataFrame(
        classification_report(
            y,
            pred,
            labels=np.arange(6),
            target_names=[
                str(i) for i in range(1, 7)
            ],
            output_dict=True,
            zero_division=0,
        )
    ).T

    cm = pd.DataFrame(
        confusion_matrix(
            y, pred, labels=np.arange(6)
        ),
        index=[
            f"true_{i}" for i in range(1, 7)
        ],
        columns=[
            f"pred_{i}" for i in range(1, 7)
        ],
    )

    predictions = val_df[
        ["sample_id", "experiment_id", "label"]
    ].copy()

    predictions["pred"] = pred + 1

    for c in range(6):
        predictions[f"p{c+1}"] = prob[:, c]

    pd.DataFrame(history).to_csv(
        folder / "history.csv",
        index=False,
    )

    predictions.to_csv(
        folder / "val_predictions.csv",
        index=False,
    )

    report.to_csv(
        folder / "class_report.csv"
    )

    cm.to_csv(
        folder / "confusion_matrix.csv"
    )

    per_class = f1_score(
        y,
        pred,
        labels=np.arange(6),
        average=None,
        zero_division=0,
    )

    result = {
        "mode": mode,
        "macro_f1": score,
        "best_epoch": best_epoch,
        "minutes": round(
            (time.time() - started) / 60, 2
        ),
        **{
            f"F1_{c+1}": float(per_class[c])
            for c in range(6)
        },
    }

    print(
        "\n",
        report.loc[
            [str(i) for i in range(1, 7)]
        ].round(4),
    )
    print("\n", cm, "\n")

    del model, optimizer, train_loader, val_loader

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return result

results = []

for mode in MODES:
    results.append(run_one(mode))

    pd.DataFrame(results).to_csv(
        OUT / "summary.csv",
        index=False,
    )

summary = pd.DataFrame(results).set_index("mode")

print(summary.round(4).to_string())
print("결과 저장 위치:", OUT)

if {"flat", "conditional"}.issubset(summary.index):
    delta = (
        summary.loc["conditional", "macro_f1"]
        - summary.loc["flat", "macro_f1"]
    )

    print(
        f"conditional - flat: "
        f"{delta:+.4f} ({delta * 100:+.2f}%p)"
    )
