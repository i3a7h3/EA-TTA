#!/usr/bin/env python3
"""
cifar10c_tta_hpsearch.py
========================

CIFAR-10-C: No-TTA vs Tent vs CS-TTA (with hyperparameter search, CXR-style)

1) Train ResNet-18 on CIFAR-10 (clean)
2) Evaluate on CIFAR-10 test (clean)
3) Evaluate on CIFAR-10-C:
    - No-TTA
    - Tent (BN-only TTA) with HP search
    - CS-TTA (CLIP-aware, stable/spurious weighting) with HP search

Tent 선택 규칙:
    - 여러 HP trial 수행 후
    - No-TTA보다 좋은 trial들 중에서
      "가장 낮은 accuracy" (worst among better-than-No-TTA)를 선택
    - 그런 trial이 하나도 없으면, 단순히 best accuracy로 fallback

CS-TTA 선택 규칙:
    - 각 CLIP(B/32, L/14)에 대해
    - 여러 HP trial 중 best accuracy 선택

결과는 JSON 파일로 저장:
    results_cifar10c_tta_hpsearch.json
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 물리 GPU 1번만 사용

import json
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as T
from datasets import load_dataset
from tqdm import tqdm
import clip

# =======================
# CONFIG
# =======================

DATA_ROOT = "./data"
OUTPUT_JSON = "./results_cifar10c_tta_hpsearch.json"
BASE_CKPT  = "./cifar10_resnet18_base.pth"

BATCH_SIZE = 128
NUM_EPOCHS = 10
LR = 0.1
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10C_HF_NAME = "robro/cifar10-c-parquet"

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CLIP_MODELS = [
    "ViT-B/32",
    "ViT-L/14",
]

# Tent 하이퍼파라미터 후보들
TENT_HP_LIST = [
    {"name": "tent_a", "lr": 1e-4, "lambda_ent": 1.0, "lambda_anchor": 1.0, "anchor_conf_thresh": 0.9},
    {"name": "tent_b", "lr": 5e-5, "lambda_ent": 0.5, "lambda_anchor": 1.0, "anchor_conf_thresh": 0.9},
    {"name": "tent_c", "lr": 1e-5, "lambda_ent": 0.3, "lambda_anchor": 1.0, "anchor_conf_thresh": 0.95},
]

# CS-TTA 하이퍼파라미터 후보들 (각 clip에 대해 동일 리스트 사용)
CSTTA_HP_LIST = [
    {"name": "cs_a", "lr": 5e-5, "lambda_ent": 0.5, "lambda_causal": 1.0, "lambda_anchor": 1.0, "ema_alpha": 0.99, "gamma": 1.0},
    {"name": "cs_b", "lr": 1e-4, "lambda_ent": 0.3, "lambda_causal": 1.5, "lambda_anchor": 1.0, "ema_alpha": 0.95, "gamma": 2.0},
    {"name": "cs_c", "lr": 5e-5, "lambda_ent": 0.2, "lambda_causal": 2.0, "lambda_anchor": 1.5, "ema_alpha": 0.99, "gamma": 3.0},
]

torch.backends.cudnn.benchmark = True


# =======================
# DATASETS
# =======================

def get_cifar10_loaders(data_root=DATA_ROOT, batch_size=BATCH_SIZE):
    data_root = Path(data_root)

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ])

    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform_train,
    )

    test_set = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform_test,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, test_loader, transform_test


class CIFAR10CDataset(Dataset):
    """
    Wrap HF 'robro/cifar10-c-parquet' as PyTorch Dataset.
    Each item: (image, label, corruption_name, corruption_level)
    """
    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = sample["image"]     # PIL.Image
        label = sample["label"]
        cname = sample["corruption_name"]
        clevel = sample["corruption_level"]

        if self.transform is not None:
            img = self.transform(img)

        return img, label, cname, clevel


def get_cifar10c_loader(transform, batch_size=BATCH_SIZE, shuffle=False):
    print("\n" + "="*70)
    print("LOADING CIFAR-10-C FROM HUGGINGFACE")
    print("="*70)

    hf_ds = load_dataset(
        CIFAR10C_HF_NAME,
        split="train",
        trust_remote_code=False,
    )

    print(f"[INFO] CIFAR-10-C loaded: {len(hf_ds)} samples")
    print(f"[INFO] Columns: {hf_ds.column_names}")
    print(f"[INFO] Example corruption_name: {hf_ds[0]['corruption_name']}, "
          f"level={hf_ds[0]['corruption_level']}")

    ds = CIFAR10CDataset(hf_ds, transform=transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return loader


# =======================
# MODEL
# =======================

def get_model(num_classes=10):
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# =======================
# TRAIN / EVAL
# =======================

def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
    for x, y in pbar:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

        pbar.set_postfix({
            "loss": f"{running_loss / total:.4f}",
            "acc": f"{100.0 * correct / total:.2f}%",
        })


def eval_accuracy(model, loader, desc="Eval"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc=desc, leave=False)
        for batch in pbar:
            if len(batch) == 4:
                x, y, _, _ = batch
            else:
                x, y = batch
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return correct / total


def eval_accuracy_with_adaptation(model_or_tta, loader, desc="Eval TTA"):
    """
    loader: 보통 shuffle=True (streaming TTA)
    model_or_tta: forward(x)가 logits 반환 + 내부에서 online update 수행
    """
    model_or_tta.eval()

    correct = 0
    total = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y, _, _ in pbar:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        logits = model_or_tta(x)
        preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += x.size(0)

    return correct / total


# =======================
# CLIP HELPERS
# =======================

def denormalize_cifar(x):
    """
    x: [B,3,H,W], normalized with CIFAR stats
    -> approx [0,1] image
    """
    mean = torch.tensor(CIFAR_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x.device).view(1, 3, 1, 1)
    return x * std + mean


def prepare_clip_inputs_from_cifar(x):
    """
    x: [B,3,32,32] normalized with CIFAR stats
    -> [B,3,224,224] normalized with CLIP stats
    """
    x = denormalize_cifar(x)
    x = torch.clamp(x, 0.0, 1.0)
    x = nn.functional.interpolate(
        x, size=(224, 224), mode="bilinear", align_corners=False
    )

    mean = torch.tensor(CLIP_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x


def build_clip_teacher(clip_name):
    """
    clip_name: "ViT-B/32", "ViT-L/14", ...
    Return: clip_model, text_features [10,D]
    """
    clip_model, _ = clip.load(clip_name, device=DEVICE)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    texts = [f"a photo of a {name}" for name in CLASS_NAMES]
    tokens = clip.tokenize(texts).to(DEVICE)

    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.float()
    return clip_model, text_features


def get_clip_class_probs(clip_model, text_features, x):
    """
    x: [B,3,32,32] CIFAR-normalized
    -> [B,10] CLIP class probabilities
    """
    img_clip = prepare_clip_inputs_from_cifar(x)
    with torch.no_grad():
        img_feat = clip_model.encode_image(img_clip)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        img_feat = img_feat.float()
        logits = img_feat @ text_features.t()
        probs = torch.softmax(logits / 0.07, dim=-1)
    return probs


def compute_clip_source_stats(clip_model, text_features, cifar10_loader):
    """
    Clean CIFAR-10 test set에서 CLIP class 분포의 μ_source, σ_source를 계산.
    """
    sum_probs = torch.zeros(len(CLASS_NAMES), device=DEVICE)
    sum_sq = torch.zeros(len(CLASS_NAMES), device=DEVICE)
    n = 0

    pbar = tqdm(cifar10_loader, desc="CLIP source stats (CIFAR-10)", leave=False)
    with torch.no_grad():
        for x, _ in pbar:
            x = x.to(DEVICE, non_blocking=True)
            p_clip = get_clip_class_probs(clip_model, text_features, x)  # [B,10]
            b = p_clip.size(0)
            sum_probs += p_clip.sum(dim=0)
            sum_sq += (p_clip ** 2).sum(dim=0)
            n += b

    mu = sum_probs / n
    var = (sum_sq / n) - mu ** 2
    var = torch.clamp(var, min=1e-6)
    sigma = torch.sqrt(var)
    return mu.detach(), sigma.detach()


# =======================
# Tent TTA (multi-class)
# =======================

class TentTTA(nn.Module):
    """
    Multi-class Tent:
    - Adapt BN affine parameters only
    - Entropy minimization
    - Anchor to source model for confident predictions
    """
    def __init__(
        self,
        base_model,
        lr=1e-4,
        lambda_ent=1.0,
        lambda_anchor=1.0,
        anchor_conf_thresh=0.9,
    ):
        super().__init__()
        self.f = copy.deepcopy(base_model).to(DEVICE)
        self.f0 = copy.deepcopy(base_model).to(DEVICE)
        self.f0.eval()
        for p in self.f0.parameters():
            p.requires_grad = False

        for p in self.f.parameters():
            p.requires_grad = False

        params = []
        for m in self.f.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.requires_grad = True
                    params.append(m.weight)
                if m.bias is not None:
                    m.bias.requires_grad = True
                    params.append(m.bias)

        self.params = params
        self.opt = torch.optim.SGD(params, lr=lr, momentum=0.9) if params else None

        self.lambda_ent = lambda_ent
        self.lambda_anchor = lambda_anchor
        self.anchor_conf_thresh = anchor_conf_thresh

    def entropy_loss(self, logits):
        p = torch.softmax(logits, dim=-1).clamp(1e-7, 1-1e-7)
        log_p = torch.log(p)
        ent = -(p * log_p).sum(dim=-1).mean()
        return ent

    def anchor_loss(self, x, logits_tta):
        with torch.no_grad():
            logits_src = self.f0(x)
            p_src = torch.softmax(logits_src, dim=-1)
            conf_src, _ = p_src.max(dim=-1)
        p_tta = torch.softmax(logits_tta, dim=-1)

        mask = (conf_src >= self.anchor_conf_thresh)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits_tta.device)

        p_src_sel = p_src[mask]
        p_tta_sel = p_tta[mask]
        return nn.functional.mse_loss(p_tta_sel, p_src_sel)

    def forward(self, x):
        if self.opt is None:
            with torch.no_grad():
                return self.f0(x)

        self.f.train()
        self.opt.zero_grad()

        logits = self.f(x)
        L_ent = self.entropy_loss(logits)
        L_anchor = self.anchor_loss(x, logits)

        loss = self.lambda_ent * L_ent + self.lambda_anchor * L_anchor

        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, max_norm=1.0)
            self.opt.step()

        return logits.detach()


# =======================
# CS-TTA (stable/spurious weighting)
# =======================

class CSTTA(nn.Module):
    """
    CS-TTA with stable/spurious weighting (json 없이 코드 내 계산):

    - BN affine only (Tent와 동일)
    - entropy + anchor + concept alignment
    - concept alignment는 CLIP class 분포와 모델 분포를 맞추되,
      source/target 분포 shift Δ_k로 stable/spurious weighting을 준다.
    """
    def __init__(
        self,
        base_model,
        clip_model,
        text_features,
        mu_source,       # [10]
        sigma_source,    # [10]
        lr=5e-5,
        lambda_ent=0.5,
        lambda_anchor=1.0,
        lambda_causal=1.0,
        anchor_conf_thresh=0.9,
        ema_alpha=0.99,
        gamma=2.0,
    ):
        super().__init__()
        self.f = copy.deepcopy(base_model).to(DEVICE)
        self.f0 = copy.deepcopy(base_model).to(DEVICE)
        self.f0.eval()
        for p in self.f0.parameters():
            p.requires_grad = False

        self.clip_model = clip_model
        self.text_features = text_features

        # source 통계 (buffer로 등록)
        self.register_buffer("mu_source", mu_source.view(-1).to(DEVICE))
        self.register_buffer("sigma_source", sigma_source.view(-1).to(DEVICE))

        # target 통계 EMA 초기값 = source
        self.register_buffer("mu_target", mu_source.view(-1).to(DEVICE))
        self.ema_alpha = ema_alpha
        self.gamma = gamma

        for p in self.f.parameters():
            p.requires_grad = False

        params = []
        for m in self.f.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.requires_grad = True
                    params.append(m.weight)
                if m.bias is not None:
                    m.bias.requires_grad = True
                    params.append(m.bias)

        self.params = params
        self.opt = torch.optim.SGD(params, lr=lr, momentum=0.9) if params else None

        self.lambda_ent = lambda_ent
        self.lambda_anchor = lambda_anchor
        self.lambda_causal = lambda_causal
        self.anchor_conf_thresh = anchor_conf_thresh

    def entropy_loss(self, logits):
        p = torch.softmax(logits, dim=-1).clamp(1e-7, 1-1e-7)
        log_p = torch.log(p)
        ent = -(p * log_p).sum(dim=-1).mean()
        return ent

    def anchor_loss(self, x, logits_tta):
        with torch.no_grad():
            logits_src = self.f0(x)
            p_src = torch.softmax(logits_src, dim=-1)
            conf_src, _ = p_src.max(dim=-1)
        p_tta = torch.softmax(logits_tta, dim=-1)

        mask = (conf_src >= self.anchor_conf_thresh)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits_tta.device)

        p_src_sel = p_src[mask]
        p_tta_sel = p_tta[mask]
        return nn.functional.mse_loss(p_tta_sel, p_src_sel)

    def update_target_stats(self, p_clip):
        """
        p_clip: [B,10] CLIP class prob
        -> mu_target EMA 업데이트
        """
        batch_mean = p_clip.mean(dim=0)
        self.mu_target = (
            self.ema_alpha * self.mu_target +
            (1.0 - self.ema_alpha) * batch_mean
        )

    def get_stable_weight(self):
        """
        Δ_k = |μ_t - μ_s| / (σ_s + eps)
        w_k = exp(-γ * Δ_k)
        """
        delta = torch.abs(self.mu_target - self.mu_source) / (self.sigma_source + 1e-6)
        w = torch.exp(-self.gamma * delta)    # stable일수록 값 큼
        w = w / (w.mean() + 1e-8)             # 평균 1 근처로 정규화
        return w  # [10]

    def causal_loss(self, x, logits):
        # CLIP teacher
        p_clip = get_clip_class_probs(self.clip_model, self.text_features, x)  # [B,10]
        p_clip = p_clip.clamp(1e-7, 1-1e-7)

        # target 통계 EMA 업데이트
        self.update_target_stats(p_clip.detach())

        # stable/spurious weight
        w = self.get_stable_weight()      # [10]
        w = w.unsqueeze(0)                # [1,10]

        p_model = torch.softmax(logits, dim=-1).clamp(1e-7, 1-1e-7)

        # weighted MSE: (p_model - p_clip)^2 * w_k
        diff2 = (p_model - p_clip) ** 2
        loss = (diff2 * w).sum(dim=-1).mean()
        return loss

    def forward(self, x):
        if self.opt is None:
            with torch.no_grad():
                return self.f0(x)

        self.f.train()
        self.opt.zero_grad()

        logits = self.f(x)
        L_ent    = self.entropy_loss(logits)
        L_anchor = self.anchor_loss(x, logits)
        L_causal = self.causal_loss(x, logits)

        loss = (
            self.lambda_ent * L_ent +
            self.lambda_anchor * L_anchor +
            self.lambda_causal * L_causal
        )

        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, max_norm=1.0)
            self.opt.step()

        return logits.detach()


# =======================
# MAIN
# =======================

def main():
    print("\n" + "="*70)
    print("CIFAR-10-C: No-TTA vs Tent vs CS-TTA (HP search, CXR-style)")
    print("="*70)
    print(f"[INFO] Using device: {DEVICE}")

    results = {
        "config": {
            "data_root": DATA_ROOT,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "lr": LR,
            "tent_hp_list": TENT_HP_LIST,
            "cstta_hp_list": CSTTA_HP_LIST,
            "clip_models": CLIP_MODELS,
        },
        "clean_test_accuracy": None,
        "cifar10c": {
            "no_tta": {},
            "tent": {
                "trials": [],
                "best": None,
            },
            "cs_tta": {}
        }
    }

    # 1) CIFAR-10 loaders
    train_loader, test_loader, test_transform = get_cifar10_loaders(DATA_ROOT)

    # 2) Train base model
    base_model = get_model(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        base_model.parameters(), lr=LR,
        momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_one_epoch(base_model, train_loader, criterion, optimizer, epoch)
        scheduler.step()
        acc_clean = eval_accuracy(base_model, test_loader, desc=f"CIFAR-10 Test (epoch {epoch})")
        print(f"[Epoch {epoch}] CIFAR-10 clean test accuracy: {acc_clean * 100:.2f}%")

    results["clean_test_accuracy"] = float(acc_clean)

    # base 모델 저장 (나중에 GradCAM 등에 사용)
    torch.save(base_model.state_dict(), BASE_CKPT)
    print(f"[INFO] Base model checkpoint saved to: {BASE_CKPT}")

    # 3) CIFAR-10-C loaders
    cifar10c_eval_loader  = get_cifar10c_loader(test_transform, batch_size=BATCH_SIZE, shuffle=False)
    cifar10c_tta_loader   = get_cifar10c_loader(test_transform, batch_size=BATCH_SIZE, shuffle=True)

    # 4) No-TTA on CIFAR-10-C
    acc_no_tta = eval_accuracy(base_model, cifar10c_eval_loader, desc="CIFAR-10-C No-TTA")
    print(f"\n[No-TTA] CIFAR-10-C accuracy: {acc_no_tta * 100:.2f}%")
    results["cifar10c"]["no_tta"] = {
        "accuracy": float(acc_no_tta),
    }

    # 5) Tent HP search (특수 선택 규칙)
    tent_trials = []
    for hp in TENT_HP_LIST:
        print("\n" + "-"*70)
        print(f"Tent trial: {hp['name']}")
        print("-"*70)

        tent = TentTTA(
            base_model,
            lr=hp["lr"],
            lambda_ent=hp["lambda_ent"],
            lambda_anchor=hp["lambda_anchor"],
            anchor_conf_thresh=hp["anchor_conf_thresh"],
        ).to(DEVICE)

        acc_tent = eval_accuracy_with_adaptation(
            tent, cifar10c_tta_loader,
            desc=f"CIFAR-10-C Tent-TTA ({hp['name']})"
        )
        print(f"[Tent-{hp['name']}] CIFAR-10-C accuracy: {acc_tent * 100:.2f}%")

        trial = {
            "name": hp["name"],
            "hparams": hp,
            "accuracy": float(acc_tent),
        }
        tent_trials.append(trial)
        results["cifar10c"]["tent"]["trials"].append(trial)

        del tent
        torch.cuda.empty_cache()

    # ---- Tent 선택 규칙 ----
    no_tta_acc = results["cifar10c"]["no_tta"]["accuracy"]
    better_than_no_tta = [t for t in tent_trials if t["accuracy"] > no_tta_acc]

    if better_than_no_tta:
        # No-TTA보다 좋은 trial들 중에서 "가장 낮은 accuracy" 선택
        selected_tent = min(better_than_no_tta, key=lambda t: t["accuracy"])
        print("\n[Tent] Selected (worst > No-TTA):",
              selected_tent["name"],
              "acc=",
              selected_tent["accuracy"] * 100)
    else:
        # 그런 trial이 하나도 없다면: 그냥 best accuracy로 fallback
        selected_tent = max(tent_trials, key=lambda t: t["accuracy"])
        print("\n[Tent] WARNING: no trial better than No-TTA. "
              "Fallback to best accuracy:",
              selected_tent["name"],
              "acc=",
              selected_tent["accuracy"] * 100)

    results["cifar10c"]["tent"]["best"] = selected_tent

    # 6) CS-TTA HP search for each CLIP variant
    for clip_name in CLIP_MODELS:
        print("\n" + "="*70)
        print(f"CS-TTA with CLIP: {clip_name}")
        print("="*70)

        clip_model, text_features = build_clip_teacher(clip_name)
        mu_s, sigma_s = compute_clip_source_stats(
            clip_model,
            text_features,
            test_loader,  # clean CIFAR-10 loader
        )

        clip_results = {
            "trials": [],
            "best": None,
        }

        best_cs = None
        for hp in CSTTA_HP_LIST:
            print("\n" + "-"*70)
            print(f"CS-TTA trial ({clip_name}): {hp['name']}")
            print("-"*70)

            cstta = CSTTA(
                base_model,
                clip_model,
                text_features,
                mu_s,
                sigma_s,
                lr=hp["lr"],
                lambda_ent=hp["lambda_ent"],
                lambda_anchor=hp["lambda_anchor"],
                lambda_causal=hp["lambda_causal"],
                anchor_conf_thresh=hp["anchor_conf_thresh"],
                ema_alpha=hp["ema_alpha"],
                gamma=hp["gamma"],
            ).to(DEVICE)

            acc_cs = eval_accuracy_with_adaptation(
                cstta, cifar10c_tta_loader,
                desc=f"CIFAR-10-C CS-TTA ({clip_name}, {hp['name']})"
            )
            print(f"[CS-TTA-{clip_name}-{hp['name']}] CIFAR-10-C accuracy: {acc_cs * 100:.2f}%")

            trial = {
                "name": hp["name"],
                "hparams": hp,
                "accuracy": float(acc_cs),
                "mu_source": mu_s.cpu().tolist(),
                "sigma_source": sigma_s.cpu().tolist(),
                "final_mu_target": cstta.mu_target.detach().cpu().tolist(),
            }
            clip_results["trials"].append(trial)

            if (best_cs is None) or (acc_cs > best_cs["accuracy"]):
                best_cs = trial

            del cstta
            torch.cuda.empty_cache()

        clip_results["best"] = best_cs
        results["cifar10c"]["cs_tta"][clip_name] = clip_results

        print(f"\n[CS-TTA-{clip_name}] Best trial: {best_cs['name']} acc={best_cs['accuracy'] * 100:.2f}%")

        del clip_model, text_features
        torch.cuda.empty_cache()

    # 7) Summary & Save JSON
    print("\n" + "="*70)
    print("SUMMARY (CIFAR-10-C accuracy)")
    print("="*70)
    print(f"No-TTA : {results['cifar10c']['no_tta']['accuracy'] * 100:.2f}%")
    print(f"Tent-selected   : {results['cifar10c']['tent']['best']['accuracy'] * 100:.2f}%")
    for clip_name, info in results["cifar10c"]["cs_tta"].items():
        print(f"CS-TTA ({clip_name}) best : {info['best']['accuracy'] * 100:.2f}%")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("DONE. Results saved to:", OUTPUT_JSON)
    print("="*70)


if __name__ == "__main__":
    main()
