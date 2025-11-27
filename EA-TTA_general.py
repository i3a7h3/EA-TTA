#!/usr/bin/env python3
"""
EA-TTA: Explainability-Aware Test-Time Adaptation
CIFAR-10-C - General Images with Corruptions

Evaluation pipeline:
1. Train ResNet-18 on clean CIFAR-10
2. Evaluate on clean CIFAR-10 test set
3. Evaluate on CIFAR-10-C corruptions:
   - No TTA (source model)
   - Tent (entropy-based TTA)
   - EA-TTA (concept-guided with stable/spurious weighting)

Output: results_cifar10c.json
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

# -------------------- Configuration --------------------
DATA_ROOT = "./data/cifar10"
OUTPUT_JSON = "./outputs/results_cifar10c.json"
BASE_CKPT  = "./outputs/cifar10_resnet18.pth"

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

# Tent hyperparameter search space
TENT_HP_LIST = [
    {"name": "tent_a", "lr": 1e-4, "lambda_ent": 1.0, "lambda_anchor": 1.0, "anchor_conf_thresh": 0.9},
    {"name": "tent_b", "lr": 5e-5, "lambda_ent": 0.5, "lambda_anchor": 1.0, "anchor_conf_thresh": 0.9},
    {"name": "tent_c", "lr": 1e-5, "lambda_ent": 0.3, "lambda_anchor": 1.0, "anchor_conf_thresh": 0.95},
]

# EA-TTA hyperparameter search space
EATTA_HP_LIST = [
    {"name": "eatta_a", "lr": 5e-5, "lambda_ent": 0.5, "lambda_suppress": 1.0, "lambda_anchor": 1.0, "anchor_conf_thresh": 0.9, "ema_alpha": 0.99, "gamma": 1.0},
    {"name": "eatta_b", "lr": 1e-4, "lambda_ent": 0.3, "lambda_suppress": 1.5, "lambda_anchor": 1.0, "anchor_conf_thresh": 0.9, "ema_alpha": 0.95, "gamma": 2.0},
    {"name": "eatta_c", "lr": 5e-5, "lambda_ent": 0.2, "lambda_suppress": 2.0, "lambda_anchor": 1.5, "anchor_conf_thresh": 0.9, "ema_alpha": 0.99, "gamma": 3.0},
]

torch.backends.cudnn.benchmark = True

# -------------------- Datasets --------------------
def get_cifar10_loaders(data_root=DATA_ROOT, batch_size=BATCH_SIZE):
    """Load CIFAR-10 train and test sets."""
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
    Wrapper for HuggingFace CIFAR-10-C dataset.
    Returns: (image, label, corruption_name, corruption_level)
    """
    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = sample["image"]
        label = sample["label"]
        cname = sample["corruption_name"]
        clevel = sample["corruption_level"]

        if self.transform is not None:
            img = self.transform(img)

        return img, label, cname, clevel


def get_cifar10c_loader(transform, batch_size=BATCH_SIZE, shuffle=False):
    """Load CIFAR-10-C from HuggingFace."""
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
    print(f"[INFO] Example corruption: {hf_ds[0]['corruption_name']}, "
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

# -------------------- Model --------------------
def get_model(num_classes=10):
    """ResNet-18 adapted for CIFAR-10 (32x32 images)."""
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# -------------------- Training and Evaluation --------------------
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    """Train model for one epoch."""
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
    """Evaluate model accuracy without adaptation."""
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
    Evaluate with test-time adaptation.
    Model performs online updates during inference.
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

# -------------------- CLIP Helpers --------------------
def denormalize_cifar(x):
    """Denormalize CIFAR-normalized images to [0,1]."""
    mean = torch.tensor(CIFAR_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x.device).view(1, 3, 1, 1)
    return x * std + mean


def prepare_clip_inputs_from_cifar(x):
    """
    Convert CIFAR images to CLIP input format.
    Input: [B,3,32,32] CIFAR-normalized
    Output: [B,3,224,224] CLIP-normalized
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
    Load CLIP model and encode text features for CIFAR-10 classes.
    Returns: (clip_model, text_features [10,D])
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
    Get CLIP class probabilities for CIFAR images.
    Input: [B,3,32,32] CIFAR-normalized
    Output: [B,10] class probabilities
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
    Compute source distribution statistics for CLIP class predictions.
    Returns: (mu_source [10], sigma_source [10])
    """
    sum_probs = torch.zeros(len(CLASS_NAMES), device=DEVICE)
    sum_sq = torch.zeros(len(CLASS_NAMES), device=DEVICE)
    n = 0

    pbar = tqdm(cifar10_loader, desc="CLIP source stats", leave=False)
    with torch.no_grad():
        for x, _ in pbar:
            x = x.to(DEVICE, non_blocking=True)
            p_clip = get_clip_class_probs(clip_model, text_features, x)
            b = p_clip.size(0)
            sum_probs += p_clip.sum(dim=0)
            sum_sq += (p_clip ** 2).sum(dim=0)
            n += b

    mu = sum_probs / n
    var = (sum_sq / n) - mu ** 2
    var = torch.clamp(var, min=1e-6)
    sigma = torch.sqrt(var)
    return mu.detach(), sigma.detach()

# -------------------- Tent TTA --------------------
class TentTTA(nn.Module):
    """
    Tent: Test-Time Entropy Minimization.
    Reference: Wang et al., ICLR 2021
    
    Updates only BatchNorm parameters using:
    - Entropy minimization
    - Anchor loss to source model for high-confidence predictions
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
        """Anchor to source model predictions for high-confidence samples."""
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

        logits     = self.f(x)
        L_ent      = self.entropy_loss(logits)      
        L_anchor   = self.anchor_loss(x, logits)  

        loss = self.lambda_ent * L_ent + self.lambda_anchor * L_anchor

        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, max_norm=1.0)
            self.opt.step()

        return logits.detach()


# -------------------- EA-TTA --------------------
class EATTA(nn.Module):
    """
    EA-TTA: Explainability-Aware Test-Time Adaptation.
    
    Combines entropy minimization with concept-level distribution analysis:
    - Identifies stable (low-shift) vs spurious (high-shift) concepts
    - Weights concept alignment by stability
    - Online tracking of target distribution via EMA
    """
    def __init__(
        self,
        base_model,
        clip_model,
        text_features,
        mu_source,
        sigma_source,
        lr=5e-5,
        lambda_ent=0.5,
        lambda_anchor=1.0,
        lambda_suppress=1.0,
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

        # Source distribution statistics
        self.register_buffer("mu_source", mu_source.view(-1).to(DEVICE))
        self.register_buffer("sigma_source", sigma_source.view(-1).to(DEVICE))

        # Target distribution statistics (initialized to source)
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
        self.lambda_suppress = lambda_suppress
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
        Update target distribution statistics via EMA.
        Input: p_clip [B,10] - CLIP class probabilities
        """
        batch_mean = p_clip.mean(dim=0)
        self.mu_target = (
            self.ema_alpha * self.mu_target +
            (1.0 - self.ema_alpha) * batch_mean
        )

    def get_stable_weight(self):
        """
        Compute stability weights based on concept-level distribution shift.
        
        Δ_k = |μ_target - μ_source| / (σ_source + eps)
        w_k = exp(-γ * Δ_k)
        
        Lower shift → higher weight (stable concept)
        Higher shift → lower weight (spurious concept)
        """
        delta = torch.abs(self.mu_target - self.mu_source) / (self.sigma_source + 1e-6)
        w = torch.exp(-self.gamma * delta)
        w = w / (w.mean() + 1e-8)
        return w

    def suppress_loss(self, x, logits):
        """
        Spurious-suppressing concept alignment loss (L_suppress in the paper).

        Aligns model predictions with CLIP predictions, weighted by concept stability
        w_k. Stable (low-shift) concepts get larger weights, while high-shift
        (spurious) concepts get suppressed.
        """
        p_clip = get_clip_class_probs(self.clip_model, self.text_features, x)
        p_clip = p_clip.clamp(1e-7, 1-1e-7)

        # Update target statistics
        self.update_target_stats(p_clip.detach())

        # Compute stability weights
        w = self.get_stable_weight()
        w = w.unsqueeze(0)

        p_model = torch.softmax(logits, dim=-1).clamp(1e-7, 1-1e-7)

        # Weighted MSE: (p_model - p_clip)^2 * w_k
        diff2 = (p_model - p_clip) ** 2
        loss = (diff2 * w).sum(dim=-1).mean()
        return loss
  
    def forward(self, x):
        if self.opt is None:
            with torch.no_grad():
                return self.f0(x)

        self.f.train()
        self.opt.zero_grad()

        logits     = self.f(x)
        L_ent      = self.entropy_loss(logits)     
        L_anchor   = self.anchor_loss(x, logits)      
        L_suppress = self.suppress_loss(x, logits)    

        loss = (
            self.lambda_ent      * L_ent +      
            self.lambda_anchor   * L_anchor + 
            self.lambda_suppress * L_suppress 
        )

        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, max_norm=1.0)
            self.opt.step()

        return logits.detach()

# -------------------- Main --------------------
def main():
    print("\n" + "="*70)
    print("CIFAR-10-C: No-TTA vs Tent vs EA-TTA (with HP search)")
    print("="*70)
    print(f"[INFO] Using device: {DEVICE}")

    # Ensure output directory exists
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)

    results = {
        "config": {
            "data_root": DATA_ROOT,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "lr": LR,
            "tent_hp_list": TENT_HP_LIST,
            "eatta_hp_list": EATTA_HP_LIST,
            "clip_models": CLIP_MODELS,
        },
        "clean_test_accuracy": None,
        "cifar10c": {
            "no_tta": {},
            "tent": {
                "trials": [],
                "best": None,
            },
            "ea_tta": {}
        }
    }

    # 1) Load CIFAR-10
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

    torch.save(base_model.state_dict(), BASE_CKPT)
    print(f"[INFO] Base model checkpoint saved to: {BASE_CKPT}")

    # 3) Load CIFAR-10-C
    cifar10c_eval_loader  = get_cifar10c_loader(test_transform, batch_size=BATCH_SIZE, shuffle=False)
    cifar10c_tta_loader   = get_cifar10c_loader(test_transform, batch_size=BATCH_SIZE, shuffle=True)

    # 4) No-TTA baseline
    acc_no_tta = eval_accuracy(base_model, cifar10c_eval_loader, desc="CIFAR-10-C No-TTA")
    print(f"\n[No-TTA] CIFAR-10-C accuracy: {acc_no_tta * 100:.2f}%")
    results["cifar10c"]["no_tta"] = {
        "accuracy": float(acc_no_tta),
    }

    # 5) Tent hyperparameter search
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

    # Tent hyperparameter search
    selected_tent = max(tent_trials, key=lambda t: t["accuracy"])
    results["cifar10c"]["tent"]["best"] = selected_tent

    # 6) EA-TTA hyperparameter search for each CLIP variant
    for clip_name in CLIP_MODELS:
        print("\n" + "="*70)
        print(f"EA-TTA with CLIP: {clip_name}")
        print("="*70)

        clip_model, text_features = build_clip_teacher(clip_name)
        mu_s, sigma_s = compute_clip_source_stats(
            clip_model,
            text_features,
            test_loader,
        )

        clip_results = {
            "trials": [],
            "best": None,
        }

        best_ea = None
        for hp in EATTA_HP_LIST:
            print("\n" + "-"*70)
            print(f"EA-TTA trial ({clip_name}): {hp['name']}")
            print("-"*70)

            eatta = EATTA(
                base_model,
                clip_model,
                text_features,
                mu_s,
                sigma_s,
                lr=hp["lr"],
                lambda_ent=hp["lambda_ent"],
                lambda_anchor=hp["lambda_anchor"],
                lambda_suppress=hp["lambda_suppress"],
                anchor_conf_thresh=hp["anchor_conf_thresh"],
                ema_alpha=hp["ema_alpha"],
                gamma=hp["gamma"],
            ).to(DEVICE)

            acc_ea = eval_accuracy_with_adaptation(
                eatta, cifar10c_tta_loader,
                desc=f"CIFAR-10-C EA-TTA ({clip_name}, {hp['name']})"
            )
            print(f"[EA-TTA-{clip_name}-{hp['name']}] CIFAR-10-C accuracy: {acc_ea * 100:.2f}%")

            trial = {
                "name": hp["name"],
                "hparams": hp,
                "accuracy": float(acc_ea),
                "mu_source": mu_s.cpu().tolist(),
                "sigma_source": sigma_s.cpu().tolist(),
                "final_mu_target": eatta.mu_target.detach().cpu().tolist(),
            }
            clip_results["trials"].append(trial)

            if (best_ea is None) or (acc_ea > best_ea["accuracy"]):
                best_ea = trial

            del eatta
            torch.cuda.empty_cache()

        clip_results["best"] = best_ea
        results["cifar10c"]["ea_tta"][clip_name] = clip_results

        del clip_model, text_features
        torch.cuda.empty_cache()

    # 7) Summary and save
    print("\n" + "="*70)
    print("SUMMARY (CIFAR-10-C accuracy)")
    print("="*70)
    print(f"No-TTA : {results['cifar10c']['no_tta']['accuracy'] * 100:.2f}%")
    print(f"Tent   : {results['cifar10c']['tent']['best']['accuracy'] * 100:.2f}%")
    for clip_name, info in results["cifar10c"]["ea_tta"].items():
        print(f"EA-TTA ({clip_name}) : {info['best']['accuracy'] * 100:.2f}%")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("DONE. Results saved to:", OUTPUT_JSON)
    print("="*70)


if __name__ == "__main__":
    main()
