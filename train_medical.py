#!/usr/bin/env python3
"""
Binary Enlarged Cardiomediastinum Classification
+ Tent / CS-TTA + Grad-CAM (4-panel + per-method)

- GPU 0
- Task: Enlarged Cardiomediastinum vs Normal

- Data ROOT:
    /data/dusrb37/cvpr2026/1_EA-TTA/medical_test_1114

- For each combo (CheXpert, CLIP-B32/L14):
    gradcam-v_pneumo_enlarged/Enlarged Cardiomediastinum/<MODEL>__<CLIP>/FourPanel/<img>.png
    gradcam-v_pneumo_enlarged/Enlarged Cardiomediastinum/<MODEL>__<CLIP>/NoTTA/<img>.png
    gradcam-v_pneumo_enlarged/Enlarged Cardiomediastinum/<MODEL>__<CLIP>/Tent/<img>.png
    gradcam-v_pneumo_enlarged/Enlarged Cardiomediastinum/<MODEL>__<CLIP>/CSTTA/<img>.png

- Lists:
    lists-v_pneumo_enlarged/Enlarged Cardiomediastinum__CheXpert__<CLIP>__cond1_CS_better_than_both.(csv|json)
    lists-v_pneumo_enlarged/Enlarged Cardiomediastinum__CheXpert__<CLIP>__cond2_Tent_better_than_No_then_CS_best.(csv|json)

- Results JSON:
    results_binary_enlarged_only_gradcam.json
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
import json, copy, csv
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import matplotlib.pyplot as plt
import cv2
import clip  # OpenAI CLIP
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

# -------------------- Paths / Config --------------------
ROOT = Path("/data/dusrb37/cvpr2026/1_EA-TTA/medical_test_1114")

TASK_NAME = "Enlarged Cardiomediastinum"
PATHOLOGY_NAME = "Enlarged Cardiomediastinum"

TEST_DIR             = ROOT / "datasets/test-normal-enlarged"
TEST_NORMAL_LABELDIR = ROOT / "datasets/test-normal-forlabel"
TEST_POS_LABELDIR    = ROOT / "datasets/test-Enlarged Cardiomediastinum-forlabel"

GRADCAM_ROOT = ROOT / "gradcam-v_pneumo_enlarged-22"
LISTS_ROOT   = ROOT / "lists-v_pneumo_enlarged-22"
RESULTS_JSON = ROOT / "results_binary_enlarged_only_gradcam-22.json"
GRADCAM_ROOT.mkdir(parents=True, exist_ok=True)
LISTS_ROOT.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda:0")

# 이 스크립트에서는 Enlarged Cardiomediastinum + CheXpert만 사용
COMBOS = [
    ("CheXpert", "CLIP-B32"),
    ("CheXpert", "CLIP-L14"),
]

PRETRAINED_MODELS = {
    "CheXpert": "densenet121-res224-chex",
}

# CLIP: B32, L14
CLIP_MODELS_CONFIG = {
    "CLIP-B32": {"name": "ViT-B/32"},
    "CLIP-L14": {"name": "ViT-L/14"},
}

# 개념 목록: 폐 관련 + Pneumothorax/Enlarged cardiomediastinum 포함
concepts_config = {
    "pathology": [
        "pneumonia", "consolidation", "infiltration",
        "atelectasis", "pleural effusion", "pulmonary edema",
        "pneumothorax", "enlarged cardiomediastinum",
    ],
    "anatomy": [
        "lung field", "heart silhouette", "diaphragm",
        "mediastinum", "costophrenic angle", "trachea",
    ],
    "devices": [
        "endotracheal tube", "central venous line", "nasogastric tube",
        "chest tube", "pacemaker wires", "monitoring electrodes",
    ],
    "acquisition": [
        "portable chest x-ray", "anteroposterior view", "posteroanterior view",
        "supine positioning", "upright positioning", "bedside radiograph",
    ],
    "artifacts": [
        "patient identification marker", "date and time stamp",
        "radiation dose indicator", "anatomical position marker",
        "image quality indicator",
    ],
}
all_concepts = []
concept_type = {}
for cat, items in concepts_config.items():
    for c in items:
        all_concepts.append(c)
        concept_type[c] = cat
num_concepts = len(all_concepts)
print(f"[INFO] Total concepts: {num_concepts}")

BALANCE_PRIOR = 0.5
MAX_GRADCAM_SAMPLES = None  # None => all

# -------------------- Stable weights (no JSON) --------------------
def build_stable_weight() -> torch.Tensor:
    """
    concept_analysis.json 없이 사용하는 기본 가중치:
    - 초기: 전부 1
    - pathology / anatomy: ×1.5
    - devices / artifacts: ×0.5
    """
    w = torch.ones(len(all_concepts), dtype=torch.float32)
    for i, c in enumerate(all_concepts):
        if concept_type[c] in ["pathology", "anatomy"]:
            w[i] *= 1.5
        elif concept_type[c] in ["devices", "artifacts"]:
            w[i] *= 0.5
    return w

# -------------------- Dataset --------------------
class TestWithLabelsDataset(Dataset):
    def __init__(self, test_dir, normal_label_dir, pos_label_dir):
        test_dir = Path(test_dir)
        normal_label_dir = Path(normal_label_dir)
        pos_label_dir = Path(pos_label_dir)

        files = sorted(test_dir.glob("*.png"))
        label_map = {}
        for p in normal_label_dir.glob("*.png"):
            label_map[p.name] = 0
        for p in pos_label_dir.glob("*.png"):
            if p.name in label_map:
                raise RuntimeError(f"File appears in both label dirs: {p.name}")
            label_map[p.name] = 1

        self.items = [(p, label_map[p.name]) for p in files if p.name in label_map]
        missing = len(files) - len(self.items)

        print(f"[INFO] Labeled test samples: {len(self.items)} (ignored unlabeled: {missing})")
        ys = [lab for _, lab in self.items]
        pos = sum(ys)
        neg = len(ys) - pos
        print(f"[INFO] Label distribution: positive={pos}, normal={neg}")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        img = np.array(Image.open(path)).astype(np.float32)
        # [0,255] -> [-1024,1024]
        img = (img / 255.0) * 2048.0 - 1024.0
        x = torch.from_numpy(img[None, ...]).float()
        return x, torch.tensor(y, dtype=torch.long), path.name

# -------------------- Models --------------------
class BinaryClassifier(nn.Module):
    def __init__(self, backbone, pathology_idx):
        super().__init__()
        self.backbone = backbone
        self.idx = pathology_idx

    def forward(self, x):
        out = self.backbone(x)
        return out[:, self.idx:self.idx+1]

    def get_features(self, x):
        feats = self.backbone.features(x)
        pooled = F.adaptive_avg_pool2d(feats, 1)
        return pooled.view(pooled.size(0), -1)

# -------------------- CLIP helpers --------------------
def load_openai_clip(clip_tag: str):
    model, _ = clip.load(CLIP_MODELS_CONFIG[clip_tag]["name"], device=device)
    model.float()  # fp32
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    texts = [f"a chest x-ray showing {c}" for c in all_concepts]
    tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        txt = model.encode_text(tokens)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        txt = txt.float()
    return model, txt

def prepare_clip_inputs_from_xrv(x):
    img = (x + 1024.0) / 2048.0
    img = img.clamp(0.0, 1.0).repeat(1, 3, 1, 1)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                        device=img.device).view(1,3,1,1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                        device=img.device).view(1,3,1,1)
    return (img - mean) / std

@torch.no_grad()
def get_clip_concept_scores(clip_model, concept_emb, x):
    img_clip = prepare_clip_inputs_from_xrv(x)
    emb = clip_model.encode_image(img_clip)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    emb = emb.float()
    concept_emb = concept_emb.float()
    logits = emb @ concept_emb.t()
    return torch.softmax(logits / 0.07, dim=-1)

# -------------------- Tent --------------------
class TentBinaryTTA:
    def __init__(self, model, lr=1e-3, lambda_balance=1.0):
        self.model = copy.deepcopy(model).to(device)
        for p in self.model.parameters():
            p.requires_grad = False

        params = []
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.requires_grad = True
                    params.append(m.weight)
                if m.bias is not None:
                    m.bias.requires_grad = True
                    params.append(m.bias)
        self.params = params
        self.opt = torch.optim.Adam(params, lr=lr) if params else None
        self.lambda_balance = lambda_balance

    def __call__(self, x):
        if self.opt is None:
            with torch.no_grad():
                return self.model(x)
        self.model.train()
        self.opt.zero_grad()
        logits = self.model(x)
        p = torch.sigmoid(logits).clamp(1e-7, 1-1e-7)
        loss = (-p*torch.log(p) - (1-p)*torch.log(1-p)).mean()
        loss = loss + self.lambda_balance * (p.mean() - BALANCE_PRIOR)**2
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, 1.0)
        self.opt.step()
        return logits.detach()

# -------------------- CS-TTA --------------------
class ConceptHead(nn.Module):
    def __init__(self, feat_dim, num_concepts):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_concepts),
        )
    def forward(self, f):
        return torch.softmax(self.mlp(f), dim=-1)

class CSTTA(nn.Module):
    def __init__(self, base_model, clip_model, concept_emb, stable_weight_cpu,
                 lr=5e-4, lambda_ent=0.5, lambda_causal=1.0, lambda_anchor=2.0,
                 lambda_balance=0.5, anchor_conf_thresh=0.9):
        super().__init__()
        self.f  = copy.deepcopy(base_model).to(device)
        self.f0 = copy.deepcopy(base_model).to(device)
        self.f0.eval()
        for p in self.f0.parameters():
            p.requires_grad = False

        self.clip_model = clip_model
        self.concept_emb = concept_emb.float().to(device)
        self.stable_weight = stable_weight_cpu.to(device)

        with torch.no_grad():
            feat_dim = self.f.get_features(torch.randn(1,1,224,224, device=device)).shape[1]
        self.head = ConceptHead(feat_dim, num_concepts).to(device)

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
        for p in self.head.parameters():
            p.requires_grad = True
            params.append(p)

        self.params = params
        self.opt = torch.optim.Adam(params, lr=lr) if params else None

        self.lambda_ent = lambda_ent
        self.lambda_causal = lambda_causal
        self.lambda_anchor = lambda_anchor
        self.lambda_balance = lambda_balance
        self.anchor_conf_thresh = anchor_conf_thresh

    def entropy_loss(self, logits):
        p = torch.sigmoid(logits).clamp(1e-7, 1-1e-7)
        return (-p*torch.log(p) - (1-p)*torch.log(1-p)).mean()

    def causal_loss(self, c_hat, c_clip):
        w = self.stable_weight / (self.stable_weight.sum() + 1e-8)
        return ((c_hat - c_clip)**2 * w.unsqueeze(0)).sum(dim=1).mean()

    def anchor_loss(self, x, logits_tta):
        with torch.no_grad():
            p_src = torch.sigmoid(self.f0(x)).view(-1)
        p_tta = torch.sigmoid(logits_tta).view(-1)
        mask = (p_src >= self.anchor_conf_thresh) | (p_src <= 1-self.anchor_conf_thresh)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits_tta.device)
        return F.mse_loss(p_tta[mask], p_src[mask])

    def balance_loss(self, logits):
        return (torch.sigmoid(logits).mean() - BALANCE_PRIOR)**2

    def forward(self, x):
        if self.opt is None:
            with torch.no_grad():
                return self.f0(x)
        self.f.train()
        self.head.train()
        self.opt.zero_grad()
        logits = self.f(x)
        feats  = self.f.get_features(x)
        c_clip = get_clip_concept_scores(self.clip_model, self.concept_emb, x)
        c_hat  = self.head(feats)
        loss = (
            self.lambda_ent * self.entropy_loss(logits) +
            self.lambda_causal * self.causal_loss(c_hat, c_clip) +
            self.lambda_anchor * self.anchor_loss(x, logits) +
            self.lambda_balance * self.balance_loss(logits)
        )
        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            nn.utils.clip_grad_norm_(self.params, 1.0)
            self.opt.step()
        return logits.detach()

    # Grad-CAM용
    def forward_cam(self, x):
        self.f.eval()
        return self.f(x)

# -------------------- Eval --------------------
def tune_threshold(probs, labels):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    best_t, best_acc = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 99):
        pred = (probs > t).astype(int)
        acc = accuracy_score(labels, pred)
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t

def evaluate_model(model_or_tta, loader, is_tta, desc):
    all_logits, all_labels = [], []
    for x, y, _ in tqdm(loader, desc=desc, leave=False):
        x = x.to(device)
        y = y.to(device)
        if is_tta:
            logits = model_or_tta(x)
        else:
            with torch.no_grad():
                logits = model_or_tta(x)
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.cpu())
        del x, y, logits
        torch.cuda.empty_cache()

    logits = torch.cat(all_logits).view(-1)
    labels = torch.cat(all_labels).view(-1)
    probs = torch.sigmoid(logits).numpy()
    y = labels.numpy()

    auc = roc_auc_score(y, probs)
    thr = tune_threshold(probs, y)
    yhat = (probs > thr).astype(int)
    acc = accuracy_score(y, yhat)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, yhat, average="binary", zero_division=0
    )
    pos_acc = accuracy_score(y[y == 1], yhat[y == 1])
    neg_acc = accuracy_score(y[y == 0], yhat[y == 0])

    return {
        "threshold": float(thr),
        "accuracy": float(acc),
        "balanced_accuracy": float(0.5 * (pos_acc + neg_acc)),
        "auc": float(auc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "mean_prob": float(probs.mean()),
    }

# -------------------- Grad-CAM --------------------
def _find_last_conv(module: nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found")
    return last

class GradCAMHelper:
    def __init__(self, wrapper_model: nn.Module):
        self.model = wrapper_model
        features = self.model.backbone.features
        self.target = _find_last_conv(features)
        self.activations = None
        self.gradients = None

        def fwd_hook(_, __, out): self.activations = out
        def bwd_hook(_, gin, gout): self.gradients = gout[0]

        self._fh = self.target.register_forward_hook(fwd_hook)
        self._bh = self.target.register_backward_hook(bwd_hook)

    def remove(self):
        self._fh.remove()
        self._bh.remove()

    def cam(self, x):
        self.model.zero_grad()
        self.model.eval()
        logits = self.model(x)
        score = logits.sum()
        score.backward(retain_graph=False)

        A = self.activations
        dA = self.gradients
        w = dA.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * A).sum(dim=1))[0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = F.interpolate(
            cam[None, None], size=x.shape[2:], mode="bilinear", align_corners=False
        )
        return cam.squeeze().detach().cpu().numpy()

def overlay(gray_1ch, cam_1ch):
    base = gray_1ch.copy()
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)
    rgb = cv2.cvtColor((base * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    heat = cv2.applyColorMap((cam_1ch * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(rgb, 0.4, heat, 0.6, 0)

def vis4_and_save(base_img, cam_no, cam_te, cam_cs, save_path: Path):
    base = (base_img - base_img.min()) / (base_img.max() - base_img.min() + 1e-8)
    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    ax[0].imshow(base, cmap="gray")
    ax[0].set_title("Original")
    for i, (cam, title) in enumerate(
        [(cam_no, "No TTA"), (cam_te, "Tent"), (cam_cs, "CS-TTA")], start=1
    ):
        ax[i].imshow(base, cmap="gray")
        ax[i].imshow(cam, cmap="jet", alpha=0.4)
        ax[i].set_title(title)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

# -------------------- Per-image lists --------------------
def bce_per_image(prob, y):
    prob = np.clip(prob, 1e-7, 1 - 1e-7)
    return -(y * np.log(prob) + (1 - y) * np.log(1 - prob))

def collect_per_image_probs(model_name, clip_tag,
                            wrapper_no, wrapper_tent, wrapper_cstta,
                            loader):
    names, ys = [], []
    p_no, p_te, p_cs = [], [], []

    with torch.no_grad():
        for x, y, name in tqdm(loader, desc="Per-image probs", leave=False):
            x = x.to(device)
            p0 = torch.sigmoid(wrapper_no(x)).item()
            p1 = torch.sigmoid(wrapper_tent(x)).item()
            p2 = torch.sigmoid(wrapper_cstta(x)).item()
            names.append(name[0])
            ys.append(int(y.item()))
            p_no.append(float(p0))
            p_te.append(float(p1))
            p_cs.append(float(p2))
            del x

    loss_no = [bce_per_image(p, y) for p, y in zip(p_no, ys)]
    loss_te = [bce_per_image(p, y) for p, y in zip(p_te, ys)]
    loss_cs = [bce_per_image(p, y) for p, y in zip(p_cs, ys)]

    # cond1: CS-TTA가 No/Tent 둘 다보다 더 좋을 때
    cond1_idx = [
        i for i in range(len(names))
        if (loss_cs[i] < loss_no[i] and loss_cs[i] < loss_te[i])
    ]
    # cond2: Tent > No, 그리고 CS-TTA가 Tent보다 더 좋을 때
    cond2_idx = [
        i for i in range(len(names))
        if (loss_te[i] < loss_no[i] and loss_cs[i] < loss_te[i])
    ]

    def dump(idx_list, tag):
        rows = []
        for i in idx_list:
            rows.append({
                "filename": names[i],
                "label": ys[i],
                "prob_no_tta": p_no[i],
                "prob_tent": p_te[i],
                "prob_cs_tta": p_cs[i],
                "loss_no_tta": loss_no[i],
                "loss_tent": loss_te[i],
                "loss_cs_tta": loss_cs[i],
            })

        prefix = f"{TASK_NAME}__{model_name}__{clip_tag}"
        csv_path  = LISTS_ROOT / f"{prefix}__{tag}.csv"
        json_path = LISTS_ROOT / f"{prefix}__{tag}.json"

        if rows:
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=list(rows[0].keys()),
                )
                w.writeheader()
                w.writerows(rows)
        else:
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "filename","label","prob_no_tta","prob_tent","prob_cs_tta",
                        "loss_no_tta","loss_tent","loss_cs_tta"
                    ],
                )
                w.writeheader()

        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2)

        print(f"[LIST][{TASK_NAME}] {tag}: {len(rows)} images -> {csv_path}")

    dump(cond1_idx, "cond1_CS_better_than_both")
    dump(cond2_idx, "cond2_Tent_better_than_No_then_CS_best")

    return names, ys, p_no, p_te, p_cs, loss_no, loss_te, loss_cs

# -------------------- Main --------------------
def main():
    all_results = {}

    stable_w = build_stable_weight()

    print("\n" + "#" * 70)
    print(f"TASK: {TASK_NAME}")
    print("#" * 70)

    # Dataset & loaders
    test_ds = TestWithLabelsDataset(
        TEST_DIR,
        TEST_NORMAL_LABELDIR,
        TEST_POS_LABELDIR,
    )
    base_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False,
        num_workers=4, pin_memory=True
    )
    tta_loader = DataLoader(
        test_ds, batch_size=8, shuffle=True,
        num_workers=4, pin_memory=True
    )
    eval1_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True
    )

    task_results = {}

    for model_name, clip_tag in COMBOS:
        print(f"\n=== [{TASK_NAME}] COMBO: {model_name} + {clip_tag} ===")

        base = xrv.models.DenseNet(weights=PRETRAINED_MODELS[model_name]).to(device)
        if PATHOLOGY_NAME not in base.pathologies:
            print(f"  [Skip] {PATHOLOGY_NAME} not found in pathologies for {model_name}")
            del base
            torch.cuda.empty_cache()
            continue

        patho_idx = base.pathologies.index(PATHOLOGY_NAME)
        base_model = BinaryClassifier(base, patho_idx).to(device).eval()

        # CLIP
        clip_model, concept_emb = load_openai_clip(clip_tag)

        # No-TTA
        res_no = evaluate_model(
            base_model, base_loader, is_tta=False,
            desc=f"{TASK_NAME}-{model_name}-{clip_tag}-NoTTA"
        )
        thr_star = res_no["threshold"]
        print(f"  No-TTA: acc={res_no['accuracy']:.3f}, auc={res_no['auc']:.3f}, thr*={thr_star:.2f}")

        # Tent
        tent = TentBinaryTTA(base_model, lr=1e-3, lambda_balance=1.0)
        res_te = evaluate_model(
            tent, tta_loader, is_tta=True,
            desc=f"{TASK_NAME}-{model_name}-{clip_tag}-Tent"
        )
        print(f"  Tent  : acc={res_te['accuracy']:.3f}, auc={res_te['auc']:.3f}")
        tent_wrapper = tent.model.to(device).eval()

        # CS-TTA
        cstta = CSTTA(
            base_model, clip_model, concept_emb, stable_w,
            lr=5e-4, lambda_ent=0.5, lambda_causal=1.0,
            lambda_anchor=2.0, lambda_balance=0.5
        )
        res_cs = evaluate_model(
            cstta, tta_loader, is_tta=True,
            desc=f"{TASK_NAME}-{model_name}-{clip_tag}-CS-TTA"
        )
        print(f"  CS-TTA: acc={res_cs['accuracy']:.3f}, auc={res_cs['auc']:.3f}")
        cstta_wrapper = cstta.f.to(device).eval()

        # Per-image lists
        names, ys, p_no, p_te, p_cs, l_no, l_te, l_cs = collect_per_image_probs(
            model_name, clip_tag,
            base_model, tent_wrapper, cstta_wrapper, eval1_loader
        )

        # Grad-CAM dirs
        base_dir = GRADCAM_ROOT / TASK_NAME / f"{model_name}__{clip_tag}"
        dir_four = base_dir / "FourPanel"
        dir_no   = base_dir / "NoTTA"
        dir_te   = base_dir / "Tent"
        dir_cs   = base_dir / "CSTTA"
        for d in [dir_four, dir_no, dir_te, dir_cs]:
            d.mkdir(parents=True, exist_ok=True)

        cam_no = GradCAMHelper(base_model)
        cam_te = GradCAMHelper(tent_wrapper)
        cam_cs = GradCAMHelper(cstta_wrapper)

        saved = 0
        for idx, (x, y, fname) in enumerate(
            tqdm(eval1_loader,
                 desc=f"GradCAM [{TASK_NAME}] {model_name}-{clip_tag}",
                 leave=False)
        ):
            x = x.to(device)
            yv = int(y.item())
            base_img = x[0, 0].detach().cpu().numpy()

            cam_map_no = cam_no.cam(x)
            cam_map_te = cam_te.cam(x)
            with torch.enable_grad():
                cam_map_cs = cam_cs.cam(x)

            ov_no = overlay(base_img, cam_map_no)
            ov_te = overlay(base_img, cam_map_te)
            ov_cs = overlay(base_img, cam_map_cs)

            name = fname[0]
            info = (
                f"y{yv}_"
                f"pNO{p_no[idx]:.3f}_pTE{p_te[idx]:.3f}_pCS{p_cs[idx]:.3f}_"
                f"lNO{l_no[idx]:.3f}_lTE{l_te[idx]:.3f}_lCS{l_cs[idx]:.3f}"
            )

            four_path = dir_four / f"{Path(name).stem}__{info}.png"
            no_path   = dir_no   / f"{Path(name).stem}__{info}.png"
            te_path   = dir_te   / f"{Path(name).stem}__{info}.png"
            cs_path   = dir_cs   / f"{Path(name).stem}__{info}.png"

            vis4_and_save(base_img, cam_map_no, cam_map_te, cam_map_cs, four_path)
            cv2.imwrite(str(no_path), ov_no)
            cv2.imwrite(str(te_path), ov_te)
            cv2.imwrite(str(cs_path), ov_cs)

            saved += 1
            if MAX_GRADCAM_SAMPLES is not None and saved >= MAX_GRADCAM_SAMPLES:
                break
            del x

        cam_no.remove()
        cam_te.remove()
        cam_cs.remove()

        task_results[f"{model_name}__{clip_tag}"] = {
            "No TTA": res_no,
            "Tent": res_te,
            "CS-TTA": res_cs,
        }

        del base_model, base, clip_model, tent, cstta, tent_wrapper, cstta_wrapper
        torch.cuda.empty_cache()

    all_results = {TASK_NAME: task_results}

    with open(RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n[DONE]")
    print(f"Results JSON : {RESULTS_JSON}")
    print(f"GradCAM dir  : {GRADCAM_ROOT}")
    print(f"Lists dir    : {LISTS_ROOT}")

if __name__ == "__main__":
    main()
