#!/usr/bin/env python3
"""
EA-TTA: Explainability-Aware Test-Time Adaptation
Medical Imaging - Binary Classification with Grad-CAM

Task: Enlarged Cardiomediastinum Detection
Source: CheXpert (DenseNet-121)
Target: MIMIC-CXR (unlabeled test-time)
Methods: No TTA, Tent, EA-TTA (concept-guided)

Outputs:
- Grad-CAM visualizations (4-panel + per-method)
- Per-image performance lists (CSV/JSON)
- Results JSON with metrics
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
import json, copy
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
import clip
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

# -------------------- Configuration --------------------
ROOT = Path("./data/medical_imaging")

TASK_NAME = "Enlarged_Cardiomediastinum"
PATHOLOGY_NAME = "Enlarged Cardiomediastinum"

# Dataset paths
TEST_DIR             = ROOT / "test"
TEST_NORMAL_LABELDIR = ROOT / "test_normal_labels"
TEST_POS_LABELDIR    = ROOT / "test_positive_labels"

# Output paths
GRADCAM_ROOT = ROOT / "outputs/gradcam"
RESULTS_JSON = ROOT / "outputs/results.json"
GRADCAM_ROOT.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda:0")

# Model combinations
COMBOS = [
    ("CheXpert", "CLIP-B32"),
    ("CheXpert", "CLIP-L14"),
]

PRETRAINED_MODELS = {
    "CheXpert": "densenet121-res224-chex",
}

CLIP_MODELS_CONFIG = {
    "CLIP-B32": {"name": "ViT-B/32"},
    "CLIP-L14": {"name": "ViT-L/14"},
}

# Concept definitions for medical imaging
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
MAX_GRADCAM_SAMPLES = None

# -------------------- Stable Weights --------------------
def build_stable_weight() -> torch.Tensor:
    """
    Build concept stability weights without pre-computed JSON.
    Default strategy:
    - Pathology/anatomy concepts: 1.5x (stable, causal)
    - Device/artifact concepts: 0.5x (spurious, domain-specific)
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
    """
    Test dataset with labels for evaluation.
    - test_dir: images
    - normal_label_dir: normal case filenames
    - pos_label_dir: positive case filenames
    """
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

        print(f"[INFO] Labeled test samples: {len(self.items)} (unlabeled: {missing})")
        ys = [lab for _, lab in self.items]
        pos = sum(ys)
        neg = len(ys) - pos
        print(f"[INFO] Label distribution: positive={pos}, normal={neg}")

    def __len__(self): 
        return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        img = np.array(Image.open(path)).astype(np.float32)
        # Normalize [0,255] -> [-1024,1024] for TorchXRayVision
        img = (img / 255.0) * 2048.0 - 1024.0
        x = torch.from_numpy(img[None, ...]).float()
        return x, torch.tensor(y, dtype=torch.long), path.name

# -------------------- Models --------------------
class BinaryClassifier(nn.Module):
    """Wrapper for binary classification using specific pathology index."""
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

# -------------------- CLIP Helpers --------------------
def load_openai_clip(clip_tag: str):
    """Load CLIP model and encode concept text embeddings."""
    model, _ = clip.load(CLIP_MODELS_CONFIG[clip_tag]["name"], device=device)
    model.float()
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
    """Convert TorchXRayVision format to CLIP input format."""
    img = (x + 1024.0) / 2048.0
    img = img.clamp(0.0, 1.0).repeat(1, 3, 1, 1)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                        device=img.device).view(1,3,1,1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                        device=img.device).view(1,3,1,1)
    return (img - mean) / std

@torch.no_grad()
def get_clip_concept_scores(clip_model, concept_emb, x):
    """Extract concept activation scores using CLIP."""
    img_clip = prepare_clip_inputs_from_xrv(x)
    emb = clip_model.encode_image(img_clip)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    emb = emb.float()
    concept_emb = concept_emb.float()
    logits = emb @ concept_emb.t()
    return torch.softmax(logits / 0.07, dim=-1)

# -------------------- Tent TTA --------------------
class TentBinaryTTA:
    """
    Tent: Entropy minimization with BatchNorm updates.
    Reference: Wang et al., ICLR 2021
    """
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

# -------------------- EA-TTA --------------------
class ConceptHead(nn.Module):
    """MLP for predicting concept scores from features."""
    def __init__(self, feat_dim, num_concepts):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_concepts),
        )
    def forward(self, f):
        return torch.softmax(self.mlp(f), dim=-1)

class EATTA(nn.Module):
    """
    EA-TTA: Explainability-Aware Test-Time Adaptation.
    Combines entropy minimization with concept-guided stability objectives.
    """
    def __init__(self, base_model, clip_model, concept_emb, stable_weight_cpu,
                 lr=5e-4, lambda_ent=0.5, lambda_suppress=1.0, lambda_anchor=2.0,
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
        self.lambda_suppress = lambda_suppress
        self.lambda_anchor = lambda_anchor
        self.lambda_balance = lambda_balance
        self.anchor_conf_thresh = anchor_conf_thresh

    def entropy_loss(self, logits):
        p = torch.sigmoid(logits).clamp(1e-7, 1-1e-7)
        return (-p*torch.log(p) - (1-p)*torch.log(1-p)).mean()

    def suppress_loss(self, c_hat, c_clip):
        """
        Spurious-suppressing concept alignment loss (L_suppress in the paper).
        Align predicted concepts with CLIP concepts, weighted by stability.
        """
        w = self.stable_weight / (self.stable_weight.sum() + 1e-8)
        return ((c_hat - c_clip)**2 * w.unsqueeze(0)).sum(dim=1).mean()


    def anchor_loss(self, x, logits_tta):
        """Anchor to source model predictions for high-confidence samples."""
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
            self.lambda_suppress * self.suppress_loss(c_hat, c_clip) +
            self.lambda_anchor * self.anchor_loss(x, logits) +
            self.lambda_balance * self.balance_loss(logits)
        )
        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            nn.utils.clip_grad_norm_(self.params, 1.0)
            self.opt.step()
        return logits.detach()

# -------------------- Evaluation --------------------
def tune_threshold(probs, labels):
    """Find optimal classification threshold."""
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
    """Evaluate model performance."""
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
    """Find last convolutional layer for Grad-CAM."""
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found")
    return last

class GradCAMHelper:
    """Grad-CAM visualization helper."""
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
    """Overlay heatmap on grayscale image."""
    base = gray_1ch.copy()
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)
    rgb = cv2.cvtColor((base * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    heat = cv2.applyColorMap((cam_1ch * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(rgb, 0.4, heat, 0.6, 0)

def vis4_and_save(base_img, cam_no, cam_te, cam_ea, save_path: Path):
    """Create 4-panel visualization and save."""
    base = (base_img - base_img.min()) / (base_img.max() - base_img.min() + 1e-8)
    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    ax[0].imshow(base, cmap="gray")
    ax[0].set_title("Original")
    for i, (cam, title) in enumerate(
        [(cam_no, "No TTA"), (cam_te, "Tent"), (cam_ea, "EA-TTA")], start=1
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

# -------------------- Main --------------------
def main():
    all_results = {}

    stable_w = build_stable_weight()

    print("\n" + "#" * 70)
    print(f"TASK: {TASK_NAME}")
    print("#" * 70)

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
            print(f"  [Skip] {PATHOLOGY_NAME} not found")
            del base
            torch.cuda.empty_cache()
            continue

        patho_idx = base.pathologies.index(PATHOLOGY_NAME)
        base_model = BinaryClassifier(base, patho_idx).to(device).eval()

        clip_model, concept_emb = load_openai_clip(clip_tag)

        # No-TTA
        res_no = evaluate_model(
            base_model, base_loader, is_tta=False,
            desc=f"{TASK_NAME}-{model_name}-{clip_tag}-NoTTA"
        )
        print(f"  No-TTA: acc={res_no['accuracy']:.3f}, auc={res_no['auc']:.3f}")

        # Tent
        tent = TentBinaryTTA(base_model, lr=1e-3, lambda_balance=1.0)
        res_te = evaluate_model(
            tent, tta_loader, is_tta=True,
            desc=f"{TASK_NAME}-{model_name}-{clip_tag}-Tent"
        )
        print(f"  Tent  : acc={res_te['accuracy']:.3f}, auc={res_te['auc']:.3f}")
        tent_wrapper = tent.model.to(device).eval()

        # EA-TTA
        eatta = EATTA(
            base_model, clip_model, concept_emb, stable_w,
            lr=5e-4, lambda_ent=0.5, lambda_suppress=1.0,
            lambda_anchor=2.0, lambda_balance=0.5
        )
        res_ea = evaluate_model(
            eatta, tta_loader, is_tta=True,
            desc=f"{TASK_NAME}-{model_name}-{clip_tag}-EATTA"
        )
        print(f"  EA-TTA: acc={res_ea['accuracy']:.3f}, auc={res_ea['auc']:.3f}")
        eatta_wrapper = eatta.f.to(device).eval()

        # Grad-CAM directories
        base_dir = GRADCAM_ROOT / TASK_NAME / f"{model_name}__{clip_tag}"
        dir_four = base_dir / "FourPanel"
        dir_no   = base_dir / "NoTTA"
        dir_te   = base_dir / "Tent"
        dir_ea   = base_dir / "EATTA"
        for d in [dir_four, dir_no, dir_te, dir_ea]:
            d.mkdir(parents=True, exist_ok=True)

        cam_no = GradCAMHelper(base_model)
        cam_te = GradCAMHelper(tent_wrapper)
        cam_ea = GradCAMHelper(eatta_wrapper)

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
                cam_map_ea = cam_ea.cam(x)

            ov_no = overlay(base_img, cam_map_no)
            ov_te = overlay(base_img, cam_map_te)
            ov_ea = overlay(base_img, cam_map_ea)

            name = fname[0]
            stem = Path(name).stem

            four_path = dir_four / f"{stem}.png"
            no_path   = dir_no   / f"{stem}.png"
            te_path   = dir_te   / f"{stem}.png"
            ea_path   = dir_ea   / f"{stem}.png"

            vis4_and_save(base_img, cam_map_no, cam_map_te, cam_map_ea, four_path)
            cv2.imwrite(str(no_path), ov_no)
            cv2.imwrite(str(te_path), ov_te)
            cv2.imwrite(str(ea_path), ov_ea)

            saved += 1
            if MAX_GRADCAM_SAMPLES is not None and saved >= MAX_GRADCAM_SAMPLES:
                break
            del x

        cam_no.remove()
        cam_te.remove()
        cam_ea.remove()

        task_results[f"{model_name}__{clip_tag}"] = {
            "No TTA": res_no,
            "Tent": res_te,
            "EA-TTA": res_ea,
        }

        del base_model, base, clip_model, tent, eatta, tent_wrapper, eatta_wrapper
        torch.cuda.empty_cache()

    all_results = {TASK_NAME: task_results}

    with open(RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n[DONE]")
    print(f"Results JSON : {RESULTS_JSON}")
    print(f"GradCAM dir  : {GRADCAM_ROOT}")

if __name__ == "__main__":
    main()
