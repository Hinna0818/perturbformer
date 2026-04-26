## Perturbformer-V3: Drug_SMILES_Embedding + Latent perturbation encoder–decoder model
import os
import re
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


## 1. settings
BASE_DIR = "/biostack/home/henan/perturb-seq"

META_FILE = os.path.join(BASE_DIR, "processed", "meta_l1000_test_5000.tsv")
Y_FILE = os.path.join(BASE_DIR, "processed", "Y_l1000_test_5000_landmark.tsv")
PERT_FILE = os.path.join(
    BASE_DIR,
    "l1000_data",
    "GSE70138_Broad_LINCS_pert_info.txt.gz"
)

MODEL_DIR = os.path.join(BASE_DIR, "latent")
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 64
EPOCHS = 200
LR = 1e-3

FP_DIM = 2048
FP_RADIUS = 2

DRUG_EMB_DIM = 128
CELL_EMB_DIM = 32
HIDDEN_DIM = 512
Z_DIM = 64

PATIENCE = 10
MIN_DELTA = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

print("=" * 80)
print("Device:", device)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("=" * 80)

## 2. tool function (extract dose/time data)
def extract_number(x):
    """
    convert '10.0 um' / '24 h' / '3.33 um' into numbers
    if fail, return to NaN.
    """
    if pd.isna(x):
        return np.nan
    x = str(x)
    match = re.search(r"[-+]?\d*\.?\d+", x)
    if match:
        return float(match.group())
    return np.nan

# convert smiles into morgan
def smiles_to_morgan(smiles, n_bits=2048, radius=2):
    """
    Convert a SMILES string into a Morgan fingerprint vector.

    Input:
        smiles: chemical structure string
    Output:
        arr: numpy array with shape [n_bits], dtype float32
    """
    if pd.isna(smiles):
        return None

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=n_bits
    )

    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr


## 3. load data
meta = pd.read_csv(META_FILE, sep="\t", index_col=0)
Y = pd.read_csv(Y_FILE, sep="\t", index_col=0)

print("meta shape:", meta.shape)
print("Y shape:", Y.shape)

common_ids = meta.index.intersection(Y.index)
meta = meta.loc[common_ids].copy()
Y = Y.loc[common_ids].copy()

print("aligned meta shape:", meta.shape)
print("aligned Y shape:", Y.shape)

required_cols = ["pert_id", "cell_id", "pert_idose", "pert_itime"]
for col in required_cols:
    if col not in meta.columns:
        raise ValueError(f"Missing column in meta: {col}")


## 4. load pert_info and build Morgan fingerprints
meta = meta.reset_index().rename(columns={"index": "sig_id"})

pert_info = pd.read_csv(PERT_FILE, sep="\t", low_memory=False)

print("pert_info shape:", pert_info.shape)
print("pert_info columns:", pert_info.columns.tolist())

required_pert_cols = ["pert_id", "canonical_smiles"]
for col in required_pert_cols:
    if col not in pert_info.columns:
        raise ValueError(f"Missing column in pert_info: {col}")

pert_info = pert_info[["pert_id", "canonical_smiles"]].drop_duplicates("pert_id")
pert_info["pert_id"] = pert_info["pert_id"].astype(str)
meta["pert_id"] = meta["pert_id"].astype(str)

meta = meta.merge(
    pert_info,
    on="pert_id",
    how="left"
)

print("after merging pert_info:", meta.shape)
print("missing SMILES:", meta["canonical_smiles"].isna().sum())

meta = meta.dropna(subset=["canonical_smiles"]).copy()

meta = meta.set_index("sig_id")
Y = Y.loc[meta.index].copy()

# cell embedding
cell_list = sorted(meta["cell_id"].astype(str).unique())
cell_to_idx = {cell: i for i, cell in enumerate(cell_list)}
meta["cell_idx"] = meta["cell_id"].astype(str).map(cell_to_idx)

num_cells = len(cell_list)
out_dim = Y.shape[1]

print("num_cells:", num_cells)
print("out_dim:", out_dim)
print("remaining samples after SMILES filter:", meta.shape[0])

# Morgan fingerprint
print("Generating Morgan fingerprints...")

fps = []
valid_ids = []

for sig_id, row in meta.iterrows():
    fp = smiles_to_morgan(
        row["canonical_smiles"],
        n_bits=FP_DIM,
        radius=FP_RADIUS
    )

    if fp is not None:
        fps.append(fp)
        valid_ids.append(sig_id)

drug_fp = np.vstack(fps).astype("float32")

meta = meta.loc[valid_ids].copy()
Y = Y.loc[valid_ids].copy()

print("drug_fp:", drug_fp.shape)
print("meta after valid fp:", meta.shape)
print("Y after valid fp:", Y.shape)

## 5. solve NaN
meta["dose_value"] = meta["pert_idose"].apply(extract_number)
meta["time_value"] = meta["pert_itime"].apply(extract_number)

meta["dose_value"] = meta["dose_value"].fillna(meta["dose_value"].median())
meta["time_value"] = meta["time_value"].fillna(meta["time_value"].median())

numeric = meta[["dose_value", "time_value"]].values.astype("float32")

scaler = StandardScaler()
numeric = scaler.fit_transform(numeric).astype("float32")

print("numeric example:")
print(meta[["pert_idose", "dose_value", "pert_itime", "time_value"]].head())

## 6. to numpy
cell_idx = meta["cell_idx"].values.astype("int64")
Y_np = Y.values.astype("float32")

print("drug_fp:", drug_fp.shape)
print("cell_idx:", cell_idx.shape)
print("numeric:", numeric.shape)
print("Y_np:", Y_np.shape)

## 7. Dataset
class L1000Dataset(Dataset):
    """
    PyTorch Dataset：
    告诉 DataLoader 每个样本怎么取。
    
    __len__ 返回样本数量。
    __getitem__ 返回第 idx 个样本。
    """
    def __init__(self, drug_fp, cell_idx, numeric, y):
        self.drug_fp = torch.tensor(drug_fp, dtype=torch.float32)
        self.cell_idx = torch.tensor(cell_idx, dtype=torch.long)
        self.numeric = torch.tensor(numeric, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "drug_fp": self.drug_fp[idx],
            "cell_idx": self.cell_idx[idx],
            "numeric": self.numeric[idx],
            "y": self.y[idx],
        }

dataset = L1000Dataset(drug_fp, cell_idx, numeric, Y_np)
n_total = len(dataset)

n_train = int(n_total * 0.7)
n_valid = int(n_total * 0.15)
n_test = n_total - n_train - n_valid

train_ds, valid_ds, test_ds = random_split(
    dataset,
    [n_train, n_valid, n_test],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print("train samples:", len(train_ds))
print("valid samples:", len(valid_ds))
print("test samples :", len(test_ds))

## 8. Model
class LatentPerturbationModel(nn.Module):
    """
    input:
    drug_fp: [batch, 2048]
    cell_idx: [batch]
    numeric:  [batch, 2]

    output:
    pred: [batch, 978]
    z: [batch, z_dim]
    """
    def __init__(self, fp_dim, num_cells, out_dim, drug_emb_dim=128, cell_emb_dim=32, hidden_dim=512, z_dim=64, ):
        super().__init__()
        
        #self.drug_emb = nn.Embedding(num_drugs, drug_emb_dim)
        self.drug_encoder = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, drug_emb_dim),
            nn.ReLU(),
        )
        self.cell_emb = nn.Embedding(num_cells, cell_emb_dim)
        input_dim = drug_emb_dim + cell_emb_dim + 2

        # self.mlp = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),

        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),

        #     nn.Linear(hidden_dim, out_dim),
        #     )


        # encoder: input -> latent z
        self.condition_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, z_dim)
        )

        # decoder: latent z -> delta expression
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim)
        )


    def forward(self, drug_fp, cell_idx, numeric):
        drug_vec = self.drug_encoder(drug_fp)
        cell_vec = self.cell_emb(cell_idx)
        x = torch.cat([drug_vec, cell_vec, numeric], dim=1)
        z = self.condition_encoder(x)
        pred = self.decoder(z)
        return pred, z

## set model
model = LatentPerturbationModel(
    fp_dim=FP_DIM,
    num_cells=num_cells,
    out_dim=out_dim,
    drug_emb_dim=DRUG_EMB_DIM,
    cell_emb_dim=CELL_EMB_DIM,
    hidden_dim=HIDDEN_DIM,
    z_dim = Z_DIM
).to(device)

## 9. loss function
mse_loss = nn.MSELoss()

def pearson_loss(pred, target, eps=1e-8):
    """
    对每个样本计算 pred 和 target 在 978 个基因上的 Pearson correlation。
    loss = 1 - mean(correlation)
    """
    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)

    numerator = (pred_centered * target_centered).sum(dim=1)
    denominator = torch.sqrt((pred_centered ** 2).sum(dim=1) + eps) * torch.sqrt(
        (target_centered ** 2).sum(dim=1) + eps
    )

    corr = numerator / denominator
    return 1 - corr.mean()


def combined_loss(pred, target):
    return mse_loss(pred, target) + 0.2 * pearson_loss(pred, target)

## 10. optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

## 11. evaluation
def evaluate_model(model, loader):
    model.eval()

    total_loss = 0.0
    preds = []
    trues = []

    with torch.no_grad():
        for batch in loader:
            drug = batch["drug_fp"].to(device)
            cell = batch["cell_idx"].to(device)
            num = batch["numeric"].to(device)
            y = batch["y"].to(device)

            pred, z = model(drug, cell, num)
            loss = combined_loss(pred, y)

            total_loss += loss.item()
            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())

    preds = np.vstack(preds)
    trues = np.vstack(trues)

    # 每个样本一个相关系数
    pearsons = []
    spearmans = []

    for i in range(preds.shape[0]):
        try:
            pearsons.append(pearsonr(preds[i], trues[i])[0])
            spearmans.append(spearmanr(preds[i], trues[i])[0])
        except Exception:
            pearsons.append(np.nan)
            spearmans.append(np.nan)

    rmse = np.sqrt(np.mean((preds - trues) ** 2))

    metrics = {
        "loss": total_loss / len(loader),
        "rmse": rmse,
        "pearson": np.nanmean(pearsons),
        "spearman": np.nanmean(spearmans),
    }

    return metrics

## 12. train
best_valid_loss = float("inf")
best_epoch = 0
epochs_no_improve = 0

best_model_path = os.path.join(MODEL_DIR, "latent_perturbation_best.pt")

for epoch in range(1, EPOCHS + 1):
    model.train()

    train_loss = 0.0

    for batch in train_loader:
        drug = batch["drug_fp"].to(device)
        cell = batch["cell_idx"].to(device)
        num = batch["numeric"].to(device)
        y = batch["y"].to(device)

        pred, z = model(drug, cell, num)

        loss = combined_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss = train_loss / len(train_loader)

    valid_metrics = evaluate_model(model, valid_loader)

    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Valid Loss: {valid_metrics['loss']:.4f} | "
        f"RMSE: {valid_metrics['rmse']:.4f} | "
        f"Pearson: {valid_metrics['pearson']:.4f} | "
        f"Spearman: {valid_metrics['spearman']:.4f}"
    )

    # 判断 valid loss 是否有明显改善
    if valid_metrics["loss"] < best_valid_loss - MIN_DELTA:
        best_valid_loss = valid_metrics["loss"]
        best_epoch = epoch
        epochs_no_improve = 0

        save_obj = {
            "model_state_dict": model.state_dict(),
            "cell_to_idx": cell_to_idx,
            "gene_names": Y.columns.tolist(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "fp_dim": FP_DIM,
            "fp_radius": FP_RADIUS,
            "config": {
                "fp_dim": FP_DIM,
                "drug_emb_dim": DRUG_EMB_DIM,
                "cell_emb_dim": CELL_EMB_DIM,
                "hidden_dim": HIDDEN_DIM,
                "z_dim": Z_DIM,
                "out_dim": out_dim,
                "num_cells": num_cells,
            },
        }

        torch.save(save_obj, best_model_path)
        print(f"Saved best model to: {best_model_path}")

    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve}/{PATIENCE} epochs.")

    # Early stopping
    if epochs_no_improve >= PATIENCE:
        print("=" * 80)
        print(f"Early stopping triggered at epoch {epoch}.")
        print(f"Best epoch: {best_epoch}")
        print(f"Best valid loss: {best_valid_loss:.4f}")
        print("=" * 80)
        break

print("=" * 80)
print("Training finished.")
print("Best epoch:", best_epoch)
print("Best valid loss:", best_valid_loss)
print("Best model:", best_model_path)
print("=" * 80)

## 13. test
print("=" * 80)
print("Evaluating best model on test set...")
print("=" * 80)

checkpoint = torch.load(best_model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

test_metrics = evaluate_model(model, test_loader)

print(
    f"Test Loss: {test_metrics['loss']:.4f} | "
    f"Test RMSE: {test_metrics['rmse']:.4f} | "
    f"Test Pearson: {test_metrics['pearson']:.4f} | "
    f"Test Spearman: {test_metrics['spearman']:.4f}"
)

print("=" * 80)
print("Final test evaluation finished.")
print("=" * 80)



