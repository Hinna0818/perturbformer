import os
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse

# =========================
# 1. 路径设置
# =========================

BASE_DIR = "/biostack/home/henan/perturb-seq"
DATA_DIR = os.path.join(BASE_DIR, "l1000_data")
OUT_DIR = os.path.join(BASE_DIR, "processed")

os.makedirs(OUT_DIR, exist_ok=True)

gctx_file = os.path.join(
    DATA_DIR,
    "GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx"
)

sig_file = os.path.join(
    DATA_DIR,
    "GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz"
)

gene_file = os.path.join(
    DATA_DIR,
    "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz"
)

# 测试阶段先读 1000 条 signature
N_TEST = 1000

print("=" * 80)
print("Input files")
print("gctx_file:", gctx_file)
print("sig_file :", sig_file)
print("gene_file:", gene_file)
print("=" * 80)

for f in [gctx_file, sig_file, gene_file]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"File not found: {f}")

# =========================
# 2. 读取 metadata
# =========================

sig_info = pd.read_csv(sig_file, sep="\t", low_memory=False)
gene_info = pd.read_csv(gene_file, sep="\t", low_memory=False)

print("sig_info shape :", sig_info.shape)
print("gene_info shape:", gene_info.shape)
print("sig columns    :", sig_info.columns.tolist())
print("gene columns   :", gene_info.columns.tolist())

# =========================
# 3. 筛选小分子药物扰动
# =========================

if "pert_type" not in sig_info.columns:
    raise ValueError("sig_info does not contain column: pert_type")

sig_use = sig_info[sig_info["pert_type"] == "trt_cp"].copy()

print("trt_cp signatures:", sig_use.shape[0])

# 你的这个版本没有 is_gold 字段，所以这里只在存在时才筛
if "is_gold" in sig_use.columns:
    sig_use = sig_use[sig_use["is_gold"].astype(str).isin(["1", "True", "true"])]
    print("after is_gold filter:", sig_use.shape[0])

if "sig_id" not in sig_use.columns:
    raise ValueError("sig_info does not contain column: sig_id")

sig_ids = sig_use["sig_id"].dropna().astype(str).head(N_TEST).tolist()

print("selected signatures:", len(sig_ids))
print("first 3 sig_ids:", sig_ids[:3])

# =========================
# 4. 筛选 978 landmark genes
# =========================

required_gene_cols = ["pr_gene_id", "pr_is_lm"]
for col in required_gene_cols:
    if col not in gene_info.columns:
        raise ValueError(f"gene_info does not contain column: {col}")

# pr_is_lm 有时是 1，有时可能被读成字符串
lm_mask = gene_info["pr_is_lm"].astype(str).isin(["1", "True", "true"])

lm_gene_info = gene_info.loc[lm_mask].copy()

# cmapPy 读取 gctx 时，rid 通常需要字符串 ID
lm_genes = lm_gene_info["pr_gene_id"].astype(str).tolist()

print("selected landmark genes:", len(lm_genes))
print("first 5 landmark gene ids:", lm_genes[:5])

if len(lm_genes) == 0:
    raise ValueError("No landmark genes selected. Please check pr_is_lm column.")

# =========================
# 5. 读取 GCTX 子矩阵
# =========================

print("=" * 80)
print("Reading GCTX subset...")
print("This may take a while.")
print("=" * 80)

try:
    gct = parse(gctx_file, rid=lm_genes, cid=sig_ids)
except Exception as e:
    print("Failed when reading with string gene ids.")
    print("Trying integer gene ids...")
    lm_genes_int = lm_gene_info["pr_gene_id"].astype(int).tolist()
    gct = parse(gctx_file, rid=lm_genes_int, cid=sig_ids)

expr = gct.data_df

print("expr shape:", expr.shape)
print("expr index example:", list(expr.index[:5]))
print("expr columns example:", list(expr.columns[:3]))

# expr 是 genes × signatures
# 模型训练通常需要 samples × genes
Y = expr.T
Y.index.name = "sig_id"

print("Y shape:", Y.shape)

# =========================
# 6. metadata 与 Y 对齐
# =========================

sig_info_indexed = sig_info.copy()
sig_info_indexed["sig_id"] = sig_info_indexed["sig_id"].astype(str)
sig_info_indexed = sig_info_indexed.set_index("sig_id")

meta = sig_info_indexed.loc[Y.index].copy()

print("meta shape:", meta.shape)
print("meta head:")
print(meta.head())

# =========================
# 7. 给 Y 的列名从 gene_id 改成 gene_symbol，方便后续看
# =========================

id_to_symbol = (
    gene_info[["pr_gene_id", "pr_gene_symbol"]]
    .dropna()
    .assign(pr_gene_id=lambda x: x["pr_gene_id"].astype(str))
    .set_index("pr_gene_id")["pr_gene_symbol"]
    .to_dict()
)

# Y.columns 可能是字符串或整数，统一转字符串匹配
new_cols = []
for col in Y.columns:
    col_str = str(col)
    symbol = id_to_symbol.get(col_str, col_str)
    new_cols.append(symbol)

Y.columns = new_cols

print("Y columns example after mapping:", list(Y.columns[:10]))

# =========================
# 8. 保存结果
# =========================

y_out = os.path.join(OUT_DIR, f"Y_l1000_test_{N_TEST}_landmark.tsv")
meta_out = os.path.join(OUT_DIR, f"meta_l1000_test_{N_TEST}.tsv")

Y.to_csv(y_out, sep="\t")
meta.to_csv(meta_out, sep="\t")

print("=" * 80)
print("Saved:")
print(y_out)
print(meta_out)
print("=" * 80)

print("Done.")