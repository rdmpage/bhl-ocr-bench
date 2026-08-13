"""Decisive test: is the blank-page rubric leak a v4 regression, or did NHM clean it away?

Run mistral-ocr-3 (== mistral-ocr-2512, which is what they scored) over the SAME sparse pages
my v4-1 run already covered, and compare raw output volume.
"""
import pathlib, re, sys
import pandas as pd

BENCH = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH/"producers"))
import mistral_ocr as M

gt = pd.read_parquet(BENCH/"benchmark/gt/train.parquet")
v4 = pd.read_parquet(BENCH/"runs/mistral-ocr-4-1/run.parquet").set_index("PageID")
snap = pathlib.Path.home()/".cache/huggingface/hub/datasets--finebooks--bhl-impact-gt/snapshots/b7bda5fac0471d6d2237360abc799c6d13559465"
meta = pd.read_parquet(snap/"metadata.parquet").set_index("PageID")

# The 50 sparse pages where v4-1 hallucinated hardest — the clearest signal.
sparse = gt[gt.sample_stratum == "sparse_blank"].copy()
sparse["v4_len"] = sparse.PageID.map(lambda p: len(v4.loc[p, "markdown"]))
sel = sparse.nlargest(50, "v4_len")

key = M._read_api_key(None)
M._init_worker(key, "mistral-ocr-3", 120.0, 4, "https://api.mistral.ai/v1")

RUBRIC = re.compile(r"Ground Truth image|UNDERSCORE & LINE RULES|According to Rule", re.I)
rows = []
for _, r in sel.iterrows():
    img = (snap/meta.loc[r.PageID, "file_name"]).read_bytes()
    text, _ = M.ocr_page({"PageID": r.PageID, "image_bytes": img})
    rows.append({"PageID": r.PageID, "v4_len": r.v4_len, "v3_len": len(text),
                 "v4_rubric": bool(RUBRIC.search(v4.loc[r.PageID, "markdown"])),
                 "v3_rubric": bool(RUBRIC.search(text))})
d = pd.DataFrame(rows)
print(f"50 hardest sparse pages (by v4 output length)")
print(f"  v4-1 total chars : {d.v4_len.sum():>9,}   rubric-leak pages: {d.v4_rubric.sum()}/50")
print(f"  v3   total chars : {d.v3_len.sum():>9,}   rubric-leak pages: {d.v3_rubric.sum()}/50")
print(f"  v3 emitted nothing on {(d.v3_len==0).sum()}/50;  v4 on {(d.v4_len==0).sum()}/50")
print(f"  median chars  v4={int(d.v4_len.median())}  v3={int(d.v3_len.median())}")
d.to_csv("/private/tmp/claude-501/-Users-rpage-Development-bhl-ocr-eval/aefc54db-3295-46a4-b8ac-bf77aaf38098/scratchpad/v3_vs_v4.csv", index=False)
