"""Task 1: inference NER dengan model final JobBERT-CRF (models/ner_jobbert_crf, dilatih di notebooks/task1_ner_crf.ipynb).
Jika folder model belum ada, model diunduh otomatis dari Hugging Face Hub (HF_REPO).

Pakai sebagai modul:
    from t1_ner_predict import load, extract
    model, tk = load()
    extract(["teks CV atau JD ..."], model, tk)   # -> [{"SKILL": [...], ..., "experience_years": float|None}]

Atau buat ulang prediksi untuk data tim (data/ner_pred_*.parquet), lalu jalankan scripts/t1_export_outputs.py:
    python scripts/t1_ner_predict.py --export
"""
import os, re, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR, DATA = ROOT / "models" / "ner_jobbert_crf", ROOT / "data"
HF_REPO = os.environ.get("T1_NER_REPO", "WwzFwz/jobbert-crf-cv-jd-ner")   # repo model di Hugging Face Hub
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ENTITIES = ["SKILL", "JOB_TITLE", "EDUCATION", "EXPERIENCE"]
TAGS = ["O"] + [f"{p}-{L}" for L in ENTITIES for p in "BI"]; NT = len(TAGS)
TOK = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|\w+(?:[.+#'-]\w+)*[+#]*|[^\w\s]")
YEAR, MAX_LEN, NEG = r"(?:19|20)\d\d", 512, -1e9

ALLOW = np.ones((NT, NT), bool)                                     # ALLOW[i, j]: tag i boleh diikuti tag j
for j, t in enumerate(TAGS):
    if t.startswith("I-"): ALLOW[:, j] = [s != "O" and s[2:] == t[2:] for s in TAGS]
START_OK = np.array([not t.startswith("I-") for t in TAGS])


class BertCRF(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = AutoModel.from_config(config, add_pooling_layer=False)
        self.drop, self.emit = torch.nn.Dropout(0.1), torch.nn.Linear(config.hidden_size, NT)
        self.trans = torch.nn.Parameter(torch.zeros(NT, NT))
        self.start, self.end = torch.nn.Parameter(torch.zeros(NT)), torch.nn.Parameter(torch.zeros(NT))

    def emissions(self, ids, att):
        return self.emit(self.drop(self.bert(input_ids=ids, attention_mask=att).last_hidden_state))


def load(model_dir=MODEL_DIR, repo_id=HF_REPO):
    """Muat model dari models/ner_jobbert_crf; jika belum ada, unduh dari Hugging Face Hub (repo publik, tanpa login)."""
    model_dir = Path(model_dir)
    if not (model_dir / "model.safetensors").exists() and not (model_dir / "model.pt").exists():
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id, local_dir=model_dir, allow_patterns=["*.json", "*.txt", "*.safetensors"])
    model = BertCRF(AutoConfig.from_pretrained(model_dir))
    if (model_dir / "model.safetensors").exists():
        from safetensors.torch import load_file
        state = load_file(model_dir / "model.safetensors")
    else:
        state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(DEV).eval(), AutoTokenizer.from_pretrained(model_dir)


def chunk_bounds(n_sub, budget=MAX_LEN - 2):
    out, start, tot = [], 0, 0
    for i, c in enumerate(n_sub):
        if tot + c > budget and i > start: out.append((start, i)); start, tot = i, 0
        tot += c
    out.append((start, len(n_sub)))
    return out


def viterbi(em, trans, start, end):
    score, back = start + em[0], []
    for t in range(1, len(em)):
        cand = score[:, None] + trans
        back.append(cand.argmax(0)); score = cand.max(0) + em[t]
    best = [int((score + end).argmax())]
    for b in reversed(back): best.append(int(b[best[-1]]))
    return best[::-1]


@torch.no_grad()
def predict_tags(model, tk, token_lists, bs=16):
    tr, st, en = (x.detach().float().cpu().numpy() for x in (model.trans, model.start, model.end))
    tr, st = np.where(ALLOW, tr, NEG), np.where(START_OK, st, NEG)
    jobs = []
    for d, toks in enumerate(token_lists):
        if not toks: continue
        wid = tk(toks, is_split_into_words=True, add_special_tokens=False).word_ids()
        n_sub = np.bincount([w for w in wid if w is not None], minlength=len(toks)).tolist()
        for s, e in chunk_bounds(n_sub):
            if s < e:
                enc = tk(toks[s:e], is_split_into_words=True, truncation=True, max_length=MAX_LEN)
                jobs.append((d, s, enc["input_ids"], enc.word_ids()))
    out = [["O"] * len(t) for t in token_lists]
    order = sorted(range(len(jobs)), key=lambda k: len(jobs[k][2]))
    for i in range(0, len(order), bs):
        b = order[i:i + bs]; L = max(len(jobs[k][2]) for k in b)
        ids = torch.full((len(b), L), tk.pad_token_id); att = torch.zeros((len(b), L), dtype=torch.long)
        for r, k in enumerate(b): n = len(jobs[k][2]); ids[r, :n] = torch.tensor(jobs[k][2]); att[r, :n] = 1
        with torch.autocast(DEV, enabled=DEV == "cuda"):
            em = model.emissions(ids.to(DEV), att.to(DEV)).float().cpu().numpy()
        for r, k in enumerate(b):
            d, s, _, wid = jobs[k]; pos, prev = [], None
            for p, w in enumerate(wid):
                if w is not None and w != prev: pos.append(p)
                prev = w
            for j, t in enumerate(viterbi(em[r][pos], tr, st, en)): out[d][s + j] = TAGS[t]
    return out


def bio_spans(tags):
    out, cur = [], None
    for i, t in enumerate(list(tags) + ["O"]):
        if cur and not (t.startswith("I-") and t[2:] == cur[0]): out.append((cur[0], cur[1], i)); cur = None
        if t.startswith("B-") or (t.startswith("I-") and cur is None): cur = (t[2:], i)
    return out


def experience_years(spans, this_year=2026):
    """Maksimum dari durasi eksplisit ("5 years") dan total rentang tahun yang digabung ("2015 - 2018")."""
    dur, iv = [], []
    for s in spans:
        for n, unit in re.findall(r"(\d{1,2})\s*\+?\s*(years?|yrs?|months?)", s):
            dur.append(int(n) / (12 if unit.startswith("month") else 1))
        ys = [int(y) for y in re.findall(rf"\b{YEAR}\b", s)]
        if re.search(r"present|current|now|till|date", s): ys.append(this_year)
        if len(ys) >= 2 and 0 <= max(ys) - min(ys) <= 50: iv.append((min(ys), max(ys)))
    tot, cur = 0, None
    for a, b in sorted(iv):
        if cur and a <= cur[1]: cur = (cur[0], max(cur[1], b))
        else:
            if cur: tot += cur[1] - cur[0]
            cur = (a, b)
    if cur: tot += cur[1] - cur[0]
    vals = [v for v in [max(dur, default=0), tot] if v > 0]
    return float(max(vals)) if vals else None


def extract(texts, model, tk):
    toks = [TOK.findall(t or "") for t in texts]
    out = []
    for t, tags in zip(toks, predict_tags(model, tk, toks)):
        d = {L: [] for L in ENTITIES}
        for L, s, e in bio_spans(tags):
            v = re.sub(r"\s+", " ", " ".join(t[s:e])).strip(" .,;:-").lower()
            if v and v not in d[L]: d[L].append(v)
        d["experience_years"] = experience_years(d["EXPERIENCE"])
        out.append(d)
    return out


if __name__ == "__main__" and "--export" in sys.argv:
    import time
    model, tk = load()
    atlas = pd.read_parquet(DATA / "resume_atlas.parquet").rename(columns={"Text": "text"})
    ats = pd.read_parquet(DATA / "resume_ats.parquet")
    parts = ats.text.str.split(r"\s+SEP\s+", n=1, regex=True)
    targets = {"atlas": atlas.text.fillna(""), "ats_resume": parts.str[0].drop_duplicates(), "ats_job": parts.str[1].fillna("").drop_duplicates()}
    for name, s in targets.items():
        t0 = time.time()
        df = pd.DataFrame(extract(s.tolist(), model, tk)).assign(source_index=s.index.values, text=s.values)
        if name == "atlas": df["Category"] = atlas.Category.loc[s.index].values
        df.to_parquet(DATA / f"ner_pred_{name}.parquet", index=False)
        print(f"{name}: {len(df)} teks, {time.time() - t0:.0f}s |", {L: round(df[L].str.len().mean(), 1) for L in ENTITIES},
              f"| experience_years terisi {df.experience_years.notna().mean():.0%}")
