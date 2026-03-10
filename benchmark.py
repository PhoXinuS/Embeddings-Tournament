import time, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from classic_w2v import train_word2vec

# Settings
CORPUS_MB, EMB_DIM, EPOCHS = 20, 100, 10
EVAL_DIR = Path("eval_data")
WORDSIM_FILE = EVAL_DIR / "wordsim353/combined.tab"
ANALOGY_FILE = EVAL_DIR / "questions-words.txt"

def get_stats(model, dataset):
    # Vector helpers
    normed = model.input_embeddings / np.linalg.norm(model.input_embeddings, axis=1, keepdims=True).clip(min=1e-9)
    get_v = lambda w: normed[dataset.word_to_index[w]] if w in dataset.word_to_index else None

    # WordSim-353
    gold, pred = [], []
    if WORDSIM_FILE.exists():
        for line in WORDSIM_FILE.read_text().splitlines()[1:]:
            p = line.lower().split('\t')
            v1, v2 = get_v(p[0]), get_v(p[1])
            if v1 is not None and v2 is not None:
                gold.append(float(p[2])); pred.append(np.dot(v1, v2))
    rho = spearmanr(gold, pred)[0] if len(gold) > 1 else 0

    # Analogies (a:b :: c:d -> b-a+c)
    cor, total = 0, 0
    if ANALOGY_FILE.exists():
        for line in ANALOGY_FILE.read_text().lower().splitlines():
            if line.startswith(':') or len(p := line.split()) != 4: continue
            idxs = [dataset.word_to_index.get(w) for w in p]
            if None not in idxs:
                query = normed[idxs[1]] - normed[idxs[0]] + normed[idxs[2]]
                sims = normed @ query
                for i in idxs[:3]: sims[i] = -np.inf
                if np.argmax(sims) == idxs[3]: cor += 1
                total += 1
    return rho, (cor/total*100 if total > 0 else 0)

# 1. Prep Data
EVAL_DIR.mkdir(exist_ok=True)

results = {}
with open("dataset/text8", "r") as f: raw_corpus = f.read(CORPUS_MB * 1024 * 1024)

# 2. Run Benchmark
print(f"| Architecture | Words/Sec | WordSim (ρ) | Analogy Acc |")
print(f"| :--- | :--- | :--- | :--- |")

for arch in ["cbow", "skip-gram"]:
    t0 = time.perf_counter()
    model, ds = train_word2vec(raw_corpus, EPOCHS, EMB_DIM, 5, arch)
    wps = len(ds.processed_corpus) * EPOCHS / (time.perf_counter() - t0)
    rho, acc = get_stats(model, ds)
    results[arch] = [wps, rho, acc]
    print(f"| {arch:12} | {wps:9,.0f} | {rho:11.3f} | {acc:10.1f}% |")

# 3. Plot
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
titles = ["Training Speed (WPS)", "WordSim-353 (ρ)", "Analogy Accuracy (%)"]
for i, ax in enumerate(axes):
    ax.bar(results.keys(), [v[i] for v in results.values()], color=['#4C72B0', '#DD8452'])
    ax.set_title(titles[i])
plt.tight_layout()
plt.savefig("benchmark.png")
plt.show()