"""
name_matching.py
Three complementary name-matching signals:
  1. TF-IDF character n-gram cosine similarity
  2. Fuzzy string matching (Levenshtein, Jaro-Winkler, token ratios)
  3. BERT semantic embeddings cosine similarity
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, distance as rfuzz_dist
import os


# ── TEXT PREPROCESSING ─────────────────────────────────────────────────────
def preprocess(name: str) -> str:
    """Lowercase, strip punctuation except spaces, collapse whitespace."""
    name = name.lower()
    name = re.sub(r'[^a-z0-9 ]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


# ── SIGNAL 1: TF-IDF ────────────────────────────────────────────────────────
class TFIDFMatcher:
    def __init__(self, ngram_range=(2, 4)):
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=ngram_range,
            min_df=1,
            sublinear_tf=True   # log(tf)+1 — reduces impact of very common n-grams
        )
        self.watch_vectors = None
        self.df_names = None

    def fit(self, df_names: pd.DataFrame):
        """Fit TF-IDF on all watchlist names (primary + aliases)."""
        self.df_names = df_names.copy()
        self.df_names['clean'] = self.df_names['name'].apply(preprocess)
        self.watch_vectors = self.vectorizer.fit_transform(self.df_names['clean'])
        print(f"[TF-IDF] Vocabulary: {len(self.vectorizer.vocabulary_):,} n-grams")

    def match(self, query: str, top_k: int = 3) -> list:
        q_vec = self.vectorizer.transform([preprocess(query)])
        sims = cosine_similarity(q_vec, self.watch_vectors)[0]
        top_idx = sims.argsort()[::-1]

        seen, results = set(), []
        for idx in top_idx:
            wid = self.df_names.iloc[idx]['watch_id']
            if wid not in seen:
                seen.add(wid)
                results.append({
                    'watch_id': wid,
                    'matched_name': self.df_names.iloc[idx]['name'],
                    'tfidf_score': float(sims[idx])
                })
            if len(results) == top_k:
                break
        return results

    def similarity_matrix(self, queries: list) -> np.ndarray:
        """Return (n_queries × n_watchlist_entities) max-similarity matrix."""
        from watchlist import WATCHLIST
        q_vecs = self.vectorizer.transform([preprocess(q) for q in queries])
        full = cosine_similarity(q_vecs, self.watch_vectors)
        n_entities = len(WATCHLIST)
        mat = np.zeros((len(queries), n_entities))
        for i, entry in enumerate(WATCHLIST):
            mask = self.df_names['watch_id'] == entry['id']
            cols = [self.df_names.index.get_loc(j) for j in self.df_names[mask].index]
            mat[:, i] = full[:, cols].max(axis=1)
        return mat


# ── SIGNAL 2: FUZZY ─────────────────────────────────────────────────────────
class FuzzyMatcher:
    def __init__(self, df_names: pd.DataFrame):
        self.df_names = df_names.copy()
        self.df_names['clean'] = self.df_names['name'].apply(preprocess)

    def match(self, query: str, top_k: int = 3) -> list:
        q = preprocess(query)
        rows = []
        for _, row in self.df_names.iterrows():
            w = row['clean']
            jaro    = fuzz.WRatio(q, w) / 100.0
            tsort   = fuzz.token_sort_ratio(q, w) / 100.0
            tset    = fuzz.token_set_ratio(q, w) / 100.0
            lev     = rfuzz_dist.Levenshtein.normalized_similarity(q, w)
            combined = 0.30 * jaro + 0.30 * tsort + 0.25 * tset + 0.15 * lev
            rows.append({'watch_id': row['watch_id'],
                         'matched_name': row['name'],
                         'jaro': jaro, 'token_sort': tsort,
                         'token_set': tset, 'lev': lev,
                         'fuzzy_score': combined})

        df_r = pd.DataFrame(rows).sort_values('fuzzy_score', ascending=False)
        seen, results = set(), []
        for _, r in df_r.iterrows():
            if r['watch_id'] not in seen:
                seen.add(r['watch_id'])
                results.append(r.to_dict())
            if len(results) == top_k:
                break
        return results


# ── SIGNAL 3: BERT EMBEDDINGS ────────────────────────────────────────────────
class BERTMatcher:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"[BERT] Loading model '{model_name}' (first run downloads ~80MB)...")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.watch_embeddings = None
        self.df_names = None

    def fit(self, df_names: pd.DataFrame):
        self.df_names = df_names.copy()
        print("[BERT] Encoding watchlist names...")
        self.watch_embeddings = self.model.encode(
            df_names['name'].tolist(),
            normalize_embeddings=True,  # L2-norm: cosine_sim = dot product
            show_progress_bar=True
        )
        print(f"[BERT] Embeddings shape: {self.watch_embeddings.shape}")

    def match(self, query: str, top_k: int = 3) -> list:
        q_emb = self.model.encode([query], normalize_embeddings=True)
        sims = (q_emb @ self.watch_embeddings.T)[0]
        top_idx = sims.argsort()[::-1]

        seen, results = set(), []
        for idx in top_idx:
            wid = self.df_names.iloc[idx]['watch_id']
            if wid not in seen:
                seen.add(wid)
                results.append({'watch_id': wid,
                                'matched_name': self.df_names.iloc[idx]['name'],
                                'bert_score': float(sims[idx])})
            if len(results) == top_k:
                break
        return results

    def embedding_2d_plot(self, extra_names: list, output_dir: str):
        """PCA projection of all name embeddings for visualization."""
        from sklearn.decomposition import PCA
        os.makedirs(output_dir, exist_ok=True)

        extra_embs = self.model.encode(extra_names, normalize_embeddings=True)
        all_embs = np.vstack([self.watch_embeddings, extra_embs])

        pca = PCA(n_components=2, random_state=42)
        pts = pca.fit_transform(all_embs)
        var = pca.explained_variance_ratio_

        plt.figure(figsize=(12, 8))
        n_w = len(self.df_names)
        plt.scatter(pts[:n_w, 0], pts[:n_w, 1],
                    c='steelblue', marker='s', s=80, label='Watchlist', zorder=3)
        for i, name in enumerate(self.df_names['name']):
            plt.annotate(name[:22], pts[i], textcoords='offset points',
                         xytext=(4, 4), fontsize=7, color='steelblue')

        plt.scatter(pts[n_w:, 0], pts[n_w:, 1],
                    c='tomato', marker='^', s=80, label='Test Transactions', zorder=3)
        for i, name in enumerate(extra_names):
            plt.annotate(name[:22], pts[n_w + i], textcoords='offset points',
                         xytext=(4, -10), fontsize=7, color='tomato')

        plt.title(f'BERT Embedding Space (PCA)\n'
                  f'Variance explained: {var.sum() * 100:.1f}%  '
                  f'— Close points = semantically similar names')
        plt.xlabel(f'PC1 ({var[0] * 100:.1f}%)')
        plt.ylabel(f'PC2 ({var[1] * 100:.1f}%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, 'bert_embedding_space.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[BERT] Saved: {path}")


if __name__ == "__main__":
    from watchlist import get_expanded_watchlist, TEST_TRANSACTIONS
    df_names = get_expanded_watchlist()

    tfidf = TFIDFMatcher()
    tfidf.fit(df_names)

    fuzzy = FuzzyMatcher(df_names)
    bert = BERTMatcher()
    bert.fit(df_names)

    for tx in TEST_TRANSACTIONS[:3]:
        name = tx['counterparty']
        print(f"\n{'='*55}\nQuery: {name}")
        print("TF-IDF:", tfidf.match(name, 3))
        print("Fuzzy: ", [(m['watch_id'], round(m['fuzzy_score'], 3))
                          for m in fuzzy.match(name, 3)])
        print("BERT:  ", [(m['watch_id'], round(m['bert_score'], 3))
                          for m in bert.match(name, 3)])
