"""
scorer.py
Bayesian score fusion: combines TF-IDF, Fuzzy, and BERT signals
into a posterior match probability using Bayes' theorem.

Also includes Precision@K and Recall@K evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm
import os


# ── BAYESIAN FUSION ─────────────────────────────────────────────────────────
def _likelihood_ratio(score: float,
                      match_mean=0.85, match_std=0.10,
                      nonmatch_mean=0.30, nonmatch_std=0.15) -> float:
    """
    LR = P(score | true_match) / P(score | no_match)
    Both modelled as Gaussians.
    LR > 1 means the score is more consistent with a true match.
    """
    p_match    = scipy_norm.pdf(score, match_mean, match_std) + 1e-12
    p_nonmatch = scipy_norm.pdf(score, nonmatch_mean, nonmatch_std) + 1e-12
    return p_match / p_nonmatch


def bayesian_posterior(tfidf_score: float,
                       fuzzy_score: float,
                       bert_score: float,
                       prior: float = 0.001) -> float:
    """
    Compute P(match | tfidf, fuzzy, bert) via Bayes:
        posterior_odds = prior_odds × LR_tfidf × LR_fuzzy × LR_bert
    Assumes conditional independence of signals given match status.
    """
    prior_odds = prior / (1.0 - prior)
    lr_t = _likelihood_ratio(tfidf_score)
    lr_f = _likelihood_ratio(fuzzy_score)
    lr_b = _likelihood_ratio(bert_score, match_mean=0.80)

    posterior_odds = prior_odds * lr_t * lr_f * lr_b
    prob = posterior_odds / (1.0 + posterior_odds)
    return float(np.clip(prob, 1e-6, 1.0 - 1e-6))


class BayesianScreener:
    def __init__(self, tfidf_matcher, fuzzy_matcher, bert_matcher, watchlist_df):
        self.tfidf  = tfidf_matcher
        self.fuzzy  = fuzzy_matcher
        self.bert   = bert_matcher
        self.watch_df = watchlist_df

    def _get_score(self, matches: list, watch_id: int, score_key: str) -> float:
        for m in matches:
            if m['watch_id'] == watch_id:
                return m[score_key]
        return 0.0

    def screen(self, query_name: str, top_k: int = 3) -> list:
        """
        Screen a single name against the watchlist.
        Returns top_k candidates ranked by Bayesian posterior probability.
        """
        tfidf_matches = self.tfidf.match(query_name, top_k=top_k + 3)
        fuzzy_matches = self.fuzzy.match(query_name, top_k=top_k + 3)
        bert_matches  = self.bert.match(query_name,  top_k=top_k + 3)

        all_ids = set(
            [m['watch_id'] for m in tfidf_matches] +
            [m['watch_id'] for m in fuzzy_matches] +
            [m['watch_id'] for m in bert_matches] +
            list(self.watch_df['watch_id'].unique())
        )

        results = []
        for wid in all_ids:
            t = self._get_score(tfidf_matches, wid, 'tfidf_score')
            f = self._get_score(fuzzy_matches, wid, 'fuzzy_score')
            b = self._get_score(bert_matches,  wid, 'bert_score')
            post = bayesian_posterior(t, f, b)
            w_name = self.watch_df[self.watch_df['watch_id'] == wid]['name'].values[0] \
                if (self.watch_df['watch_id'] == wid).any() else '?'
            results.append({
                'watch_id': wid, 'watch_name': w_name,
                'tfidf_score': t, 'fuzzy_score': f, 'bert_score': b,
                'posterior': post
            })

        return sorted(results, key=lambda x: x['posterior'], reverse=True)[:top_k]


# ── EVALUATION: Precision@K and Recall@K ────────────────────────────────────
def precision_at_k(top_k_ids: list, relevant_ids: set) -> float:
    """Of the top-K results, what fraction are relevant?"""
    hits = sum(1 for i in top_k_ids if i in relevant_ids)
    return hits / len(top_k_ids) if top_k_ids else 0.0


def recall_at_k(top_k_ids: list, relevant_ids: set) -> float:
    """Of all relevant items, what fraction appear in top-K?"""
    if not relevant_ids:
        return 1.0
    hits = sum(1 for i in top_k_ids if i in relevant_ids)
    return hits / len(relevant_ids)


def evaluate_system(screener, test_transactions: list, k: int = 3) -> pd.DataFrame:
    """
    Evaluate the full screening pipeline on test transactions.
    Returns per-transaction results and prints aggregate P@K and R@K.
    """
    records = []
    for tx in test_transactions:
        expected = tx['expected_match']
        results  = screener.screen(tx['counterparty'], top_k=k)
        top_ids  = [r['watch_id'] for r in results]
        relevant = {expected} if expected is not None else set()

        p_k = precision_at_k(top_ids, relevant)
        r_k = recall_at_k(top_ids, relevant)
        best_post = results[0]['posterior'] if results else 0.0

        records.append({
            'tx_id':       tx['tx_id'],
            'counterparty': tx['counterparty'],
            'expected':    expected,
            'top1_id':     top_ids[0] if top_ids else None,
            'top1_posterior': round(best_post, 4),
            f'P@{k}':      round(p_k, 3),
            f'R@{k}':      round(r_k, 3),
            'correct':     (expected is None and best_post < 0.30) or
                           (expected is not None and top_ids[0] == expected)
        })

    df = pd.DataFrame(records)

    # ── Only score on transactions that have a ground-truth match ──────────
    has_match = df[df['expected'].notna()]
    no_match  = df[df['expected'].isna()]

    print(f"\n=== EVALUATION RESULTS (K={k}) ===")
    print(df[[c for c in df.columns if c not in ['counterparty']]].to_string(index=False))
    print(f"\nMean P@{k} (match txns): {has_match[f'P@{k}'].mean():.3f}")
    print(f"Mean R@{k} (match txns): {has_match[f'R@{k}'].mean():.3f}")
    print(f"Overall accuracy:        {df['correct'].mean() * 100:.0f}%")
    return df


def plot_comparison(eval_results: pd.DataFrame, output_dir: str):
    """Bar chart of posterior probabilities for all test transactions."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ['tomato' if e is not None else 'steelblue'
              for e in eval_results['expected']]
    bars = ax.bar(eval_results['tx_id'], eval_results['top1_posterior'], color=colors)
    ax.axhline(0.30, color='orange', linestyle='--', lw=1.5, label='Review threshold (0.30)')
    ax.axhline(0.85, color='red', linestyle='--', lw=1.5, label='Block threshold (0.85)')
    ax.set_xlabel('Transaction ID')
    ax.set_ylabel('Bayesian Posterior P(match)')
    ax.set_title('EntityGuard — Screening Results\n'
                 'Red bars = true matches, Blue = clean transactions')
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = os.path.join(output_dir, 'screening_results.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Evaluation] Saved: {path}")


if __name__ == "__main__":
    from watchlist import get_expanded_watchlist, TEST_TRANSACTIONS, WATCHLIST
    from name_matching import TFIDFMatcher, FuzzyMatcher, BERTMatcher
    import pandas as pd

    df_names = get_expanded_watchlist()
    watch_df = pd.DataFrame([{'watch_id': e['id'], 'name': e['name']} for e in WATCHLIST])

    tfidf = TFIDFMatcher(); tfidf.fit(df_names)
    fuzzy = FuzzyMatcher(df_names)
    bert  = BERTMatcher(); bert.fit(df_names)

    screener = BayesianScreener(tfidf, fuzzy, bert, watch_df)
    df_eval  = evaluate_system(screener, TEST_TRANSACTIONS, k=3)
    plot_comparison(df_eval, "outputs")
