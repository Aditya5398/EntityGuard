"""
main.py — EntityGuard
Run the complete denied party screening pipeline end to end.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import os

from watchlist import (get_expanded_watchlist, TEST_TRANSACTIONS,
                       WATCHLIST, COMPLIANCE_DOCS)
from name_matching import TFIDFMatcher, FuzzyMatcher, BERTMatcher
from scorer import BayesianScreener, evaluate_system, plot_comparison
from bedrock_llm import call_claude, rag_screen, MOCK_MODE

OUTPUT_DIR = "outputs"


def main():
    print("=" * 60)
    print("  ENTITYGUARD — LLM-Powered Denied Party Screening")
    print("  Amazon Compliance Screening Pipeline")
    print(f"  Bedrock mode: {'MOCK' if MOCK_MODE else 'LIVE AWS'}")
    print("=" * 60)

    # ── Step 1: Build watchlist ────────────────────────────────────────────
    print("\n[1/5] Loading watchlist...")
    df_names  = get_expanded_watchlist()
    watch_df  = pd.DataFrame([{'watch_id': e['id'], 'name': e['name']}
                               for e in WATCHLIST])
    print(f"      {len(WATCHLIST)} entities | {len(df_names)} names (incl. aliases)")

    # ── Step 2: Fit matching models ────────────────────────────────────────
    print("\n[2/5] Fitting TF-IDF matcher...")
    tfidf = TFIDFMatcher()
    tfidf.fit(df_names)

    print("\n[2/5] Initialising fuzzy matcher...")
    fuzzy = FuzzyMatcher(df_names)

    print("\n[2/5] Loading BERT model...")
    bert = BERTMatcher()
    bert.fit(df_names)

    # Save BERT embedding visualization
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    bert.embedding_2d_plot(
        extra_names=[tx['counterparty'] for tx in TEST_TRANSACTIONS],
        output_dir=OUTPUT_DIR
    )

    # TF-IDF similarity heatmap
    print("\n[3/5] Running TF-IDF similarity analysis...")
    queries = [tx['counterparty'] for tx in TEST_TRANSACTIONS]
    sim_mat = tfidf.similarity_matrix(queries)
    entity_labels = [f"E{e['id']}:{e['name'][:12]}" for e in WATCHLIST]
    tx_labels = [tx['tx_id'] for tx in TEST_TRANSACTIONS]

    plt.figure(figsize=(14, 6))
    import seaborn as sns
    sns.heatmap(sim_mat, xticklabels=entity_labels, yticklabels=tx_labels,
                annot=True, fmt='.2f', cmap='YlOrRd', vmin=0, vmax=1)
    plt.title('TF-IDF Cosine Similarity\n(Rows=Test Transactions, Cols=Watchlist Entities)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'tfidf_similarity_matrix.png'), dpi=150)
    plt.close()
    print(f"      Saved: outputs/tfidf_similarity_matrix.png")

    # ── Step 3: Bayesian scoring + evaluation ─────────────────────────────
    print("\n[3/5] Running Bayesian score fusion...")
    screener = BayesianScreener(tfidf, fuzzy, bert, watch_df)
    eval_df  = evaluate_system(screener, TEST_TRANSACTIONS, k=3)
    plot_comparison(eval_df, OUTPUT_DIR)

    # ── Step 4: Bedrock LLM screening ─────────────────────────────────────
    print(f"\n[4/5] Running Bedrock LLM screening ({('MOCK' if MOCK_MODE else 'LIVE')})...")
    llm_results = []
    for tx in TEST_TRANSACTIONS:
        candidates = screener.screen(tx['counterparty'], top_k=3)
        llm_result = call_claude(tx['counterparty'], candidates)

        expected = tx['expected_match']
        decision = llm_result.get('decision', 'ERROR')
        action   = llm_result.get('recommended_action', 'UNKNOWN')
        conf     = llm_result.get('confidence', 0.0)

        correct = (
            (expected is not None and decision != 'NO_MATCH') or
            (expected is None and decision == 'NO_MATCH')
        )
        status = "✓" if correct else "✗"

        print(f"  {status} {tx['tx_id']}: '{tx['counterparty']}'")
        print(f"      Decision={decision:<15} Confidence={conf:.2f}  Action={action}")

        llm_results.append({**tx, 'decision': decision,
                             'confidence': conf, 'action': action, 'correct': correct})

    # ── Step 5: RAG demo on highest-risk transaction ───────────────────────
    print("\n[5/5] RAG compliance reasoning demo...")
    high_risk_tx = TEST_TRANSACTIONS[0]   # 'Al Baraka Trading Co.'
    candidates   = screener.screen(high_risk_tx['counterparty'], top_k=3)
    tx_context   = f"Amount: $45,000 | Country: Iran | Hour: 02:00"
    rag_result   = rag_screen(
        high_risk_tx['counterparty'], tx_context,
        COMPLIANCE_DOCS, candidates
    )

    print(f"\n  RAG Result for '{high_risk_tx['counterparty']}':")
    print(f"  Decision: {rag_result.get('decision')}")
    print(f"  Basis:    {rag_result.get('sanctions_basis')}")
    print(f"  Summary:  {rag_result.get('analyst_summary')}")
    retrieved = rag_result.get('retrieved_docs', [])
    if retrieved:
        print(f"  Cited docs: {', '.join(retrieved)}")

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    n_correct = sum(r['correct'] for r in llm_results)
    print(f"  LLM accuracy: {n_correct}/{len(llm_results)} = "
          f"{n_correct / len(llm_results) * 100:.0f}%")
    print(f"  All outputs saved to: {OUTPUT_DIR}/")
    print("=" * 60)

    print("\nGenerated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  outputs/{f}")


if __name__ == "__main__":
    main()
