# EntityGuard
### LLM-Powered Denied Party Screening System
**Amazon Applied Scientist Interview Project — Aditya Deshmukh**

---

## What This Does
Screens transaction party names against a denied party watchlist using three NLP
signals (TF-IDF, fuzzy matching, BERT embeddings), fuses them with Bayes' theorem,
and uses AWS Bedrock (Claude) to generate compliance reasoning — exactly the
LLM-powered detection system described in the Amazon job description.

## Concepts Covered
- **NLP**: TF-IDF, character n-grams, cosine similarity
- **Fuzzy Matching**: Levenshtein distance, Jaro-Winkler similarity, token ratios
- **Embeddings**: BERT semantic embeddings (sentence-transformers), PCA visualization
- **Statistics**: Bayes' theorem, likelihood ratios, posterior probability
- **Evaluation**: Precision@K, Recall@K
- **AWS Bedrock**: Claude (Anthropic) via Messages API, Amazon Titan Embeddings
- **RAG**: Retrieval-Augmented Generation with simulated Knowledge Base

## Project Structure
```
entityguard/
├── main.py            ← Run this — executes the full pipeline
├── watchlist.py       ← Denied party watchlist + test transactions
├── name_matching.py   ← TF-IDF matcher, Fuzzy matcher, BERT matcher
├── scorer.py          ← Bayesian fusion + Precision@K / Recall@K evaluation
├── bedrock_llm.py     ← AWS Bedrock: Claude, Titan Embeddings, RAG pipeline
├── requirements.txt
└── outputs/           ← All charts saved here (auto-created)
```

## Setup & Run
```bash
pip install -r requirements.txt
python main.py          # First run downloads BERT model (~80MB, one time only)
```

## Output Files
| File | Description |
|------|-------------|
| `outputs/tfidf_similarity_matrix.png` | Cosine similarity: 10 transactions × 8 watchlist entities |
| `outputs/bert_embedding_space.png` | PCA of BERT embeddings — similar names cluster together |
| `outputs/screening_results.png` | Bayesian posterior per transaction with decision thresholds |

## Enabling Real AWS Bedrock
```bash
# 1. Install AWS CLI and configure credentials
aws configure    # enter Access Key, Secret, Region: us-east-1

# 2. Enable model access in AWS Console:
#    Bedrock → Model Access → Request:
#      anthropic.claude-3-haiku-20240307-v1:0
#      amazon.titan-embed-text-v2:0

# 3. In bedrock_llm.py, change:
MOCK_MODE = False
```
Cost: ~$0.001 per full pipeline run (Claude Haiku pricing).
