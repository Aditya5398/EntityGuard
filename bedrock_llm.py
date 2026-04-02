"""
bedrock_llm.py
AWS Bedrock integration for LLM-powered compliance reasoning.

Supports:
  - Claude (Anthropic) via Messages API
  - Amazon Titan Embeddings
  - RAG via simulated Knowledge Base (real KB code included as comments)

Set MOCK_MODE = False and run `aws configure` to use real Bedrock.
"""

import json
import numpy as np


MOCK_MODE = True   # ← Change to False after running: aws configure

SYSTEM_PROMPT = """You are an expert sanctions compliance analyst AI.
Determine whether a transaction party matches a denied party on a watchlist.

Consider: name variations, transliterations, abbreviations (LLC/Ltd/Corp), 
cultural name patterns (Al/Al-/Al_ are equivalent; Mohammed/Mohammad/Muhammad 
are equivalent), and semantic meaning.

Always respond ONLY in valid JSON with exactly these fields:
{
  "decision": "LIKELY_MATCH" or "POSSIBLE_MATCH" or "NO_MATCH",
  "confidence": 0.0 to 1.0,
  "reasoning": "detailed explanation",
  "matched_watch_id": integer or null,
  "sanctions_basis": "regulation name" or null,
  "recommended_action": "BLOCK_AND_ESCALATE" or "HOLD_FOR_REVIEW" or "ALLOW",
  "analyst_summary": "one sentence for compliance analyst"
}"""


# ── MOCK RESPONSE SIMULATOR ─────────────────────────────────────────────────
def _mock_claude_response(tx_name: str, candidates: list) -> str:
    """Simulates a realistic Claude response for demo/testing."""
    name_lower = tx_name.lower()
    top_candidate = candidates[0] if candidates else {}
    top_post = float(top_candidate.get('posterior', 0))

    if top_post > 0.70:
        watch_id = top_candidate.get('watch_id')
        watch_name = top_candidate.get('watch_name', 'Unknown')
        return json.dumps({
            "decision": "LIKELY_MATCH",
            "confidence": round(min(0.97, top_post + 0.08), 2),
            "reasoning": (
                f"The name '{tx_name}' shares key semantic and lexical components "
                f"with watchlist entry '{watch_name}' (ID:{watch_id}). "
                f"Variations observed are standard transliteration artifacts "
                f"(punctuation, abbreviation, word order). "
                f"Bayesian pre-screening posterior: {top_post:.3f}."
            ),
            "matched_watch_id": int(watch_id) if watch_id is not None else None,
            "sanctions_basis": "OFAC SDN List",
            "recommended_action": "BLOCK_AND_ESCALATE",
            "analyst_summary": (
                f"High-confidence match to '{watch_name}'. "
                f"Block transaction and escalate to compliance team."
            )
        })
    elif top_post > 0.30:
        return json.dumps({
            "decision": "POSSIBLE_MATCH",
            "confidence": round(top_post, 2),
            "reasoning": (
                f"Partial similarity detected between '{tx_name}' and top "
                f"watchlist candidate. Insufficient information for high-confidence "
                f"determination. Manual review recommended."
            ),
            "matched_watch_id": int(top_candidate.get('watch_id')) if top_candidate.get('watch_id') is not None else None,
            "sanctions_basis": "Pending investigation",
            "recommended_action": "HOLD_FOR_REVIEW",
            "analyst_summary": "Manual review required. Assign to compliance analyst."
        })
    else:
        return json.dumps({
            "decision": "NO_MATCH",
            "confidence": round(1.0 - top_post, 2),
            "reasoning": (
                f"No significant similarity between '{tx_name}' and any watchlist entry. "
                f"Surface-level similarity to some entries is coincidental."
            ),
            "matched_watch_id": None,
            "sanctions_basis": None,
            "recommended_action": "ALLOW",
            "analyst_summary": "No watchlist match. Transaction cleared."
        })


# ── BEDROCK API CALLS ────────────────────────────────────────────────────────
def call_claude(tx_name: str, candidates: list, max_tokens: int = 800) -> dict:
    """
    Call Claude via Bedrock (or mock) and return parsed JSON response.

    Real Bedrock call (when MOCK_MODE = False):
        bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
        response = bedrock.invoke_model(
            modelId='anthropic.claude-3-haiku-20240307-v1:0',
            contentType='application/json', accept='application/json',
            body=json.dumps({
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': max_tokens,
                'system': SYSTEM_PROMPT,
                'messages': [{'role': 'user', 'content': prompt}]
            })
        )
        text = json.loads(response['body'].read())['content'][0]['text']
    """
    candidates_text = "\n".join([
        f"  Candidate {i+1} (ID:{c['watch_id']}): {c['watch_name']} | "
        f"TF-IDF={c['tfidf_score']:.3f} Fuzzy={c['fuzzy_score']:.3f} "
        f"BERT={c['bert_score']:.3f} Posterior={c['posterior']:.4f}"
        for i, c in enumerate(candidates[:3])
    ])

    prompt = f"""TRANSACTION PARTY: {tx_name}

TOP WATCHLIST CANDIDATES (from automated pre-screening):
{candidates_text}

TASK: Analyze whether the transaction party is the same entity as any watchlist candidate.
Consider name variations, transliterations, abbreviations, and semantic meaning.
Respond ONLY in the required JSON format."""

    if MOCK_MODE:
        raw = _mock_claude_response(tx_name, candidates)
    else:
        import boto3
        bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
        body = json.dumps({
            'anthropic_version': 'bedrock-2023-05-31',
            'max_tokens': max_tokens,
            'system': SYSTEM_PROMPT,
            'messages': [{'role': 'user', 'content': prompt}]
        })
        resp = bedrock.invoke_model(
            modelId='anthropic.claude-3-haiku-20240307-v1:0',
            contentType='application/json',
            accept='application/json',
            body=body
        )
        raw = json.loads(resp['body'].read())['content'][0]['text']

    # Parse JSON (strip any accidental markdown fences)
    clean = raw.strip().lstrip('```json').lstrip('```').rstrip('```').strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"decision": "ERROR", "reasoning": raw,
                "recommended_action": "HOLD_FOR_REVIEW", "confidence": 0.0}


def call_titan_embed(text: str) -> list:
    """
    Get 1024-dim embedding from Amazon Titan via Bedrock.
    Real call shown; mock returns random vector for testing.
    """
    if MOCK_MODE:
        return list(np.random.randn(1024))

    import boto3
    bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
    resp = bedrock.invoke_model(
        modelId='amazon.titan-embed-text-v2:0',
        contentType='application/json',
        accept='application/json',
        body=json.dumps({'inputText': text})
    )
    return json.loads(resp['body'].read())['embedding']


# ── RAG PIPELINE ─────────────────────────────────────────────────────────────
def rag_screen(tx_name: str, tx_context: str,
               knowledge_base: list, candidates: list) -> dict:
    """
    Retrieval-Augmented Generation:
    1. Find relevant compliance docs from knowledge base
    2. Inject them as context into the LLM prompt
    3. LLM answers grounded in retrieved rules (citable, auditable)

    Production Bedrock KB call:
        agent_runtime = boto3.client('bedrock-agent-runtime', region_name='us-east-1')
        response = agent_runtime.retrieve_and_generate(
            input={'text': question},
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': 'YOUR-KB-ID',
                    'modelArn': 'arn:aws:bedrock:us-east-1::foundation-model/...'
                }
            }
        )
        return response['output']['text']
    """
    # Simulate retrieval: keyword overlap scoring
    question = f"sanctions screening {tx_name}"
    q_words = set(question.lower().split())

    scored_docs = []
    for doc in knowledge_base:
        content_words = set(doc['content'].lower().split())
        overlap = len(q_words & content_words)
        if overlap > 0:
            scored_docs.append((overlap, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored_docs[:2]]

    context = "\n\n".join([
        f"[{doc['title']}]\n{doc['content']}"
        for doc in top_docs
    ]) if top_docs else "No specific regulatory guidance found."

    candidates_text = "\n".join([
        f"  {c['watch_name']} (posterior={c['posterior']:.4f})"
        for c in candidates[:3]
    ])

    rag_prompt = f"""COMPLIANCE REGULATIONS (retrieved from knowledge base):
{context}

TRANSACTION: {tx_name}
{tx_context}

TOP CANDIDATES:
{candidates_text}

Based ONLY on the regulations above, provide a compliance determination in JSON format."""

    if MOCK_MODE:
        result = _mock_claude_response(tx_name, candidates)
        parsed = json.loads(result)
        parsed['retrieved_docs'] = [d['doc_id'] for d in top_docs]
        return parsed
    else:
        import boto3
        bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
        body = json.dumps({
            'anthropic_version': 'bedrock-2023-05-31',
            'max_tokens': 600,
            'system': SYSTEM_PROMPT,
            'messages': [{'role': 'user', 'content': rag_prompt}]
        })
        resp = bedrock.invoke_model(
            modelId='anthropic.claude-3-haiku-20240307-v1:0',
            contentType='application/json', accept='application/json',
            body=body
        )
        raw = json.loads(resp['body'].read())['content'][0]['text']
        clean = raw.strip().lstrip('```json').lstrip('```').rstrip('```').strip()
        return json.loads(clean)


if __name__ == "__main__":
    # Quick test of mock mode
    test_candidates = [
        {'watch_id': 1, 'watch_name': 'Al Baraka Trading Company',
         'tfidf_score': 0.82, 'fuzzy_score': 0.88, 'bert_score': 0.91,
         'posterior': 0.93}
    ]
    result = call_claude("Al Baraka Trading Co.", test_candidates)
    print(json.dumps(result, indent=2))