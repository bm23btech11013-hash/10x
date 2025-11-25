import os
import glob
import json
import hashlib
from typing import List, Dict, Any, Optional
import datetime as dt

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
import google.generativeai as genai



# CONFIG 


DOC_DIR = "data"                     
CHROMA_PATH = "chroma_store"         

METADATA_CACHE_PATH = "metadata_cache.json"    
ANSWER_CACHE_PATH   = "answer_cache.json"       

COLLECTION_NAME = "nebula_policies"

EMBED_MODEL   = "all-MiniLM-L6-v2"
RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEMINI_MODEL  = "gemini-2.5-flash"



API_KEY = os.getenv("GOOGLE_API_KEY") # use from env or just paste here
if not API_KEY:
    raise RuntimeError("❌ Please set GOOGLE_API_KEY before running this script.")

genai.configure(api_key=API_KEY)



# JSON CACHE HELPERS


def load_json(path: str):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


metadata_cache = load_json(METADATA_CACHE_PATH)
answer_cache   = load_json(ANSWER_CACHE_PATH)



# BASIC HELPERS


def clean_metadata(meta: dict) -> dict:
    return {k: ("" if v is None else v) for k, v in meta.items()}

def parse_date(d: str) -> Optional[dt.date]:
    if not d:
        return None
    d = d.strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y"):
        try:
            return dt.datetime.strptime(d, fmt).date()
        except:
            continue
    return None

def parse_version(v: str) -> float:
    if not v:
        return 0.0
    v = v.lower().strip()
    if v.startswith("v"):
        v = v[1:]
    try:
        return float(v)
    except:
        return 0.0



# METADATA EXTRACTION 


def extract_metadata_llm(text: str, filename: str) -> Dict[str, Any]:

    if filename in metadata_cache:
        return metadata_cache[filename]

    prompt = f"""
You are an HR policy metadata extractor.

Return ONLY valid JSON.

Extract:
- audience_scope
- policy_type
- effective_date
- version
- specificity_level
- override_notes

Document filename: {filename}

Document excerpt:
\"\"\"{text[:4000]}\"\"\"
"""

    model = genai.GenerativeModel(GEMINI_MODEL)
    resp = model.generate_content(prompt)

    try:
        meta = json.loads(resp.text)
    except:
        meta = {}

    final = {
        "audience_scope":   meta.get("audience_scope", "other"),
        "policy_type":      meta.get("policy_type", "uncategorized"),
        "effective_date":   meta.get("effective_date", ""),
        "version":          meta.get("version", ""),
        "specificity_level":meta.get("specificity_level", "unclear"),
        "override_notes":   meta.get("override_notes", ""),
    }

    metadata_cache[filename] = final
    save_json(METADATA_CACHE_PATH, metadata_cache)

    return final



# QUERY CONTEXT


def extract_query_context(query: str) -> Dict[str, Any]:

    prompt = f"""
You are a classifier for HR questions.

Return ONLY JSON:
- user_role
- policy_type

User question:
\"\"\"{query}\"\"\"
"""

    model = genai.GenerativeModel(GEMINI_MODEL)
    resp = model.generate_content(prompt)

    try:
        ctx = json.loads(resp.text)
    except:
        ctx = {}

    return {
        "user_role":  ctx.get("user_role", "unknown"),
        "policy_type":ctx.get("policy_type", "uncategorized"),
    }



# INIT + INGEST


def init_chroma():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn
    )


def ingest_all_txt(collection):

    if collection.count() > 0:
        return

    paths = glob.glob(f"{DOC_DIR}/*.txt")
    if not paths:
        print(f"❌ No .txt files found in {DOC_DIR}")
        return

    for path in paths:
        doc_id = os.path.basename(path)

        if collection.get(ids=[doc_id])["ids"]:
            continue

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        metadata = extract_metadata_llm(text, doc_id)
        safe_meta = clean_metadata({"source": doc_id, **metadata})

        collection.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[safe_meta]
        )

    print("Ingestion complete with metadata.")



# RETRIEVAL


def retrieve_candidates(collection, query: str, q_ctx: Dict[str, Any], top_n=50):

    where = {}
    if q_ctx.get("policy_type") != "uncategorized":
        where["policy_type"] = q_ctx["policy_type"]

    result = collection.query(
        query_texts=[query],
        n_results=top_n,
        where=where or None
    )

    docs = []
    if not result["ids"]:
        return docs

    for i, doc_id in enumerate(result["ids"][0]):
        docs.append({
            "id": doc_id,
            "text": result["documents"][0][i],
            "meta": result["metadatas"][0][i]
        })

    return docs


# RERANK WITH CONFLICT LOGIC


reranker = None

def rerank_with_conflict_logic(query: str, docs: List[Dict], q_ctx: Dict[str, Any], top_k=3):

    global reranker
    if reranker is None:
        reranker = CrossEncoder(RERANK_MODEL)

    user_role = q_ctx.get("user_role", "unknown")

    pairs = [[query, d["text"]] for d in docs]
    base_scores = reranker.predict(pairs)

    scored = []
    for d, score in zip(docs, base_scores):
        m = d["meta"]

        aud = (m.get("audience_scope") or "").lower()
        spec = (m.get("specificity_level") or "").lower()
        eff  = parse_date(m.get("effective_date", ""))
        ver  = parse_version(m.get("version", ""))

        bonus = 0.0

        if user_role in aud:
            bonus += 2.0
        elif aud in ("all_employees", "company_wide"):
            bonus += 0.5

        if spec == "role_specific":
            bonus += 0.5
        elif spec == "company_wide":
            bonus += 0.2

        if eff:
            years = (eff - dt.date(2000,1,1)).days / 365
            bonus += 0.02 * years

        if ver > 0:
            bonus += 0.1 * ver

        scored.append({**d, "score": float(score) + bonus})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]



# RELEVANT SEGMENT EXTRACTION


def extract_relevant_segment(query: str, text: str, max_chars=1200, context_margin=400):

    q = query.lower()
    t = text.lower()

    idx = -1
    for word in q.split():
        pos = t.find(word)
        if pos != -1:
            idx = pos
            break

    if idx == -1:
        return text[:max_chars]

    start = max(0, idx - context_margin)
    end   = min(len(text), start + max_chars)

    return text[start:end]



# FINAL REASONING


def reason_over_docs(query: str, docs: List[Dict], q_ctx: Dict[str, Any]):

    blocks = []
    for i, d in enumerate(docs, 1):
        m = d["meta"]
        seg = extract_relevant_segment(query, d["text"])

        blocks.append(
            f"[DOC {i}]\n"
            f"source: {m['source']}\n"
            f"audience_scope: {m['audience_scope']}\n"
            f"policy_type: {m['policy_type']}\n"
            f"effective_date: {m['effective_date']}\n"
            f"version: {m['version']}\n"
            f"specificity_level: {m['specificity_level']}\n"
            f"override_notes: {m['override_notes']}\n"
            f"--- EXCERPT ---\n"
            f"{seg}\n"
            f"--- END ---\n"
        )

    context = "\n\n".join(blocks)

    prompt = f"""
You are an HR policy assistant.

User role: {q_ctx.get("user_role")}
User question:
\"\"\"{query}\"\"\"

Resolve conflicts using:
- Role > Specificity > Date > Version > Override notes  
- Use ONLY provided excerpts  
- Cite the document filenames in Sources:  

Final Answer: <short answer>
Sources: <filenames>
"""

    model = genai.GenerativeModel(GEMINI_MODEL)
    resp = model.generate_content(prompt)
    return resp.text.strip()



# CACHE + ANSWER


def cached_answer(query: str):
    key = hashlib.md5(query.encode()).hexdigest()
    return answer_cache.get(key)

def store_answer(query: str, answer: str):
    key = hashlib.md5(query.encode()).hexdigest()
    answer_cache[key] = answer
    save_json(ANSWER_CACHE_PATH, answer_cache)


def answer(query: str) -> str:

    cached = cached_answer(query)
    if cached:
        return cached

    collection = init_chroma()
    ingest_all_txt(collection)

    q_ctx = extract_query_context(query)

    candidates = retrieve_candidates(collection, query, q_ctx)
    if not candidates:
        final = "Final Answer: No matching documents found.\nSources: none"
        store_answer(query, final)
        return final

    ranked = rerank_with_conflict_logic(query, candidates, q_ctx)
    final = reason_over_docs(query, ranked, q_ctx)

    store_answer(query, final)
    return final



# MAIN


if __name__ == "__main__":
    print(answer("I just joined as a new intern. Can I work from home?"))
