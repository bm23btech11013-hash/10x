

##  Overview  
This system retrieves, reranks, resolves contradictions, and produces **role-aware, citation-backed answers** using a lightweight hybrid architecture that runs fully on **local infrastructure + Gemini-2.5-Flash**.

It satisfies all the assignment requirements:

- Local vector store (**ChromaDB**)  
- **Gemini-2.5-Flash** for metadata extraction + final reasoning  
- Role-aware and date-aware **conflict resolution pipeline**  
- **Token-optimized** reasoning using segment extraction  
- **Open-source reranker** (MiniLM CrossEncoder)  
- Metadata caching + answer caching for cost reduction  

---

#  Features

- Local vector database using **ChromaDB**  
- Embeddings via **SentenceTransformers: all-MiniLM-L6-v2**  
- Open-source **CrossEncoder** for reranking  
- **Gemini-2.5-Flash** for metadata + reasoning  
- Robust conflict logic  
  - role > specificity > effective_date > version > override notes  
- Relevant-segment extraction (huge cost reduction)  
- Metadata caching (one-time ingestion cost)  
- Answer caching (zero-cost repeated queries)  

---


#  Conflict Logic Explanation

## **1. Role Priority (Highest Weight)**
Example: If `intern_onboarding_faq.txt` applies, it overrides:

- employee_handbook_v1.txt  
- manager_updates_2024.txt  

Even if both mention remote work.

---

## **2. Specificity Level**
Detected automatically as:

- role_specific  
- team_specific  
- company_wide  

Higher specificity = higher ranking.

---

## **3. Effective Date**
Newer policies override older ones.  
Supports multiple date formats automatically.

---

## **4. Version Number**
Documents with:

v2, v3, v2.1, Version 4


get a version bonus.

---

## **5. Override Notes**
Statements like:

- “This policy supersedes…”  
- “Effective immediately…”  
- “This update replaces…”  

receive the strongest priority weight.

---

## **6. Reranking Formula**

final_score =
base_similarity
+ role_bonus
+ specificity_bonus
+ recency_bonus
+ version_bonus
+ override_bonus



No document is hardcoded → **not overfitted**.

Any new policy added will be processed dynamically.

---

## **7. Structured Reasoning Prompt**
Gemini receives:

- Only extracted segments  
- Conflict resolution instructions  
- Role preference  
- Date/version logic  
- Required citation format  
- “Use ONLY the provided text” constraint  

Prevents hallucinations and ensures deterministic answers.

---

## **8. Negation Handling**
The pipeline handles questions such as:

> “What things can interns **NOT** do?”

because:

- CrossEncoder understands negation semantics  
- Relevant segment extraction finds context accurately  
- Role-priority logic ensures correct override  
- Gemini prompt explicitly instructs not to hallucinate  

---

#  System Architecture Overview

## **1. Document Ingestion & Metadata Extraction**
Each `.txt` policy document is passed **once** through Gemini-2.5-Flash to extract structured metadata:

- audience_scope  
- policy_type  
- effective_date  
- version  
- specificity_level  
- override_notes  

Metadata is **cached**, so ingestion cost is paid only once.

---

## **2. Vector Indexing (Local ChromaDB)**
Documents are embedded using:

SentenceTransformers: all-MiniLM-L6-v2


Stored in:

ChromaDB PersistentClient


Local retrieval = **fast**, **free**, **scalable**.

---

## **3. Hybrid Retrieval**
When a user asks a question:

1. The query is classified → user_role + policy_type  
2. Chroma retrieves top-N candidates  
3. Metadata filters refine the set  

Zero LLM cost during retrieval.

---

## **4. Conflict-Aware Reranking (Open-Source)**
This project uses an open-source reranker:

cross-encoder/ms-marco-MiniLM-L-6-v2


It evaluates (query, document) pairs to produce a semantic similarity score.

Then conflict logic adds deterministic boosts based on:

- Matching user role  
- Specificity level  
- Effective date (recency)  
- Version number  
- Explicit override notes  

This allows the system to resolve contradictory policy documents accurately.

---

## **5. Token-Optimized Reasoning with Gemini**
Only the **most relevant 1–3 document segments** are sent to Gemini-2.5-Flash.

Structured prompt ensures:

- No hallucination  
- Role rules override general rules  
- Newer/updated policies override older ones  
- Excerpts only (not full docs)  
- Clear final answer + source citations  

This reduces LLM cost drastically.

---



#  Open-Source Model Usage 

This system integrates:



cross-encoder/ms-marco-MiniLM-L-6-v2


### Why:
- Runs locally on CPU (no GPU needed)  
- Improves retrieval accuracy  
- Reduces LLM load  
- Zero recurring cost  
- Great for RAG pipelines  

Additional notes:
- HuggingFace Transformers used  
- Model can be quantized if needed  

---

#  Cost Calculations (Gemini-2.5-Flash Pricing)

### Pricing:
- **Input:** $0.30 per 1M tokens  
- **Output:** $2.50 per 1M tokens  

---

## **1. One-Time Ingestion Cost**

Document stats:

| Metric | Average |
|--------|---------|
| Characters | 9172 |
| Tokens | ~2293 |
| Prompt overhead | ~650 |
| Input per doc | ~3000 |
| Output per doc | ~150 |

### For **10,000 documents**:

- Input = 30M → **$9.00**  
- Output = 1.5M → **$3.75**  

 **Total one-time ingestion = $12.75**

---

## **2. Recurring Query Cost**
System handles **5000 employee queries/day**.

Due to optimizations:

- Input ≈ 600 tokens/query  
- Output ≈ 200 tokens/query  

### Daily tokens:
- Input = 3,000,000  
- Output = 1,000,000  

### Monthly (30 days):
- Input = 90M → **$27.00**  
- Output = 30M → **$75.00**  

 **Total monthly = $102.00**

---

#  Final Cost Summary

| Category | Cost |
|----------|------|
| **One-time ingestion** | **$12.75** |
| **Monthly (150k queries)** | **$102.00** |
| **Total 1st month** | **$114.75** |
| **Monthly after** | **$102.00** |

---

#  How to Run

### **1. Install dependencies**

pip install -r requirements.txt


### **2. Set API Key**
Mac / Linux:

export GOOGLE_API_KEY="your_key_here"


Windows:
setx GOOGLE_API_KEY "your_key_here"


### **3. Run**
