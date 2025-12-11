## FormlyAI Agent Platform

Production-grade, multi-agent, multimodal RAG platform for querying FDA 510(k) medical device summaries.

This application demonstrates a "Tool-as-a-Service" (TaaS) architecture, where autonomous agents are dynamically assembled to solve complex regulatory queries using a combination of local vector search, structured SQL filtering, and live internet data.

## 🎯 How to Demo (Direct POC Access)

1. **Open the App:** Navigate to https://formlyaichallenge-ru8zvp74m2ln23tzybg5to.streamlit.app/

2. **Register/Login:**

- Use the **"Register"** tab to create a new account (Email/Password).
- Log in with your new credentials.

3. **Configure API Keys (First Time Only):**

- Open the **"🔐 My API Keys"** expander in the sidebar.
- Enter your keys for **Google (Gemini), Cohere, Tavily, and OpenWeatherMap**.
- Click **"Save API Keys"**. Note: Keys are encrypted (Fernet) and stored securely in the Supabase database for your future sessions.

4. **Launch Workspace:**

- Select your **Mode** (Chatbot vs. Retrieval).
- Click **"Load Workspace"**.

5. **Test Queries:**

- Chatbot: "Find devices similar to the CardioScribe ECG Monitor."
- Retrieval: Enter ```K063152``` to see the split-screen comparison and justification.

## 📝 Challenge Solution: RAG Pipeline Improvements

This platform implements a specific solution to the challenge of improving retrieval over 3,000 **FDA 510(k)** documents.

### 1. Section-Aware Hybrid Architecture (Hierarchical, Multi-Vector)

- **Structured Ingestion:** Instead of blind chunking, an **LLM-driven Extractor Agent** (```ingestion.py```) that parses documents to extract specific fields: ```Intended Use```, ```Indications```, ```Device Description```, and ```Materials```.

- **Hierarchical Storage:** Vectors for each section separately are stored in Qdrant, alongside a full-document chunk.

- **Weighted Multi-Query Retrieval:** Instead of single search, the agent runs parallel searches against specific section vectors and aggregates the scores using a domain-specific weighted formula:

$$
S_{final} = 0.4 \cdot S_{indications} + 0.3 \cdot S_{intended} + 0.2 \cdot S_{description} + 0.1 \cdot S_{materials}
$$

This prioritizes regulatory equivalence over physical description.

- **Hybrid Search:** Dense vector results are combined with a sparse **SQL ILIKE** search (via **DuckDB**) to ensure exact matches for Manufacturer names and K-Numbers are never missed.

- **Cross-Encoder Reranking:** The top 25 results are passed to the **Cohere Rerank v3** model. This model reads the full query and document text to filter out "vector hallucinations" and re-orders the list based on true relevance. 

### 2. 3. Evaluation Strategy

- **Golden Set:** Established a test set (```k_numbers_to_compare.txt```) of known devices and their predicates.

- **Metrics:** Optimized for **Recall@k** (did we find the predicate?) and **NDCG@k** (was it at the top of the list?).

- **A/B Testing:** The platform supports switching between a "Text-Only" pipeline (```all-MiniLM-L6-v2```) and a "Multimodal" pipeline (```embedding-004```) to empirically measure performance differences.

## 🏗️ System Architecture

### 1. The "Brain": Dynamic Agent Runtime

- **Framework:** ```LangChain``` + ```LangGraph```

- **Tool-as-a-Service:** Agents are not hardcoded. When a user logs in, the system builds a custom ```AgentExecutor``` on the fly, binding only the tools enabled in their workspace configuration.

- **Multimodal:** Supports Text, Images, and PDFs. Uses Gemini Vision to caption images for vector search.

### 2. Security & Multi-Tenancy (Supabase)

- **Authentication:** Pure Supabase Auth.

- **Row-Level Security (RLS):** Database policies enforce strict isolation. Users can only access their own data.

- **Encryption:** User API keys are encrypted at rest using ```cryptography``` (Fernet) before storage.

## 🚀 Local Setup Guide

### 1. Installation

```
git clone [https://github.com/your-repo/formly-agent.git](https://github.com/your-repo/formly-agent.git)
cd formly-agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

### 2. Configuration

Create a ```config.yaml``` file (see ```config.yaml``` in repo) and add your ```SUPABASE_URL```, ```SUPABASE_KEY (anon)```, and ```FERNET_KEY```.

### 3. Database Initialization

Run the ```setup_database.sql``` script in your Supabase SQL Editor. This is idempotent and sets up tables, RLS policies, and the ```get_user_sessions``` function.

### 4. Data Ingestion

Build the local vector and SQL databases:

```
# 1. Download Corpus
python3 download_corpus.py

# 2. Create Sample Set
python3 create_sampled_corpus.py

# 3. Ingest (Run for both modes)
python3 ingestion.py --mode text_only
python3 ingestion.py --mode multimodal

```

### 5. Run App

```
streamlit run streamlit_app.py

```