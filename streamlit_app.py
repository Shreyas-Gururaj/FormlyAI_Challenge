import streamlit as st
import json
import uuid
import time
import re
import os
from typing import List, Dict, Any, TypedDict, Optional

# --- Import all our service *factories* from utils ---
from utils import (
    console,
    get_llm,
    get_embeddings,
    get_reranker,
    get_tavily_search,
    get_weather_tool,
    get_config,
    get_prompt,
    get_db_connection,
    get_qdrant_client,
)
# ---

# --- !! NEW: Import Auth & DB Utilities ---
from auth_utils import (
    get_user_api_keys,
    save_user_api_keys,
    get_session_list,
    get_chat_history,
    save_chat_message,
    sign_up_user,
    sign_in_user,
    sign_out_user,
    get_current_user,
    supabase,
)
# ---

# Import LangChain components
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END
from qdrant_client import models
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

# --- Page Configuration ---
st.set_page_config(
    page_title="FormlyAI 510(k) Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Logo Path ---
LOGO_PATH = "assets/logo.svg"
LOGO_URL_FALLBACK = "https://placehold.co/300x100/000000/FFFFFF?text=FormlyAI&font=inter"

# --- Pydantic Schemas (for RAG tool) ---
class QueryPlan(BaseModel):
    semantic_query: str
    sparse_query: str
    filters: Optional[Dict[str, Any]]

class RetrievedDoc(BaseModel):
    k_number: str
    device_name: str
    manufacturer: str
    text: str
    source: str
    score: float

class SimilarityJustification(BaseModel):
    is_similar: bool = Field(
        description="True if the documents are relevantly similar, False otherwise."
    )
    justification: str = Field(
        description="A 1-2 sentence explanation of *why* they are similar, focusing on indications or function."
    )
    query_doc_highlights: List[str] = Field(
        description="A list of 1-3 key phrases from the Query Document that justify the similarity."
    )
    result_doc_highlights: List[str] = Field(
        description="A list of 1-3 key phrases from the Result Document that justify the similarity."
    )

# --- Initialize Session State ---
_initial_keys = {
    "workspace_loaded": False,
    "agent_executor": None,
    "selected_mode": "Chatbot (Agentic)",
    "justification_agent": None,
    "selected_session_id": None,
    "chat_history": [],
    "api_keys": None,
    "user": None,
    "page": "login",
    "rerun_pending": False,
}
for k, v in _initial_keys.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- CACHED RESOURCE WRAPPERS ---
@st.cache_resource
def init_supabase_client():
    return supabase

@st.cache_resource
def init_llm(model_name: str, google_key: str):
    return get_llm(model_name, google_key)

@st.cache_resource
def init_embeddings():
    return get_embeddings()

@st.cache_resource
def init_reranker(cohere_key: str):
    return get_reranker(cohere_key)

@st.cache_resource
def init_internet_search(tavily_key: str):
    return get_tavily_search(tavily_key)

@st.cache_resource
def init_db_connection(read_only: bool = True):
    return get_db_connection(read_only=read_only)

@st.cache_resource
def init_qdrant_client():
    return get_qdrant_client()

# --- Helper utilities ---
def clear_user_session_state():
    keys_to_keep = set(_initial_keys.keys())
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]

# --- Dynamic Tool: 510(k) RAG Search Tool ---
@tool
def local_510k_search(query: str) -> str:
    workspace_expander = st.session_state.get("workspace_expander")
    if not workspace_expander:
        workspace_expander = st.empty()

    workspace_expander.info("Tool: Running Local 510(k) RAG Search...")

    llm = st.session_state.get("llm")
    embeddings = st.session_state.get("embeddings")
    reranker = st.session_state.get("reranker")
    db_conn = st.session_state.get("db_conn")
    qdrant_client = st.session_state.get("qdrant_client")

    if not all([llm, embeddings, reranker, db_conn, qdrant_client]):
        workspace_expander.error("Some services are not initialized. Please reload your workspace.")
        return json.dumps([])

    planner_prompt_text = get_prompt("planner_prompt")
    planner_prompt = ChatPromptTemplate.from_template(planner_prompt_text)
    parser = JsonOutputParser(pydantic_model=QueryPlan)
    planner = planner_prompt | llm | parser

    try:
        plan = planner.invoke({"query": query, "schema": QueryPlan.schema_json()})
    except Exception as e:
        workspace_expander.error(f"Query Planner failed: {e}. Using raw query.")
        plan = {"semantic_query": query, "sparse_query": query, "filters": {}}

    workspace_expander.json(plan, expanded=False)

    query_embedding = embeddings.embed_query(plan["semantic_query"])
    weighted_scores = {}
    doc_metadata = {}

    for section, weight in get_config("SECTION_WEIGHTS").items():
        hits = qdrant_client.search(
            collection_name=get_config("QDRANT_COLLECTION"),
            query_vector=query_embedding,
            limit=get_config("VECTOR_TOP_K"),
            query_filter=models.Filter(
                must=[models.FieldCondition(key="source", match=models.MatchValue(value=section))]
            ),
        )
        for hit in hits:
            kn = hit.payload["k_number"]
            score = hit.score * weight
            if kn not in weighted_scores or score > weighted_scores[kn]:
                weighted_scores[kn] = score
                doc_metadata[kn] = hit.payload

    sparse_term = f"%{plan['sparse_query']}%"
    structured_results = db_conn.execute(
        """
        SELECT k_number, device_name, manufacturer, raw_text_summary, contact_person
        FROM device_metadata 
        WHERE device_name ILIKE ? OR manufacturer ILIKE ? OR k_number ILIKE ? OR contact_person ILIKE ?
        """,
        (sparse_term, sparse_term, sparse_term, sparse_term),
    ).fetchall()

    for row in structured_results:
        kn = row[0]
        if kn not in weighted_scores:
            weighted_scores[kn] = 0.1
            doc_metadata[kn] = {
                "k_number": kn,
                "device_name": row[1],
                "manufacturer": row[2],
                "text": row[3],
                "source": "structured_search",
            }

    combined_docs = []
    for kn, score in weighted_scores.items():
        payload = doc_metadata[kn]
        combined_docs.append(
            RetrievedDoc(
                k_number=payload["k_number"],
                device_name=payload["device_name"],
                manufacturer=payload["manufacturer"],
                text=payload["text"],
                source=payload["source"],
                score=score,
            ).dict()
        )

    combined_docs = sorted(combined_docs, key=lambda x: x["score"], reverse=True)

    if not combined_docs:
        workspace_expander.warning("Local RAG: No documents found.")
        return "No documents found in the 510(k) database."

    workspace_expander.success(f"Combined to {len(combined_docs)} unique results.")

    workspace_expander.info("Tool: Running Cohere Re-ranker...")
    docs_to_rerank = [Document(page_content=doc["text"], metadata=doc) for doc in combined_docs]

    try:
        reranked_responses = reranker.compress_documents(query=plan["semantic_query"], documents=docs_to_rerank)
        final_reranked_list = [doc.metadata for doc in reranked_responses]
        workspace_expander.success(f"Cohere reranked to {len(final_reranked_list)} results.")

        for i, doc in enumerate(final_reranked_list):
            with workspace_expander.expander(f"Top Result {i+1}: {doc['k_number']} ({doc['device_name']})"):
                st.json(doc, expanded=False)

        return json.dumps(final_reranked_list)

    except Exception as e:
        workspace_expander.error(f"Cohere Reranker failed: {e}. Returning top K unsorted.")
        return json.dumps(combined_docs[: get_config("RERANKED_TOP_N")])

# --- Main LangGraph Agent ---
class AgentState(TypedDict):
    messages: List[HumanMessage | AIMessage | ToolMessage]

def build_dynamic_agent(enabled_tools_list: List[str]):
    internet_search = st.session_state.get("internet_search")
    tool_registry = {
        "local_510k_search": local_510k_search,
        "internet_search": internet_search,
        "get_weather": get_weather_tool(st.session_state.api_keys["owm"]) if st.session_state.api_keys else None,
    }
    tool_registry = {k: v for k, v in tool_registry.items() if v is not None}

    tools = [tool_registry[tool_name] for tool_name in enabled_tools_list if tool_name in tool_registry]
    tool_names = [getattr(t, "name", str(t)) for t in tools]

    if not tools:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant. You have no tools."),
                ("user", "{query}"),
                ("placeholder", "{chat_history}"),
            ]
        )
        st.session_state.agent_executor = prompt | st.session_state.llm | StrOutputParser()
        return

    prompt_text = get_prompt("main_agent_prompt")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_text.format(tool_names=tool_names)),
            ("placeholder", "{chat_history}"),
            ("user", "{query}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(st.session_state.llm, tools, prompt)
    st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- UI Mode: Retrieval ---
def fetch_one_dict(db_conn, query: str, params: tuple) -> Optional[dict]:
    cursor = db_conn.execute(query, params)
    result = cursor.fetchone()
    if not result:
        return None
    cols = [desc[0] for desc in cursor.description]
    return dict(zip(cols, result))

def get_similarity_justification(query_doc: dict, result_doc: dict) -> SimilarityJustification:
    try:
        if "justification_agent" not in st.session_state or st.session_state.justification_agent is None:
            prompt_text = get_prompt("justification_prompt")
            prompt = ChatPromptTemplate.from_template(prompt_text)
            parser = JsonOutputParser(pydantic_model=SimilarityJustification)
            st.session_state.justification_agent = prompt | st.session_state.llm | parser

        query_snippet = f"Device: {query_doc['device_name']}\nIndications: {query_doc.get('section_indications_for_use', '')}"
        result_snippet = f"Device: {result_doc['device_name']}\nIndications: {result_doc.get('section_indications_for_use', '')}"

        justification = st.session_state.justification_agent.invoke(
            {
                "query_doc": query_snippet,
                "result_doc": result_snippet,
                "schema": SimilarityJustification.schema_json(),
            }
        )
        return justification
    except Exception as e:
        st.error(f"Justification agent failed: {e}")
        return SimilarityJustification(
            is_similar=False,
            justification="Could not determine similarity.",
            query_doc_highlights=[],
            result_doc_highlights=[],
        ).dict()

def highlight_text(text: str, highlights: List[str], color: str) -> str:
    if not text:
        return ""
    style = f"background-color: {color}; border-radius: 3px; padding: 2px 4px; font-weight: 500;"
    sorted_highlights = sorted(highlights, key=len, reverse=True)
    pattern = "|".join(re.escape(phrase) for phrase in sorted_highlights if phrase)

    if not pattern:
        return text

    def replacer(match):
        return f'<span style="{style}">{match.group(0)}</span>'

    text_with_highlights = re.sub(pattern, replacer, text, flags=re.IGNORECASE)
    return text_with_highlights.replace("\n", "  \n")

def run_retrieval_mode_ui():
    st.header("Retrieval Mode: Find Similar Devices")
    st.markdown(
        "This mode finds the most semantically similar devices to a given K-Number, using its vector embedding."
    )

    k_number = st.text_input("Enter a K-Number (e.g., K000009):").strip().upper()
    if st.button("Find Similar"):
        if not k_number:
            st.error("Please enter a K-Number.")
            return

        qdrant_client = st.session_state.qdrant_client
        db_conn = st.session_state.db_conn

        try:
            with st.spinner(f"Fetching query document: {k_number}..."):
                query_doc = fetch_one_dict(
                    db_conn,
                    "SELECT * FROM device_metadata WHERE k_number = ?",
                    (k_number,),
                )

            if not query_doc:
                st.error(f"Error: K-Number {k_number} not found in our database.")
                return

            with st.spinner(f"Searching for seed vector..."):
                query_vector_data, _ = qdrant_client.scroll(
                    collection_name=get_config("QDRANT_COLLECTION"),
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(key="k_number", match=models.MatchValue(value=k_number)),
                            models.FieldCondition(
                                key="source", match=models.MatchValue(value="intended_use")
                            ),
                        ]
                    ),
                    limit=1,
                    with_vectors=True,
                )

            if not query_vector_data:
                st.error(f"Error: No 'intended_use' vector found for {k_number}.")
                st.info("Please make sure ingestion was successful.")
                return

            query_vector = query_vector_data[0].vector
            st.success(f"Found seed vector for {k_number}. Finding similar devices...")

            with st.spinner("Finding recommendations..."):
                recommendations = qdrant_client.recommend(
                    collection_name=get_config("QDRANT_COLLECTION"),
                    positive=[query_vector],
                    limit=25,
                    query_filter=models.Filter(
                        must_not=[
                            models.FieldCondition(key="k_number", match=models.MatchValue(value=k_number))
                        ]
                    ),
                )

            top_unique_hits = []
            seen_k_numbers = set()
            for hit in recommendations:
                kn = hit.payload["k_number"]
                if kn not in seen_k_numbers:
                    top_unique_hits.append(hit)
                    seen_k_numbers.add(kn)
                if len(top_unique_hits) >= get_config("RERANKED_TOP_N"):
                    break

            st.markdown("---")
            st.subheader(f"Query Document: {query_doc['k_number']}")
            st.markdown(f"**{query_doc['device_name']}** by *{query_doc['manufacturer']}*")
            with st.container(height=200):
                st.markdown(query_doc.get("section_indications_for_use", query_doc.get("raw_text_summary", "")))
            st.markdown("---")

            st.subheader(f"Found {len(top_unique_hits)} Similar Devices")

            QUERY_HIGHLIGHT_COLOR = "rgba(59, 130, 246, 0.2)"
            RESULT_HIGHLIGHT_COLOR = "rgba(34, 197, 94, 0.2)"
            LEGEND_STYLE = "display: inline-block; padding: 2px 6px; border: 1px solid #888; border-radius: 4px; margin-right: 10px; font-weight: 500;"

            legend_html = f"""
            <div style="font-size: 0.9em; color: #888;">
              <b>Legend:</b> 
              <span style="{LEGEND_STYLE} background-color: {QUERY_HIGHLIGHT_COLOR};">Query Highlight</span>
              <span style="{LEGEND_STYLE} background-color: {RESULT_HIGHLIGHT_COLOR};">Result Highlight</span>
            </div>
            """

            for i, hit in enumerate(top_unique_hits):
                payload = hit.payload

                result_doc = fetch_one_dict(
                    db_conn,
                    "SELECT * FROM device_metadata WHERE k_number = ?",
                    (payload["k_number"],),
                )

                st.markdown("---")
                st.subheader(f"Rank {i+1}: {payload['k_number']} - {payload['device_name']} (Score: {hit.score:.4f})")

                with st.spinner("Agent is analyzing similarity..."):
                    justification = get_similarity_justification(query_doc, result_doc)

                if justification.get("is_similar", False):
                    st.info(f"**Similarity Justification:** {justification.get('justification', 'N/A')}")
                else:
                    st.warning(f"**Similarity Justification:** {justification.get('justification', 'N/A')}")

                st.markdown(legend_html, unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Query: {query_doc['k_number']}**")
                    query_text = query_doc.get("section_indications_for_use", query_doc.get("raw_text_summary", ""))
                    st.markdown(
                        highlight_text(query_text, justification.get("query_doc_highlights", []), QUERY_HIGHLIGHT_COLOR),
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown(f"**Result: {result_doc['k_number']}**")
                    result_text = result_doc.get("section_indications_for_use", result_doc.get("raw_text_summary", ""))
                    st.markdown(
                        highlight_text(result_text, justification.get("result_doc_highlights", []), RESULT_HIGHLIGHT_COLOR),
                        unsafe_allow_html=True,
                    )

        except Exception as e:
            st.error(f"An error occurred during retrieval: {e}")
            return

# --- Main App ---
def main():
    st.sidebar.image(LOGO_PATH, width=200, use_column_width=True)
    st.sidebar.title("FormlyAI 510(k)")

    user = st.session_state.user
    if not user:
        st.session_state.page = "login"

    if st.session_state.page == "login":
        st.header("Sign In")
        email = st.text_input("Email:")
        password = st.text_input("Password:", type="password")
        if st.button("Sign In"):
            st.session_state.user = sign_in_user(email, password)
            st.session_state.page = "workspace"
            st.rerun()
        st.markdown("---")
        st.subheader("Or Sign Up")
        email_su = st.text_input("Sign Up Email:", key="su_email")
        password_su = st.text_input("Sign Up Password:", type="password", key="su_pass")
        if st.button("Sign Up"):
            sign_up_user(email_su, password_su)
            st.success("Sign Up successful. Please log in.")

        return

    st.header(f"Welcome {st.session_state.user['email']}")

    st.session_state.api_keys = get_user_api_keys(st.session_state.user["id"])

    keys_valid = (
        st.session_state.api_keys
        and st.session_state.api_keys.get("google")
        and st.session_state.api_keys.get("cohere")
        and st.session_state.api_keys.get("tavily")
        and st.session_state.api_keys.get("owm")
    )

    if not keys_valid:
        st.warning("Please enter all 4 API keys in the 'My API Keys' section to load workspace.")
        return

    # --- Automatically Initialize Workspace ---
    if not st.session_state.workspace_loaded:
        with st.spinner("Initializing services..."):
            try:
                keys = st.session_state.api_keys

                st.session_state.llm = init_llm(st.session_state.selected_model, keys["google"])
                st.session_state.embeddings = init_embeddings()
                st.session_state.reranker = init_reranker(keys["cohere"])
                st.session_state.internet_search = init_internet_search(keys["tavily"])
                st.session_state.db_conn = init_db_connection(read_only=True)
                st.session_state.qdrant_client = init_qdrant_client()

                if st.session_state.selected_mode == "Chatbot (Agentic)":
                    build_dynamic_agent(["local_510k_search", "internet_search", "get_weather"])

                st.session_state.workspace_loaded = True
                st.success("Workspace loaded successfully.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to initialize services: {e}")
                st.exception(e)
                return

    # --- Mode Selector ---
    mode = st.session_state.selected_mode
    if mode == "Chatbot (Agentic)":
        st.subheader("Chatbot Mode")
        st.markdown("Start a new chat session below.")
        chat_input = st.text_input("Ask me about 510(k) devices or regulatory guidance:")
        if st.button("Send"):
            st.session_state.agent_executor(chat_input)
    elif mode == "Retrieval (Find Similar)":
        run_retrieval_mode_ui()

if __name__ == "__main__":
    main()
