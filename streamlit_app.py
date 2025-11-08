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

# --- Pydantic Schemas ---
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
        description="A 1-2 sentence explanation of why they are similar."
    )
    query_doc_highlights: List[str] = Field(
        description="1-3 key phrases from the Query Document."
    )
    result_doc_highlights: List[str] = Field(
        description="1-3 key phrases from the Result Document."
    )

# --- Session State Defaults ---
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

# --- Helper Utilities ---
def clear_user_session_state():
    keys_to_keep = set(_initial_keys.keys())
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]

# --- Dynamic Tool: 510(k) RAG Search Tool ---
@tool
def local_510k_search(query: str) -> str:
    workspace_expander = st.session_state.get("workspace_expander") or st.empty()
    workspace_expander.info("Tool: Running Local 510(k) RAG Search...")

    llm = st.session_state.get("llm")
    embeddings = st.session_state.get("embeddings")
    reranker = st.session_state.get("reranker")
    db_conn = st.session_state.get("db_conn")
    qdrant_client = st.session_state.get("qdrant_client")

    if not all([llm, embeddings, reranker, db_conn, qdrant_client]):
        workspace_expander.error("Some services are not initialized.")
        return json.dumps([])

    # Planner Agent
    planner_prompt_text = get_prompt("planner_prompt")
    planner_prompt = ChatPromptTemplate.from_template(planner_prompt_text)
    parser = JsonOutputParser(pydantic_model=QueryPlan)
    planner = planner_prompt | llm | parser
    try:
        plan = planner.invoke({"query": query, "schema": QueryPlan.schema_json()})
    except Exception as e:
        workspace_expander.error(f"Query Planner failed: {e}")
        plan = {"semantic_query": query, "sparse_query": query, "filters": {}}

    workspace_expander.json(plan, expanded=False)

    # Weighted Vector Search
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

    # Structured Search (DuckDB)
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

    combined_docs = [
        RetrievedDoc(
            k_number=payload["k_number"],
            device_name=payload["device_name"],
            manufacturer=payload["manufacturer"],
            text=payload["text"],
            source=payload["source"],
            score=score,
        ).dict()
        for kn, score in weighted_scores.items()
        for payload in [doc_metadata[kn]]
    ]

    combined_docs = sorted(combined_docs, key=lambda x: x["score"], reverse=True)
    if not combined_docs:
        workspace_expander.warning("No documents found.")
        return "No documents found."

    workspace_expander.success(f"Combined {len(combined_docs)} results.")

    # Re-rank
    workspace_expander.info("Running Cohere Re-ranker...")
    docs_to_rerank = [Document(page_content=doc["text"], metadata=doc) for doc in combined_docs]

    try:
        reranked_responses = reranker.compress_documents(query=plan["semantic_query"], documents=docs_to_rerank)
        final_reranked_list = [doc.metadata for doc in reranked_responses]
        workspace_expander.success(f"Cohere reranked {len(final_reranked_list)} results.")
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
            [("system", "You are a helpful assistant. You have no tools."),
             ("user", "{query}"),
             ("placeholder", "{chat_history}")]
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

# --- Utility: Fetch single dict ---
def fetch_one_dict(db_conn, query: str, params: tuple) -> Optional[dict]:
    cursor = db_conn.execute(query, params)
    result = cursor.fetchone()
    if not result:
        return None
    cols = [desc[0] for desc in cursor.description]
    return dict(zip(cols, result))

# --- Highlighting ---
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
    return re.sub(pattern, replacer, text, flags=re.IGNORECASE).replace("\n", "  \n")

# --- Main App ---
def main():
    # --- 1. Authentication ---
    st.session_state.user = st.session_state.user or get_current_user()
    if st.session_state.user is None:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=300)
        else:
            st.image(LOGO_URL_FALLBACK, width=300)
        st.title("Welcome to FormlyAI Agent Demo")
        login_tab, register_tab = st.tabs(["Login", "Register"])
        with login_tab:
            with st.form("Login"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
                    user = sign_in_user(email, password)
                    if user:
                        st.session_state.user = user
                        st.success("Successfully logged in!")
                        st.experimental_rerun()
        with register_tab:
            with st.form("Register"):
                full_name = st.text_input("Full Name")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Register"):
                    user = sign_up_user(email, password, full_name)
                    if user:
                        st.session_state.user = user
                        st.success("Registration successful!")
                        st.experimental_rerun()
        return

    user_name = st.session_state.user.user_metadata.get("full_name", "User")

    # --- Sidebar ---
    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH)
        else:
            st.image(LOGO_URL_FALLBACK)
        st.title(f"Welcome, {user_name}!")
        if st.button("Logout"):
            try:
                sign_out_user()
            finally:
                st.session_state.clear()
                st.experimental_rerun()

        st.markdown("---")
        # API Keys
        with st.expander("🔐 My API Keys", expanded=False):
            if st.session_state.api_keys is None:
                st.session_state.api_keys = get_user_api_keys(st.session_state.user.id) or {}
            google_key = st.text_input("Google API Key", type="password", value=st.session_state.api_keys.get("google", ""))
            cohere_key = st.text_input("Cohere API Key", type="password", value=st.session_state.api_keys.get("cohere", ""))
            tavily_key = st.text_input("Tavily API Key", type="password", value=st.session_state.api_keys.get("tavily", ""))
            owm_key = st.text_input("OpenWeatherMap API Key", type="password", value=st.session_state.api_keys.get("owm", ""))
            if st.button("Save API Keys"):
                keys_to_save = {"google": google_key, "cohere": cohere_key, "tavily": tavily_key, "owm": owm_key}
                save_user_api_keys(st.session_state.user.id, keys_to_save)
                st.session_state.api_keys = keys_to_save
                st.success("API Keys saved!")
                st.experimental_rerun()

        st.markdown("---")
        # Debug Mode
        with st.expander("⚙️ Debug Mode", expanded=False):
            debug_mode = st.checkbox("Enable debug mode", value=False)
            st.session_state.debug_mode = debug_mode

        if st.session_state.get("debug_mode", False):
            st.markdown("### Debug Info")
            st.json({
                "user": str(st.session_state.user),
                "api_keys": {k: "****" for k in (st.session_state.api_keys or {})},
                "workspace_loaded": st.session_state.workspace_loaded,
                "selected_mode": st.session_state.selected_mode,
                "chat_history_len": len(st.session_state.chat_history or []),
            })

        st.markdown("---")
        # Mode and Model selection
        st.session_state.selected_mode = st.selectbox(
            "Select Mode", ["Chatbot (Agentic)", "Retrieval (Find Similar)"]
        )
        st.session_state.selected_model = st.selectbox(
            "Select Language Model", ["gemini-2.5-flash", "gemini-2.5-pro"]
        )
        st.session_state.selected_tools = ["local_510k_search", "internet_search", "get_weather"]

    # --- Auto-load workspace if API keys valid ---
    keys = st.session_state.api_keys
    if keys and all(keys.get(k) for k in ["google", "cohere", "tavily", "owm"]) and not st.session_state.workspace_loaded:
        try:
            st.session_state.llm = init_llm(st.session_state.selected_model, keys["google"])
            st.session_state.embeddings = init_embeddings()
            st.session_state.reranker = init_reranker(keys["cohere"])
            st.session_state.internet_search = init_internet_search(keys["tavily"])
            st.session_state.db_conn = init_db_connection(read_only=True)
            st.session_state.qdrant_client = init_qdrant_client()
            if st.session_state.selected_mode == "Chatbot (Agentic)":
                build_dynamic_agent(st.session_state.selected_tools)
            st.session_state.workspace_loaded = True
        except Exception as e:
            st.error(f"Workspace auto-load failed: {e}")
            st.exception(e)
            return

    # --- Main Interface ---
    if st.session_state.workspace_loaded:
        if st.session_state.selected_mode == "Chatbot (Agentic)":
            # Always start a new session for chatbot mode
            st.session_state.selected_session_id = f"session_{uuid.uuid4()}"
            st.session_state.chat_history = []
            from streamlit_chat_ui import run_chatbot_mode_ui  # Assume same function
            run_chatbot_mode_ui()
        else:
            from streamlit_chat_ui import run_retrieval_mode_ui  # Assume same function
            run_retrieval_mode_ui()

if __name__ == "__main__":
    main()
