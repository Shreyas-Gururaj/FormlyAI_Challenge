import streamlit as st
import json
import uuid
import time
import re
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
    get_qdrant_client
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
    supabase # Import supabase client
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
    initial_sidebar_state="expanded"
)

# --- (MOCKED) Company Logo ---
LOGO_URL = "https://placehold.co/300x100/000000/FFFFFF?text=FormlyAI&font=inter"

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
    is_similar: bool = Field(description="True if the documents are relevantly similar, False otherwise.")
    justification: str = Field(description="A 1-2 sentence explanation of *why* they are similar, focusing on indications or function.")
    query_doc_highlights: List[str] = Field(description="A list of 1-3 key phrases from the Query Document that justify the similarity.")
    result_doc_highlights: List[str] = Field(description="A list of 1-3 key phrases from the Result Document that justify the similarity.")


# --- Initialize Session State ---
# This is Streamlit's "memory"
if "workspace_loaded" not in st.session_state:
    st.session_state.workspace_loaded = False
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = "Chatbot (Agentic)"
if "justification_agent" not in st.session_state:
    st.session_state.justification_agent = None
if "selected_session_id" not in st.session_state:
    st.session_state.selected_session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_keys" not in st.session_state:
    st.session_state.api_keys = None # Will be loaded from Supabase
if "user" not in st.session_state:
    st.session_state.user = None # Will store the Supabase user object
if "page" not in st.session_state:
    st.session_state.page = "login" # Start on the login page


# --- Dynamic Tool: 510(k) RAG Search Tool ---
# This is the "Tool-as-a-Service" concept in action.
@tool
def local_510k_search(query: str) -> str:
    """
    Searches the local 510(k) database for medical devices, manufacturers,
    or specific K-Numbers. Use this for any query about a medical device.
    Returns a JSON string of the most relevant documents.
    """
    # Find the expander created in the main UI thread
    workspace_expander = st.session_state.get("workspace_expander")
    if not workspace_expander:
        workspace_expander = st.empty() 
        
    workspace_expander.info("Tool: Running Local 510(k) RAG Search...")
    
    # Get services from session state
    llm = st.session_state.llm
    embeddings = st.session_state.embeddings
    reranker = st.session_state.reranker
    db_conn = st.session_state.db_conn
    qdrant_client = st.session_state.qdrant_client
    
    # 1. Planner Agent
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

    # 2. V2 Weighted Vector Search
    query_embedding = embeddings.embed_query(plan['semantic_query'])
    weighted_scores = {}
    doc_metadata = {}

    for section, weight in get_config("SECTION_WEIGHTS").items():
        hits = qdrant_client.search(
            collection_name=get_config("QDRANT_COLLECTION"),
            query_vector=query_embedding,
            limit=get_config("VECTOR_TOP_K"),
            query_filter=models.Filter(
                must=[models.FieldCondition(key="source", match=models.MatchValue(value=section))]
            )
        )
        for hit in hits:
            kn = hit.payload['k_number']
            score = hit.score * weight
            if kn not in weighted_scores or score > weighted_scores[kn]:
                weighted_scores[kn] = score
                doc_metadata[kn] = hit.payload

    # 3. Structured Search (DuckDB)
    sparse_term = f"%{plan['sparse_query']}%"
    structured_results = db_conn.execute(
        """
        SELECT k_number, device_name, manufacturer, raw_text_summary, contact_person
        FROM device_metadata 
        WHERE device_name ILIKE ? OR manufacturer ILIKE ? OR k_number ILIKE ? OR contact_person ILIKE ?
        """,
        (sparse_term, sparse_term, sparse_term, sparse_term)
    ).fetchall()
    
    for row in structured_results:
        kn = row[0]
        if kn not in weighted_scores:
            weighted_scores[kn] = 0.1 
            doc_metadata[kn] = {"k_number": kn, "device_name": row[1], "manufacturer": row[2], "text": row[3], "source": "structured_search"}

    # 4. Combine
    combined_docs = []
    for kn, score in weighted_scores.items():
        payload = doc_metadata[kn]
        combined_docs.append(RetrievedDoc(
            k_number=payload['k_number'], device_name=payload['device_name'],
            manufacturer=payload['manufacturer'], text=payload['text'],
            source=payload['source'], score=score
        ).dict())

    combined_docs = sorted(combined_docs, key=lambda x: x['score'], reverse=True)
    
    if not combined_docs:
        workspace_expander.warning("Local RAG: No documents found.")
        return "No documents found in the 510(k) database."
        
    workspace_expander.success(f"Combined to {len(combined_docs)} unique results.")

    # 5. Rerank with Cohere
    workspace_expander.info("Tool: Running Cohere Re-ranker...")
    docs_to_rerank = [Document(page_content=doc['text'], metadata=doc) for doc in combined_docs]
    
    try:
        reranked_responses = reranker.compress_documents(
            query=plan['semantic_query'], documents=docs_to_rerank
        )
        final_reranked_list = [doc.metadata for doc in reranked_responses]
        workspace_expander.success(f"Cohere reranked to {len(final_reranked_list)} results.")
        
        for i, doc in enumerate(final_reranked_list):
            with workspace_expander.expander(f"Top Result {i+1}: {doc['k_number']} ({doc['device_name']})"):
                st.json(doc, expanded=False)
        
        return json.dumps(final_reranked_list)

    except Exception as e:
        workspace_expander.error(f"Cohere Reranker failed: {e}. Returning top K unsorted.")
        return json.dumps(combined_docs[:get_config("RERANKED_TOP_N")])

# --- Main LangGraph Agent ---
class AgentState(TypedDict):
    messages: List[HumanMessage | AIMessage | ToolMessage]

def build_dynamic_agent(enabled_tools_list: List[str]):
    """
    Builds the agent on the fly based on the tools selected in the UI.
    """
    
    # 1. Create the Tool Registry
    tool_registry = {
        "local_510k_search": local_510k_search,
        "internet_search": st.session_state.internet_search,
        "get_weather": get_weather_tool(st.session_state.api_keys["owm"]) # Pass key directly
    }
    
    # 2. Select only the tools the user enabled
    tools = [tool_registry[tool_name] for tool_name in enabled_tools_list if tool_name in tool_registry]
    tool_names = [tool.name for tool in tools]
    
    if not tools:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. You have no tools."),
            ("user", "{query}"),
            ("placeholder", "{chat_history}")
        ])
        st.session_state.agent_executor = prompt | st.session_state.llm | StrOutputParser()
        return

    # 3. Create the Agent
    prompt_text = get_prompt("main_agent_prompt")
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text.format(tool_names=tool_names)), 
        ("placeholder", "{chat_history}"),
        ("user", "{query}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    agent = create_tool_calling_agent(st.session_state.llm, tools, prompt)
    
    # 4. Create the Graph Executor
    st.session_state.agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True 
    )

# --- UI Mode: Retrieval ---

def fetch_one_dict(db_conn, query: str, params: tuple) -> Optional[dict]:
    """
    Executes a query, fetches one row, and returns it as a dictionary.
    """
    cursor = db_conn.execute(query, params)
    result = cursor.fetchone()
    if not result:
        return None
    cols = [desc[0] for desc in cursor.description]
    return dict(zip(cols, result))

def get_similarity_justification(query_doc: dict, result_doc: dict) -> SimilarityJustification:
    """
    Calls a specialized agent to explain *why* two documents are similar.
    """
    try:
        if "justification_agent" not in st.session_state or st.session_state.justification_agent is None:
            prompt_text = get_prompt("justification_prompt")
            prompt = ChatPromptTemplate.from_template(prompt_text)
            parser = JsonOutputParser(pydantic_model=SimilarityJustification)
            # Use the main session LLM for the justification agent
            st.session_state.justification_agent = prompt | st.session_state.llm | parser
        
        query_snippet = f"Device: {query_doc['device_name']}\nIndications: {query_doc.get('section_indications_for_use', '')}"
        result_snippet = f"Device: {result_doc['device_name']}\nIndications: {result_doc.get('section_indications_for_use', '')}"

        justification = st.session_state.justification_agent.invoke({
            "query_doc": query_snippet,
            "result_doc": result_snippet,
            "schema": SimilarityJustification.schema_json()
        })
        return justification
    except Exception as e:
        st.error(f"Justification agent failed: {e}")
        return SimilarityJustification(
            is_similar=False, 
            justification="Could not determine similarity.", 
            query_doc_highlights=[], 
            result_doc_highlights=[]
        ).dict()

def highlight_text(text: str, highlights: List[str], color: str) -> str:
    """
    Wraps phrases in a markdown-safe <span> tag with a specific color.
    Uses regex to be case-insensitive and safe.
    """
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
    """Renders the UI for the 'Find Similar' mode."""
    st.header("Retrieval Mode: Find Similar Devices")
    st.markdown("This mode finds the most semantically similar devices to a given K-Number, using its vector embedding.")
    
    k_number = st.text_input("Enter a K-Number (e.g., K000009):").strip().upper()
    
    if st.button("Find Similar"):
        if not k_number:
            st.error("Please enter a K-Number.")
            return

        qdrant_client = st.session_state.qdrant_client
        db_conn = st.session_state.db_conn
        
        try:
            # 1. Get the Query Document
            with st.spinner(f"Fetching query document: {k_number}..."):
                query_doc = fetch_one_dict(
                    db_conn,
                    "SELECT * FROM device_metadata WHERE k_number = ?", 
                    (k_number,)
                )

            if not query_doc:
                st.error(f"Error: K-Number {k_number} not found in our database.")
                return

            # 2. Get the Seed Vector
            with st.spinner(f"Searching for seed vector..."):
                query_vector_data, _ = qdrant_client.scroll(
                    collection_name=get_config("QDRANT_COLLECTION"),
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(key="k_number", match=models.MatchValue(value=k_number)),
                            models.FieldCondition(key="source", match=models.MatchValue(value="intended_use"))
                        ]
                    ),
                    limit=1,
                    with_vectors=True
                )
            
            if not query_vector_data:
                st.error(f"Error: No 'intended_use' vector found for {k_number}.")
                st.info("Please make sure ingestion was successful.")
                return

            query_vector = query_vector_data[0].vector
            st.success(f"Found seed vector for {k_number}. Finding similar devices...")

            # 3. Use Qdrant's `recommend` feature
            with st.spinner("Finding recommendations..."):
                recommendations = qdrant_client.recommend(
                    collection_name=get_config("QDRANT_COLLECTION"),
                    positive=[query_vector], 
                    limit=25, 
                    query_filter=models.Filter(
                        must_not=[ 
                            models.FieldCondition(key="k_number", match=models.MatchValue(value=k_number))
                        ]
                    )
                )

            # 4. De-duplicate the results
            top_unique_hits = []
            seen_k_numbers = set()
            for hit in recommendations:
                kn = hit.payload['k_number']
                if kn not in seen_k_numbers:
                    top_unique_hits.append(hit)
                    seen_k_numbers.add(kn)
                if len(top_unique_hits) >= get_config("RERANKED_TOP_N"):
                    break
            
            st.subheader(f"Top {len(top_unique_hits)} Similar Devices")

            # 5. Display in Split-Screen
            QUERY_HIGHLIGHT_COLOR = "rgba(59, 130, 246, 0.2)"  # Blue-200
            RESULT_HIGHLIGHT_COLOR = "rgba(34, 197, 94, 0.2)" # Green-200
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
                    (payload['k_number'],)
                )

                st.markdown("---")
                st.subheader(f"Rank {i+1}: {payload['k_number']} - {payload['device_name']} (Score: {hit.score:.4f})")

                with st.spinner("Agent is analyzing similarity..."):
                    justification = get_similarity_justification(query_doc, result_doc)
                
                if justification.get('is_similar', False):
                    st.info(f"**Similarity Justification:** {justification.get('justification', 'N/A')}")
                else:
                    st.warning(f"**Similarity Justification:** {justification.get('justification', 'N/A')}")

                st.markdown(legend_html, unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Query: {query_doc['k_number']}**")
                    query_text = query_doc.get('section_indications_for_use', query_doc.get('raw_text_summary', ''))
                    st.markdown(
                        highlight_text(query_text, justification.get('query_doc_highlights', []), QUERY_HIGHLIGHT_COLOR), 
                        unsafe_allow_html=True
                    )
                
                with col2:
                    st.markdown(f"**Result: {result_doc['k_number']}**")
                    result_text = result_doc.get('section_indications_for_use', result_doc.get('raw_text_summary', ''))
                    st.markdown(
                        highlight_text(result_text, justification.get('result_doc_highlights', []), RESULT_HIGHLIGHT_COLOR),
                        unsafe_allow_html=True
                    )

        except Exception as e:
            st.error(f"An error occurred during retrieval: {e}")
            st.exception(e)


# --- UI Mode: Chatbot ---
def run_chatbot_mode_ui():
    """Renders the UI for the 'Chatbot' mode."""
    st.header(f"Chatbot Mode (Model: `{st.session_state.selected_model}`)")
    
    # Display the chat history
    for msg_type, content in st.session_state.chat_history:
        if msg_type == "HumanMessage":
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        elif msg_type == "AIMessage":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)

    # Get new user input
    if prompt := st.chat_input("Ask a question..."):
        
        # Add and save user message
        st.session_state.chat_history.append(("HumanMessage", prompt))
        save_chat_message(st.session_state.user.id, st.session_state.selected_session_id, "HumanMessage", prompt)
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # This is the main "workspace"
        with st.expander("🤖 Agent Workspace (Live)"):
            st.session_state.workspace_expander = st.container()
            
        # This is the streaming part
        with st.chat_message("assistant", avatar="🤖"):
            
            stream_placeholder = st.empty()
            final_response = ""
            
            try:
                # Format chat history for the agent
                chat_history_for_agent = []
                for msg_type, content in st.session_state.chat_history[:-1]:
                    if msg_type == "HumanMessage":
                        chat_history_for_agent.append(HumanMessage(content=content))
                    elif msg_type == "AIMessage":
                        chat_history_for_agent.append(AIMessage(content=content))

                with st.spinner("Agent is thinking..."):
                    for event in st.session_state.agent_executor.stream({
                        "chat_history": chat_history_for_agent,
                        "query": prompt
                    }):
                        if "actions" in event:
                            for action in event["actions"]:
                                st.session_state.workspace_expander.warning(f"🤖 Calling Tool: `{action.tool}`")
                                st.session_state.workspace_expander.json(action.tool_input)
                        elif "steps" in event:
                            for step in event["steps"]:
                                st.session_state.workspace_expander.success(f"✅ Tool Result ({step.action.tool}):")
                                try:
                                    obs_json = json.loads(step.observation)
                                    st.session_state.workspace_expander.json(obs_json)
                                except Exception:
                                    st.session_state.workspace_expander.text(step.observation)
                        elif "output" in event:
                            final_response += event["output"]
                            stream_placeholder.markdown(final_response + "▌")
                        
                stream_placeholder.markdown(final_response)
            
            except Exception as e:
                final_response = f"An error occurred during the agent run: {e}"
                st.error(final_response)
                st.exception(e)

            # Save the final message to history
            st.session_state.chat_history.append(("AIMessage", final_response))
            save_chat_message(st.session_state.user.id, st.session_state.selected_session_id, "AIMessage", final_response)

# --- Main App Logic ---

# --- 1. Authentication (NEW: Pure Supabase) ---
# This is now the main app router
if "user" not in st.session_state:
     st.session_state.user = get_current_user()

# User is not logged in, show login/register page
if st.session_state.user is None:
    st.image(LOGO_URL, width=300)
    st.title("Welcome to the FormlyAI Agent Demo")
    
    login_tab, register_tab = st.tabs(["Login", "Register"])
    
    with login_tab:
        with st.form("Login"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                user = sign_in_user(email, password)
                if user:
                    st.session_state.user = user
                    st.rerun()

    with register_tab:
        with st.form("Register"):
            full_name = st.text_input("Full Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Register"):
                sign_up_user(email, password, full_name)

# --- 2. Main Application (If Logged In) ---
else:
    # Get user's name from their metadata
    user_name = st.session_state.user.user_metadata.get("full_name", "User")
    
    with st.sidebar:
        st.image(LOGO_URL)
        st.title(f"Welcome, {user_name}!")
        if st.button("Logout"):
            sign_out_user()
            st.session_state.user = None
            st.rerun()
        
        st.markdown("---")
        
        # --- API Key Management (Request #1) ---
        with st.expander("🔐 My API Keys", expanded=False):
            st.info("Your keys are encrypted and saved securely to your user profile.", icon="ℹ️")
            if st.session_state.api_keys is None:
                st.session_state.api_keys = get_user_api_keys(st.session_state.user.id) or {}

            google_key = st.text_input("Google API Key", type="password", value=st.session_state.api_keys.get("google", ""))
            cohere_key = st.text_input("Cohere API Key", type="password", value=st.session_state.api_keys.get("cohere", ""))
            tavily_key = st.text_input("Tavily API Key", type="password", value=st.session_state.api_keys.get("tavily", ""))
            owm_key = st.text_input("OpenWeatherMap API Key", type="password", value=st.session_state.api_keys.get("owm", ""))

            if st.button("Save API Keys"):
                keys_to_save = {
                    "google": google_key,
                    "cohere": cohere_key,
                    "tavily": tavily_key,
                    "owm": owm_key
                }
                save_user_api_keys(st.session_state.user.id, keys_to_save)
                st.session_state.api_keys = keys_to_save
        
        st.markdown("---")
        
        # --- Workspace Configuration ---
        st.header("Workspace Configuration")

        # Check if keys are valid before proceeding
        keys_valid = (
            st.session_state.api_keys and
            st.session_state.api_keys.get("google") and
            st.session_state.api_keys.get("cohere") and
            st.session_state.api_keys.get("tavily") and
            st.session_state.api_keys.get("owm")
        )

        if not keys_valid:
            st.warning("Please enter all 4 API keys in the 'My API Keys' section to load a workspace.")
        else:
            # --- Mode Selection ---
            st.session_state.selected_mode = st.selectbox(
                "Select Mode",
                ["Chatbot (Agentic)", "Retrieval (Find Similar)"]
            )
            
            # --- Model Selection ---
            st.session_state.selected_model = st.selectbox(
                "Select Language Model",
                ["gemini-pro", "gemini-1.5-flash"]
            )

            # --- Tool Selection (MCP Servers) ---
            st.session_state.selected_tools = st.multiselect(
                "Select Tools (for Chatbot Mode)",
                ["local_510k_search", "internet_search", "get_weather"],
                default=["local_510k_search", "internet_search", "get_weather"]
            )

            if st.button("Load Workspace"):
                with st.spinner("Initializing services..."):
                    try:
                        # Initialize services using the user's saved keys
                        keys = st.session_state.api_keys
                        st.session_state.llm = get_llm(st.session_state.selected_model, keys["google"])
                        st.session_state.embeddings = get_embeddings()
                        st.session_state.reranker = get_reranker(keys["cohere"])
                        st.session_state.internet_search = get_tavily_search(keys["tavily"])
                        # Weather tool is initialized dynamically by the agent
                        
                        st.session_state.db_conn = get_db_connection(read_only=True)
                        st.session_state.qdrant_client = get_qdrant_client()
                        
                        if st.session_state.selected_mode == "Chatbot (Agentic)":
                            build_dynamic_agent(st.session_state.selected_tools)

                        st.session_state.workspace_loaded = True
                        st.success("Workspace loaded successfully.")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to initialize services: {e}")
                        st.exception(e)

        # --- Chat History (Request #2) ---
        if st.session_state.workspace_loaded and st.session_state.selected_mode == "Chatbot (Agentic)":
            st.markdown("---")
            st.header("Chat History")
            
            session_list = get_session_list(st.session_state.user.id)
            
            if st.button("New Chat ➕", use_container_width=True):
                st.session_state.selected_session_id = f"session_{uuid.uuid4()}"
                st.session_state.chat_history = []
                st.rerun()
                
            session_titles = {s_id: s_id.replace("session_", "Chat ")[:12] for s_id in session_list}
            
            # Load default session if none selected
            if not st.session_state.selected_session_id and session_list:
                st.session_state.selected_session_id = session_list[0]
            elif not st.session_state.selected_session_id and not session_list:
                 st.session_state.selected_session_id = f"session_{uuid.uuid4()}"
                 st.session_state.chat_history = []

            
            # Use a radio button to select the session
            # Check if the current session_id is valid, if not, default to the first or new
            try:
                current_index = session_list.index(st.session_state.selected_session_id)
            except ValueError:
                if session_list:
                    st.session_state.selected_session_id = session_list[0]
                    current_index = 0
                else:
                    current_index = 0 # Will be empty list
            
            selected = st.radio(
                "Select a conversation:",
                session_list,
                format_func=lambda s_id: session_titles.get(s_id, "New Chat"),
                key="session_radio",
                label_visibility="collapsed",
                index=current_index
            )
            
            # Load the selected chat history
            if selected and st.session_state.selected_session_id != selected:
                st.session_state.selected_session_id = selected
                st.session_state.chat_history = get_chat_history(st.session_state.user.id, selected)
                st.rerun()
            
            # Load history for the first time if it's empty
            if selected and not st.session_state.chat_history:
                 st.session_state.chat_history = get_chat_history(st.session_state.user.id, selected)


    # --- 3. Main App Interface (If Workspace is Loaded) ---
    if st.session_state.workspace_loaded:
        if st.session_state.selected_mode == "Chatbot (Agentic)":
            run_chatbot_mode_ui()
        else:
            run_retrieval_mode_ui()
    elif st.session_state.user:
        # User is logged in, but hasn't loaded a workspace
        st.title(f"Welcome, {name}!")
        st.info("Please configure your API keys and workspace in the sidebar, then click 'Load Workspace' to begin.")