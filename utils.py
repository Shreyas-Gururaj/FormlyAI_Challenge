import os
import yaml
import duckdb
import qdrant_client
import cohere
import streamlit as st
from rich.console import Console
from typing import Any # Added for default value typing
from functools import lru_cache

# Import LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cohere import CohereRerank
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper
from langchain_core.tools import tool

# --- 1. Initialize Console ---
console = Console()

# --- 2. Load Configuration Files (NEW: Environment-Aware) ---

# --- Load Local config.yaml (if it exists) for local development ---
LOCAL_CONFIG = {}
try:
    with open("config.yaml", "r") as f:
        LOCAL_CONFIG = yaml.safe_load(f)
        console.log("[dim]Loaded local `config.yaml` file.[/dim]")
except FileNotFoundError:
    console.log("[dim]No local `config.yaml` found. Assuming deployment and using Streamlit Secrets.[/dim]")
    # This is normal and expected when deployed
    pass
except Exception as e:
    console.log(f"[yellow]Warning: Could not read `config.yaml`: {e}[/yellow]")

# --- Load Prompts File (Must exist in the repo) ---
PROMPTS_FILE = LOCAL_CONFIG.get("PROMPTS_FILE", "prompts.yaml")
try:
    with open(PROMPTS_FILE, "r") as f:
        prompts = yaml.safe_load(f)
except FileNotFoundError:
    console.log(f"[bold red]FATAL: Prompts file `{PROMPTS_FILE}` not found.[/bold red]")
    st.error(f"FATAL: Prompts file `{PROMPTS_FILE}` not found. Please make sure it's in your GitHub repo.")
    exit(1)
except Exception as e:
    console.log(f"[bold red]FATAL: Error reading `{PROMPTS_FILE}`: {e}[/bold red]")
    st.error(f"FATAL: Error reading `{PROMPTS_FILE}`: {e}")
    exit(1)


# --- 3. Getters for Configs and Prompts (NEW: Environment-Aware) ---

def get_config(key: str, default: Any = None) -> Any:
    """
    Safely gets a config value.
    Priority is:
    1. Streamlit Secrets (for cloud deployment)
    2. Local config.yaml (for local development)
    3. Default value
    """
    
    # 1. Try Streamlit Secrets first (for deployment)
    # This checks if st.secrets exists AND the key is in it
    if hasattr(st, 'secrets') and key in st.secrets:
        return st.secrets[key]
    
    # 2. Fallback to local config.yaml (for local dev)
    value = LOCAL_CONFIG.get(key, default)
    
    # Special case for DATA_DIR
    if key == "DATA_DIR":
        real_path = get_config("REAL_CORPUS_DIR", "./510k_corpus") # Use get_config to check secrets
        mock_path = get_config("MOCK_FILE_DIR", "./mock_510k_files") # Use get_config to check secrets
        if os.path.exists(real_path):
            return real_path
        else:
            return mock_path
            
    return value


def get_prompt(key: str) -> str:
    """Get a prompt string from the prompts file."""
    prompt = prompts.get(key)
    if not prompt:
        console.log(f"[bold red]FATAL: Prompt key `{key}` not found in `{PROMPTS_FILE}`.[/bold red]")
        st.error(f"FATAL: Prompt key `{key}` not found in `{PROMPTS_FILE}`.")
        exit(1)
    return prompt

# --- 4. Service Factories (with Caching) ---
# These functions initialize and cache our services.
# They are called by Streamlit when the user "Loads Workspace".

console.log("[dim]Initializing services...[/dim]")

@st.cache_resource(max_entries=2) # Cache the last 2 models
def get_llm(model_name: str, google_api_key: str) -> ChatGoogleGenerativeAI:
    """Factory for getting a Gemini LLM instance."""
    console.log(f"[dim]Initializing LLM: {model_name}[/dim]")
    if not google_api_key:
        raise ValueError("Google API Key is not set.")
    return ChatGoogleGenerativeAI(
        model=model_name, 
        temperature=0, 
        google_api_key=google_api_key
    )

@st.cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    """Factory for getting the embedding model."""
    console.log("[dim]Initializing Embedding Model...[/dim]")
    return HuggingFaceEmbeddings(
        model_name=get_config("EMBEDDING_MODEL")
    )

@st.cache_resource
def get_reranker(cohere_api_key: str) -> CohereRerank:
    """Factory for getting the Cohere Reranker."""
    console.log("[dim]Initializing Cohere Reranker...[/dim]")
    if not cohere_api_key:
        raise ValueError("Cohere API Key is not set.")
    return CohereRerank(
        model=get_config("COHERE_RERANK_MODEL"), 
        cohere_api_key=cohere_api_key, 
        top_n=get_config("RERANKED_TOP_N")
    )

@st.cache_resource
def get_tavily_search(tavily_api_key: str) -> TavilySearchResults:
    """Factory for getting the Tavily Search tool."""
    console.log("[dim]Initializing Tavily Search...[/dim]")
    if not tavily_api_key:
        raise ValueError("Tavily API Key is not set.")
    os.environ["TAVILY_API_KEY"] = tavily_api_key
    return TavilySearchResults(max_results=3)

@tool
def get_weather_tool(owm_api_key: str):
    """
    A tool to get the current weather for a specific location.
    Input must be a string (e.g., "Boston, MA").
    This function returns the actual tool object.
    """
    console.log("[dim]Initializing OpenWeatherMap Wrapper...[/dim]")
    if not owm_api_key:
        return "Weather API key is not configured."
    
    wrapper = OpenWeatherMapAPIWrapper(openweathermap_api_key=owm_api_key)
    
    # We return the .run() method as the tool function
    @tool
    def get_weather(location: str):
        """
        A tool to get the current weather for a specific location.
        Input must be a string (e.g., "Boston, MA").
        """
        console.log(f"[dim]Tool: Calling OpenWeatherMap for {location}[/dim]")
        try:
            return wrapper.run(location)
        except Exception as e:
            return f"Error getting weather: {e}"
            
    return get_weather

# --- 5. Database Connectors ---
@st.cache_resource
def get_db_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Initializes the DuckDB connection."""
    try:
        return duckdb.connect(database=get_config("DB_FILE"), read_only=read_only)
    except Exception as e:
        console.log(f"[bold red]FATAL: Could not connect to DuckDB at `{get_config('DB_FILE')}`.[/bold red]")
        console.log(f"Error: {e}")
        st.error(f"FATAL: Could not connect to DuckDB at `{get_config('DB_FILE')}`: {e}")
        exit(1)

@st.cache_resource
def get_qdrant_client() -> qdrant_client.QdrantClient:
    """Initializes the Qdrant client."""
    
    # --- NEW CLOUD LOGIC ---
    cloud_url = get_config("QDRANT_CLOUD_URL")
    cloud_key = get_config("QDRANT_CLOUD_KEY")
    
    try:
        if cloud_url and cloud_key and "YOUR_" not in cloud_url:
            console.log("[dim]Connecting to Qdrant Cloud...[/dim]")
            client = qdrant_client.QdrantClient(
                url=cloud_url, 
                api_key=cloud_key,
            )
        else:
            # Fallback to local
            console.log("[dim]Connecting to Qdrant Local...[/dim]")
            client = qdrant_client.QdrantClient(path=get_config("QDRANT_PATH"))
        
        # Test connection
        client.get_collection(collection_name=get_config("QDRANT_COLLECTION"))
        return client
    
    except Exception as e:
        console.log(f"[bold red]FATAL: Could not connect to Qdrant.[/bold red]")
        console.log(f"Error: {e}")
        st.error(f"FATAL: Could not connect to Qdrant: {e}")
        exit(1)

console.log("[dim]Services initialized.[/dim]")