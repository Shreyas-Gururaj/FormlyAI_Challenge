import json
import uuid
import cohere
from rich.panel import Panel
from typing import List, Dict, Any, TypedDict, Optional

# --- NEW: Import services and configs from utils ---
from utils import (
    console, 
    llm, 
    embeddings, 
    reranker,
    internet_search,
    get_config, 
    get_prompt,
    get_db_connection,
    get_qdrant_client
)
# ---

# Import LangChain components
try:
    from langchain_core.pydantic_v1 import BaseModel, Field
except ImportError:
    console.log("[red]Missing core LangChain component.[/red]")
    console.log("Please run `pip install -r requirements.txt`")
    exit(1)

from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END
from qdrant_client import models

# --- Configuration (Loaded from utils) ---
MODE = get_config("MODE", "chatbot").lower()
DB_FILE = get_config("DB_FILE")
QDRANT_PATH = get_config("QDRANT_PATH")
QDRANT_COLLECTION = get_config("QDRANT_COLLECTION")
VECTOR_TOP_K = get_config("VECTOR_TOP_K")
RERANKED_TOP_N = get_config("RERANKED_TOP_N")
SECTION_WEIGHTS = get_config("SECTION_WEIGHTS")

# --- Initialize Global Services (from utils) ---
try:
    # Connect to DBs in read-only mode
    db_conn = get_db_connection(read_only=True)
    qdrant_client = get_qdrant_client()
    # Test connection
    qdrant_client.count(collection_name=QDRANT_COLLECTION)
except Exception as e:
    console.log(f"[red bold]FATAL: Could not connect to databases.[/red]")
    console.log(f"Error: {e}")
    console.log("Please ensure `ingestion.py` has been run successfully first.")
    exit(1)

# --- Pydantic Schemas for Agent IO ---

class QueryPlan(BaseModel):
    """The structured plan for retrieving device information from the 510(k) DB."""
    semantic_query: str = Field(description="A rewritten query optimized for semantic vector search")
    sparse_query: str = Field(description="A 1-5 word keyword query for sparse search (e.g., product codes, manufacturer)")
    filters: Optional[Dict[str, Any]] = Field(description="Key-value metadata filters to apply, e.g., {'manufacturer': 'Alpha Diagnostics'}")

class RetrievedDoc(BaseModel):
    """A single retrieved document before reranking."""
    k_number: str
    device_name: str
    manufacturer: str
    text: str
    source: str
    score: float

# --- NEW: Relevance Check Schema ---
class RelevanceCheck(BaseModel):
    """The result of a relevance check."""
    is_relevant: bool = Field(description="True if the context is relevant to the query, False otherwise.")
    justification: str = Field(description="A brief justification for the decision.")


# --- Tool Definitions ---

def run_v2_hybrid_search(query: str) -> List[Dict[str, Any]]:
    """
    Performs the advanced V2 section-aware hybrid search.
    This is now a tool to be called by the chatbot agent.
    """
    console.log("[cyan]Tool: Running V2 Hybrid Search...[/cyan]")
    
    # 1. Planner Agent (simplified, non-langgraph)
    planner_prompt_text = get_prompt("planner_prompt")
    planner_prompt = PromptTemplate(
        template=planner_prompt_text,
        input_variables=["query"],
        partial_variables={"schema": QueryPlan.schema_json()}
    )
    from langchain.output_parsers.json import SimpleJsonOutputParser
    parser = SimpleJsonOutputParser(pydantic_model=QueryPlan)
    planner = planner_prompt | llm | parser
    
    try:
        plan = planner.invoke({"query": query})
    except Exception as e:
        console.log(f"[red]Query Planner failed: {e}. Using raw query.[/red]")
        plan = {"semantic_query": query, "sparse_query": query, "filters": {}}

    # --- FIX for Off-Topic Queries ---
    # If the planner itself rejects the query, we have no valid search terms.
    if not plan.get("semantic_query") or not plan.get("sparse_query"):
        console.log("[yellow]Planner returned empty queries. Forcing fallback to internet search.[/yellow]")
        return [] # Return an empty list to trigger the fallback

    console.log(Panel(json.dumps(plan, indent=2), title="V2 Search Plan", border_style="cyan"))

    # --- 2. V2 Weighted Vector Search (Qdrant) ---
    query_embedding = embeddings.embed_query(plan['semantic_query'])
    
    weighted_scores = {}
    doc_metadata = {}

    for section, weight in SECTION_WEIGHTS.items():
        try:
            hits = qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=query_embedding,
                limit=VECTOR_TOP_K,
                query_filter=models.Filter(
                    must=[models.FieldCondition(key="source", match=models.MatchValue(value=section))]
                )
            )
            
            for hit in hits:
                kn = hit.payload['k_number']
                score = hit.score * weight
                
                if kn not in weighted_scores or score > weighted_scores[kn]:
                    weighted_scores[kn] = score
                    doc_metadata[kn] = hit.payload # Store the best matching payload
                    
        except Exception as e:
            console.log(f"[yellow]Warning: Vector search failed for section {section}: {e}[/yellow]")

    # --- 3. Structured Search (DuckDB) ---
    try:
        sparse_term = f"%{plan['sparse_query']}%"
        structured_results = db_conn.execute(
            """
            SELECT k_number, device_name, manufacturer, raw_text_summary, contact_person
            FROM device_metadata 
            WHERE 
                device_name ILIKE ? OR 
                manufacturer ILIKE ? OR 
                k_number ILIKE ? OR
                contact_person ILIKE ?
            """,
            (sparse_term, sparse_term, sparse_term, sparse_term)
        ).fetchall()
        
        for row in structured_results:
            kn = row[0]
            if kn not in weighted_scores:
                # Give it a base score if it wasn't in the vector search
                weighted_scores[kn] = 0.1 
                doc_metadata[kn] = {
                    "k_number": kn,
                    "device_name": row[1],
                    "manufacturer": row[2],
                    "text": row[3], # Use full summary as text
                    "source": "structured_search"
                }
    except Exception as e:
        console.log(f"[yellow]Warning: Structured search failed: {e}[/yellow]")

    # --- 4. Combine and Format ---
    combined_docs = []
    for kn, score in weighted_scores.items():
        payload = doc_metadata[kn]
        combined_docs.append(RetrievedDoc(
            k_number=payload['k_number'],
            device_name=payload['device_name'],
            manufacturer=payload['manufacturer'],
            text=payload['text'],
            source=payload['source'],
            score=score
        ).dict()) # Convert to dict

    # Sort by the new weighted score
    combined_docs = sorted(combined_docs, key=lambda x: x['score'], reverse=True)
    
    console.log(f"Combined to {len(combined_docs)} unique results, sorted by weighted score.")
    
    # --- 5. Rerank with Cohere ---
    # Only rerank if we actually found documents
    if not combined_docs:
        return []
        
    console.log("[yellow]Tool: Running Cohere Re-ranker...[/yellow]")
    
    # Convert list of dicts to list of Document objects
    docs_to_rerank = [
        Document(page_content=doc['text'], metadata=doc) 
        for doc in combined_docs
    ]
    
    try:
        # Use the global reranker service
        reranked_responses = reranker.compress_documents(
            query=plan['semantic_query'],
            documents=docs_to_rerank
        )
        
        final_reranked_list = [doc.metadata for doc in reranked_responses]
        
        console.log(f"Cohere reranked to {len(final_reranked_list)} results.")
        return final_reranked_list

    except Exception as e:
        console.log(f"[red]Cohere Reranker failed: {e}. Returning top K unsorted.[/red]")
        return combined_docs[:RERANKED_TOP_N]

# Initialize the tools
local_510k_search = run_v2_hybrid_search

# --- Chatbot Mode: Graph Definition ---

class ChatbotState(TypedDict):
    """State for the chatbot graph."""
    query: str
    chat_history: list
    context: Optional[str]
    # --- NEW: State for relevance check ---
    is_relevant: bool
    final_answer: Optional[str] 

def run_local_search_node(state: ChatbotState):
    """Node to run the local 510(k) search tool."""
    console.log("[cyan]Running Local Search Node... (RAG-First)[/cyan]")
    query = state['query']
    results = local_510k_search(query)
    
    if not results:
        # If RAG finds nothing, set context to None
        console.log("[yellow]Local search found no results.[/yellow]")
        return {"context": None, "is_relevant": False}

    # Format context for the synthesizer
    context = "\n---\n".join([
        f"K_Number: {doc['k_number']}\nDevice: {doc['device_name']}\nSource: {doc['source']}\nText: {doc['text']}"
        for doc in results
    ])
    # We found context, but we don't know if it's relevant yet.
    return {"context": context, "is_relevant": False} # is_relevant is False by default

# --- NEW: Relevance Check Node ---
def run_relevance_check_node(state: ChatbotState):
    """
    Runs an LLM agent to check if the retrieved context is actually
    relevant to the user's query.
    """
    console.log("[yellow]Running Relevance Check Agent...[/yellow]")
    query = state['query']
    context = state.get('context')

    if not context:
        console.log("[dim]No context found, skipping relevance check.[/dim]")
        return {"is_relevant": False}

    prompt_text = get_prompt("relevance_check_prompt")
    prompt = PromptTemplate(
        template=prompt_text,
        input_variables=["query", "context"],
        partial_variables={"schema": RelevanceCheck.schema_json()}
    )
    
    # Use a JSON parser for this agent
    parser = JsonOutputParser(pydantic_model=RelevanceCheck)
    relevance_checker = prompt | llm | parser

    try:
        result: RelevanceCheck = relevance_checker.invoke({"query": query, "context": context[:2000]}) # Truncate context
        console.log(Panel(f"Relevant: {result['is_relevant']}\nJustification: {result['justification']}", title="Relevance Check", border_style="yellow"))
        return {"is_relevant": result['is_relevant']}
    except Exception as e:
        console.log(f"[red]Relevance Check agent failed: {e}. Defaulting to irrelevant.[/red]")
        return {"is_relevant": False}


def run_internet_search_node(state: ChatbotState):
    """Node to run the internet search tool."""
    console.log("[blue]Running Internet Search Node...[/blue]")
    query = state['query']
    results = internet_search.invoke(query)
    
    # Format context for the synthesizer
    context = "\n---\n".join([
        f"URL: {doc['url']}\nContent: {doc['content']}"
        for doc in results
    ])
    return {"context": context}

def run_synthesizer_node(state: ChatbotState):
    """
    The final node. Generates a natural language answer based on
    the query and the retrieved context (from any source).
    """
    console.log("[green]Running Final Synthesizer Agent...[/green]")
    query = state['query']
    context = state.get('context', "No context provided.")
    
    synthesizer_prompt_text = get_prompt("synthesizer_prompt")
    synthesizer_prompt = PromptTemplate(
        template=synthesizer_prompt_text,
        input_variables=["query", "context"],
    )
    
    synthesizer = synthesizer_prompt | llm | StrOutputParser()
    answer = synthesizer.invoke({"query": query, "context": context})
    
    console.rule("[bold green]Final Answer[/bold green]", style="green")
    console.print(answer)
    
    # Add to chat history
    current_history = state.get("chat_history", [])
    current_history.append(("user", query))
    current_history.append(("assistant", answer))
    
    return {"chat_history": current_history, "final_answer": answer}

# --- NEW: Updated Conditional Edge ---
def relevance_conditional_edge(state: ChatbotState) -> str:
    """
    This function checks the `is_relevant` flag.
    If True, it routes to the synthesizer.
    If False, it routes to the internet.
    """
    console.log("[dim]Checking relevance...[/dim]")
    if state.get("is_relevant") == True:
        console.log("[dim]RAG results are relevant. Routing to synthesizer.[/dim]")
        return "synthesize"
    else:
        console.log("[dim]RAG results are irrelevant. Routing to internet.[/dim]")
        return "search_internet"

def build_chatbot_graph():
    """Builds the LangGraph for the Chatbot mode."""
    workflow = StateGraph(ChatbotState)

    workflow.add_node("search_local", run_local_search_node)
    # --- NEW: Add the relevance check node ---
    workflow.add_node("check_relevance", run_relevance_check_node)
    workflow.add_node("search_internet", run_internet_search_node)
    workflow.add_node("synthesizer", run_synthesizer_node)
    
    # --- NEW ENTRY POINT ---
    workflow.set_entry_point("search_local")
    
    # --- NEW EDGES ---
    # After local search, always check relevance
    workflow.add_edge("search_local", "check_relevance")
    
    # After relevance check, decide where to go
    workflow.add_conditional_edges(
        "check_relevance",
        relevance_conditional_edge,
        {
            "synthesize": "synthesizer",
            "search_internet": "search_internet"
        }
    )
    
    # After internet search, always synthesize
    workflow.add_edge("search_internet", "synthesizer")
    
    # The synthesizer is the end of this turn
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()

# --- Retrieval Mode: Function ---

def run_retrieval_mode():
    """
    Runs the "find similar devices" mode.
    This is a pure vector search, not a chatbot.
    """
    console.rule("[bold blue]Device Retrieval Mode[/bold blue]", style="blue")
    console.log("This mode finds the most similar devices to a given K-Number.")
    
    try:
        k_number = input("Enter a K-Number (e.g., K000009): ").strip().upper()
        
        # 1. Get the vector for the query device
        # We'll get its "intended_use" vector as the seed
        query_vector_data, _ = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION,
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
            console.log(f"[red]Error: No 'intended_use' vector found for {k_number}.[/red]")
            console.log("Please make sure ingestion was successful and the K-Number is correct.")
            return

        query_vector = query_vector_data[0].vector
        console.log(f"Found seed vector for [bold]{k_number}[/bold]. Finding similar devices...")

        # 2. Use Qdrant's `recommend` feature
        # This finds vectors similar to our seed vector
        recommendations = qdrant_client.recommend(
            collection_name=QDRANT_COLLECTION,
            positive=[query_vector], # Find vectors *like* this one
            limit=RERANKED_TOP_N + 1, # +1 to exclude the query doc itself
            query_filter=models.Filter(
                must_not=[ # Don't return the query doc itself
                    models.FieldCondition(key="k_number", match=models.MatchValue(value=k_number))
                ]
            )
        )
        
        console.rule(f"[bold green]Top {len(recommendations)} Similar Devices[/bold green]", style="green")
        for i, hit in enumerate(recommendations):
            payload = hit.payload
            console.print(Panel(
                f"[bold]K-Number:[/bold] {payload['k_number']}\n"
                f"[bold]Device:[/bold] {payload['device_name']}\n"
                f"[bold]Manufacturer:[/bold] {payload['manufacturer']}\n"
                f"[bold]Match Source:[/bold] {payload['source']}\n"
                f"[bold]Text Snippet:[/bold] {payload['text'][:200]}...",
                title=f"Rank {i+1} (Score: {hit.score:.4f})",
                border_style="green"
            ))

    except Exception as e:
        console.log(f"[red]An error occurred during retrieval: {e}[/red]")
        import traceback
        traceback.print_exc()

# --- Main Execution ---

def main():
    """
    Main execution function.
    Loads the config and runs either the chatbot or retrieval mode.
    """
    if MODE == "chatbot":
        console.rule("[bold blue]Multi-Agent Chatbot Mode (v5 - Relevance Check)[/bold blue]", style="blue")
        console.log("Ask questions about 510(k) devices or general topics. Type 'exit' to quit.")
        app = build_chatbot_graph()
        
        # Simple chat history
        chat_history = []
        
        while True:
            try:
                query = input("\n👤 You: ")
                if query.lower() in ["exit", "quit"]:
                    break
                
                inputs = {"query": query, "chat_history": chat_history}
                
                # Use .invoke() for a single, complete run
                output = app.invoke(inputs)
                
                # Update history from the final state
                chat_history = output.get("chat_history", [])

            except Exception as e:
                console.log(f"[red]An unexpected error occurred: {e}[/red]")
                console.log("Please ensure your API keys are set correctly in `config.yaml`.")
                import traceback
                traceback.print_exc()

    elif MODE == "retrieval":
        run_retrieval_mode()

if __name__ == "__main__":
    main()