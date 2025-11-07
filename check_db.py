import duckdb
import qdrant_client
from qdrant_client import models
from rich.pretty import pprint
from rich.panel import Panel

# --- NEW: Import services and configs from utils ---
from utils import (
    console,
    get_config,
    get_db_connection,
    get_qdrant_client
)
# ---

# --- Configuration (Loaded from utils) ---
DB_FILE = get_config("DB_FILE")
QDRANT_COLLECTION = get_config("QDRANT_COLLECTION")
# --- End Configuration ---


def check_duckdb():
    """Connects to DuckDB and prints its contents."""
    console.rule("[bold blue]Checking DuckDB (Structured Data)[/bold blue]", style="blue")
    try:
        conn = get_db_connection(read_only=True)
        
        # Get count
        count = conn.execute("SELECT COUNT(*) FROM device_metadata").fetchone()[0]
        console.log(f"Found {count} rows in 'device_metadata' table.\n")
        
        # Get first 3 rows
        console.log("--- Sample Data (first row) ---")
        # Use DESC to show the latest, richest data
        data = conn.execute("SELECT * FROM device_metadata LIMIT 1").fetchall()
        
        # Pretty-print the rows
        if data:
            row = data[0]
            # DuckDB fetchall() returns a list of tuples
            # We need to get column names to make it a dict for printing
            cols = [desc[0] for desc in conn.description]
            row_dict = dict(zip(cols, row))
            
            # Build a dynamic string for the panel
            panel_content = ""
            for col, val in row_dict.items():
                # Truncate long text fields for display
                if isinstance(val, str) and len(val) > 200:
                    val = f"{val[:200]}..."
                
                # Highlight the section-based fields
                if col.startswith("section_"):
                    panel_content += f"\n[bold magenta]{col}:[/bold magenta]\n{val}\n"
                else:
                    panel_content += f"[bold]{col}:[/bold] {val}\n"

            console.print(Panel(
                panel_content,
                title=f"Row 1: {row_dict.get('k_number')}"
            ))
        else:
            console.log("[yellow]No data found in DuckDB.[/yellow]")
            
        conn.close()

    except Exception as e:
        console.log(f"[red]Error checking DuckDB: {e}[/red]")
        console.log("Make sure `ingestion.py` ran successfully and created 'device_510k.db'")

def check_qdrant():
    """Connects to Qdrant and prints a few vector payloads."""
    console.rule("[bold magenta]Checking Qdrant (Vector Data)[/bold magenta]", style="magenta")
    try:
        client = get_qdrant_client()
        
        # Get count
        count = client.count(collection_name=QDRANT_COLLECTION, exact=True).count
        console.log(f"Found {count} vectors in '{QDRANT_COLLECTION}' collection.\n")
        
        # --- NEW: Check for different vector types ---
        console.log("--- Sample 'Section' Vector (source: intended_use) ---")
        section_data, _ = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=1,
            with_payload=True,
            with_vectors=False,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="source", match=models.MatchValue(value="intended_use"))]
            )
        )
        
        if section_data:
            pprint(section_data[0].payload)
        else:
            console.log("[yellow]No 'intended_use' section vectors found.[/yellow]")

        console.log("\n--- Sample 'General Chunk' Vector (source: general_chunk) ---")
        chunk_data, _ = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=1,
            with_payload=True,
            with_vectors=False,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="source", match=models.MatchValue(value="general_chunk"))]
            )
        )
        
        if chunk_data:
            pprint(chunk_data[0].payload)
        else:
            console.log("[yellow]No 'general_chunk' vectors found.[/yellow]")


    except Exception as e:
        console.log(f"[red]Error checking Qdrant: {e}[/red]")
        console.log("Make sure `ingestion.py` ran successfully and created the './qdrant_db' directory")

if __name__ == "__main__":
    check_duckdb()
    console.print("\n")
    check_qdrant()