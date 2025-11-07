import os
from utils import (
    console,
    get_config,
    get_db_connection
)

# --- Configuration ---
DB_FILE = get_config("DB_FILE")
OUTPUT_FILE = "ingested_k_numbers.txt"
# ---

def export_k_numbers():
    """
    Connects to the DuckDB database, queries all K-Numbers,
    and saves them to a text file.
    """
    console.rule(f"[bold blue]Exporting K-Numbers[/bold blue]", style="blue")
    
    if not os.path.exists(DB_FILE):
        console.log(f"[red]Error: Database file not found at {DB_FILE}[/red]")
        console.log("Please run `ingestion.py` first.")
        return

    try:
        conn = get_db_connection(read_only=True)
        
        # Query all k_numbers
        console.log(f"Querying all K-Numbers from {DB_FILE}...")
        results = conn.execute("SELECT k_number FROM device_metadata ORDER BY k_number").fetchall()
        
        if not results:
            console.log("[yellow]No K-Numbers found in the database.[/yellow]")
            conn.close()
            return

        # Write to the output file
        count = 0
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for row in results:
                f.write(f"{row[0]}\n")
                count += 1
        
        conn.close()
        
        console.rule(f"[bold green]Export Complete[/bold green]", style="green")
        console.log(f"Successfully exported {count} K-Numbers to [bold]{OUTPUT_FILE}[/bold].")

    except Exception as e:
        console.log(f"[red]An error occurred: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    export_k_numbers()