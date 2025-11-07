import os
import csv
import requests
from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm
from utils import get_config, console # Use the central console

# --- Configuration ---
INPUT_FILE = "pmn96cur.txt"  # The big index file from the FDA
OUTPUT_DIR = get_config("REAL_CORPUS_DIR", "./510k_corpus")
BASE_URL = "https://501k.fra1.cdn.digitaloceanspaces.com/parsed/"
# --- End Configuration ---

def download_corpus():
    """
    Parses the FDA index file and downloads the corresponding summary
    text files.
    """
    console.rule(f"[bold blue]510(k) Corpus Downloader Started[/bold blue]", style="blue")
    
    if not os.path.exists(INPUT_FILE):
        console.log(f"[red]Error: Input file '{INPUT_FILE}' not found.[/red]")
        console.log("Please download 'pmn96cur.txt' from the FDA site and place it here.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        console.log(f"Created output directory: {OUTPUT_DIR}")

    # Open the pipe-delimited file
    with open(INPUT_FILE, 'r', encoding='latin-1') as f:
        # Use csv.DictReader to handle the headers and delimiter
        reader = csv.DictReader(f, delimiter='|')
        
        # Convert reader to a list to show a progress bar
        all_rows = []
        try:
            all_rows = list(reader)
        except csv.Error as e:
            console.log(f"[red]Error reading CSV file '{INPUT_FILE}': {e}[/red]")
            console.log("The file might be corrupt or have an unexpected format.")
            return

        console.log(f"Found {len(all_rows)} total records in '{INPUT_FILE}'.")
        
        # !! THE FIX IS HERE !!
        # Changed `all_rows` to `len(all_rows)`
        console.log(f"Starting download of all {len(all_rows)} files...")

        # Use tqdm for a progress bar
        download_count = 0
        NA_count = 0
        for row in tqdm(all_rows, desc="Downloading summaries"):
            knumber = row.get('KNUMBER')
            
            # Skip non-K files (like DEN, etc.) or empty rows
            if not knumber or not knumber.startswith('K'):
                continue

            file_name = f"{knumber}.txt"
            output_path = os.path.join(OUTPUT_DIR, file_name)
            
            # Skip if we already have it
            if os.path.exists(output_path):
                download_count += 1 # Count it as "downloaded" if it's already present
                continue
                
            download_url = f"{BASE_URL}{file_name}"
            
            try:
                response = requests.get(download_url, timeout=10)
                
                # Check if the download was successful
                if response.status_code == 200:
                    with open(output_path, 'w', encoding='utf-8') as out_f:
                        out_f.write(response.text)
                    download_count += 1
                else:
                    # Not all K-numbers have a parsed text file, this is normal
                    NA_count += 1
                    pass
                    
            except requests.RequestException:
                # Suppress noisy errors, just count as NA
                NA_count += 1

    console.rule(f"[bold green]Download Complete[/bold green]", style="green")
    console.log(f"Successfully downloaded or found {download_count} summary files in '{OUTPUT_DIR}'.")
    console.log(f"Parsed summary files NA for {NA_count} k-numbers")
    console.log("You can now run `ingestion.py` to process this new data.")
    
    # Print a sample of the first downloaded file
    try:
        # Find the first file that *was* downloaded and exists
        first_downloaded_k = None
        for row in all_rows:
             if row.get('KNUMBER', '').startswith('K'):
                first_downloaded_k = row['KNUMBER']
                break
        
        if first_downloaded_k:
            sample_file = os.path.join(OUTPUT_DIR, f"{first_downloaded_k}.txt")
            if os.path.exists(sample_file):
                with open(sample_file, 'r', encoding='utf-8') as f:
                    sample_content = f.read(500)
                console.print(Panel(
                    f"{sample_content}...",
                    title=f"Sample: {sample_file}",
                    border_style="gray"
                ))
    except Exception:
        pass # Ignore if sample can't be read

if __name__ == "__main__":
    download_corpus()