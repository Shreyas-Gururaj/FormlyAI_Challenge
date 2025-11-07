import os
import glob
import json
import duckdb
import uuid
import qdrant_client
import random
import yaml

from rich.panel import Panel
from rich.progress import Progress
from typing import List, Dict, Any, Optional

# --- NEW: Import services and configs from utils ---
from utils import (
    console, 
    # Use the factory functions from utils
    get_llm,
    get_embeddings, 
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

from langchain.prompts import PromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models

# --- Configuration (Loaded from utils) ---
DATA_DIR = get_config("DATA_DIR", "./mock_510k_files")
DB_FILE = get_config("DB_FILE")
QDRANT_PATH = get_config("QDRANT_PATH")
QDRANT_COLLECTION = get_config("QDRANT_COLLECTION")
INGESTION_SAMPLE_SIZE = get_config("INGESTION_SAMPLE_SIZE", 0)
EMBEDDING_MODEL = get_config("EMBEDDING_MODEL")
GEMINI_MODEL = get_config("GEMINI_MODEL")
GOOGLE_API_KEY = get_config("GOOGLE_API_KEY")

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Pydantic Schemas for Structured Data ---

class DeviceMetadata(BaseModel):
    """
    A Pydantic model for holding the structured metadata
    extracted from a 510(k) summary text.
    """
    k_number: str = Field(description="The 510(k) document number, e.g., K000009")
    device_name: str = Field(description="The trade name of the device (e.g., Spiral Radius Rodding System)")
    proprietary_name: Optional[str] = Field(description="The proprietary name of the device, if different")
    manufacturer: str = Field(description="The name of the company that submitted the 510(k)")
    contact_person: Optional[str] = Field(description="The name of the contact person")
    date_prepared: Optional[str] = Field(description="The date the summary was prepared")
    classification_names: List[str] = Field(description="A list of FDA classification names for the device")
    common_name: Optional[str] = Field(description="The common or usual name of the device")
    predicate_devices: List[str] = Field(description="A list of predicate device names or K-numbers cited")
    
    # --- NEW: Full-Text Sections ---
    section_device_description: Optional[str] = Field(description="The full text of the 'DEVICE DESCRIPTION' section")
    section_intended_use: Optional[str] = Field(description="The full text of the 'INTENDED USE' section")
    section_indications_for_use: Optional[str] = Field(description="The full text of the 'INDICATIONS FOR USE' section")
    section_materials: Optional[str] = Field(description="The full text of the 'MATERIALS' section")

# --- Database & Vector Store Initialization ---

def setup_database(db_conn: duckdb.DuckDBPyConnection):
    """Creates the metadata table."""
    # --- NEW: Updated table schema ---
    db_conn.execute("""
        CREATE TABLE IF NOT EXISTS device_metadata (
            k_number VARCHAR PRIMARY KEY,
            device_name VARCHAR,
            proprietary_name VARCHAR,
            manufacturer VARCHAR,
            contact_person VARCHAR,
            date_prepared VARCHAR,
            classification_names VARCHAR[],
            common_name VARCHAR,
            predicate_devices VARCHAR[],
            
            section_device_description TEXT,
            section_intended_use TEXT,
            section_indications_for_use TEXT,
            section_materials TEXT,
            
            raw_text_summary TEXT
        )
    """)
    # --- END NEW ---

def setup_vector_store(client: qdrant_client.QdrantClient, embeddings):
    """Initializes the Qdrant client and collection."""
    
    # Get embedding model dimensions
    embedding_dim = len(embeddings.embed_query("test"))

    # Recreate the collection every time for this demo.
    try:
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=embedding_dim,
                distance=models.Distance.COSINE
            ),
        )
        console.log(f"Qdrant collection '{QDRANT_COLLECTION}' created successfully.")
    except Exception as e:
        console.log(f"[red]Error creating Qdrant collection: {e}[/red]")
        console.log("This can happen if Qdrant is running in Docker and the path is bad.")
        console.log("For this demo, we're using a local file path.")
    
# --- Agent & Pipeline Definitions ---
def get_extractor_agent(llm):
    """
    Returns a LangChain agent (LLM + Prompt + Parser)
    that extracts structured metadata from a 510(k) summary.
    """
    
    # --- NEW: Load prompt from utils ---
    prompt_text = get_prompt("extractor_prompt")
    
    prompt = PromptTemplate(
        template=prompt_text,
        input_variables=["document"],
        partial_variables={"schema": DeviceMetadata.schema_json()}
    )
    
    parser = SimpleJsonOutputParser(pydantic_model=DeviceMetadata)

    return prompt | llm | parser

def get_chunker_agent():
    """
    Returns a text splitter configured for semantic chunking
    of regulatory documents.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",  # Paragraphs
            "\n",    # Lines
            " ",     # Words
            "",      # Characters
        ],
        length_function=len,
    )

# --- Main Processing Function ---
def process_file(filepath, extractor, chunker, db_conn, qdrant_client, embeddings):
    """
    Main processing pipeline for a single file.
    NOW: Implements hierarchical/section-aware ingestion.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        full_text = f.read()

    # --- 1. Run Extractor Agent ---
    try:
        metadata_dict = extractor.invoke({"document": full_text})
    except Exception as e:
        console.log(f"[red]Failed to extract metadata from {filepath}: {e}[/red]")
        return

    # --- 2. Store Metadata in DuckDB ---
    try:
        # --- NEW: Insert all new fields into the DB ---
        db_conn.execute(
            """
            INSERT INTO device_metadata (
                k_number, device_name, proprietary_name, manufacturer, contact_person,
                date_prepared, classification_names, common_name, predicate_devices,
                section_device_description, section_intended_use,
                section_indications_for_use, section_materials, raw_text_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (k_number) DO UPDATE SET
                device_name = EXCLUDED.device_name,
                proprietary_name = EXCLUDED.proprietary_name,
                manufacturer = EXCLUDED.manufacturer,
                contact_person = EXCLUDED.contact_person,
                date_prepared = EXCLUDED.date_prepared,
                classification_names = EXCLUDED.classification_names,
                common_name = EXCLUDED.common_name,
                predicate_devices = EXCLUDED.predicate_devices,
                section_device_description = EXCLUDED.section_device_description,
                section_intended_use = EXCLUDED.section_intended_use,
                section_indications_for_use = EXCLUDED.section_indications_for_use,
                section_materials = EXCLUDED.section_materials,
                raw_text_summary = EXCLUDED.raw_text_summary
            """,
            (
                metadata_dict['k_number'],
                metadata_dict['device_name'],
                metadata_dict['proprietary_name'],
                metadata_dict['manufacturer'],
                metadata_dict['contact_person'],
                metadata_dict['date_prepared'],
                metadata_dict['classification_names'],
                metadata_dict['common_name'],
                metadata_dict['predicate_devices'],
                metadata_dict['section_device_description'],
                metadata_dict['section_intended_use'],
                metadata_dict['section_indications_for_use'],
                metadata_dict['section_materials'],
                full_text
            )
        )
    except Exception as e:
        console.log(f"[red]Failed to insert metadata for {metadata_dict.get('k_number', 'UNKNOWN')}: {e}[/red]")
        return

    # --- 3. Run Chunker Agent (on full text) ---
    chunks = chunker.split_text(full_text)
    k_number = metadata_dict['k_number']

    points_to_upsert = []

    # --- 4. Create "General Chunk" Embeddings ---
    vector_embeddings = embeddings.embed_documents(chunks)
    
    for i, chunk in enumerate(chunks):
        chunk_id = f"{k_number}_chunk_{i}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
        
        payload = {
            "k_number": k_number,
            "device_name": metadata_dict['device_name'],
            "manufacturer": metadata_dict['manufacturer'],
            "text": chunk,
            "source": "general_chunk"  # <-- NEW: Source payload
        }
        
        points_to_upsert.append(
            models.PointStruct(
                id=point_id,
                vector=vector_embeddings[i],
                payload=payload
            )
        )

    # --- 5. Create "Section" Embeddings ---
    sections_to_embed = {
        "device_description": metadata_dict.get('section_device_description'),
        "intended_use": metadata_dict.get('section_intended_use'),
        "indications_for_use": metadata_dict.get('section_indications_for_use'),
        "materials": metadata_dict.get('section_materials'),
    }

    for section_name, section_text in sections_to_embed.items():
        if section_text: # Only embed if the text exists
            try:
                # Embed the *entire* section as one vector
                section_embedding = embeddings.embed_query(section_text)
                
                section_id_str = f"{k_number}_section_{section_name}"
                section_point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, section_id_str))
                
                section_payload = {
                    "k_number": k_number,
                    "device_name": metadata_dict['device_name'],
                    "manufacturer": metadata_dict['manufacturer'],
                    "text": section_text,
                    "source": section_name  # <-- NEW: Source payload (e.g., "intended_use")
                }
                
                points_to_upsert.append(
                    models.PointStruct(
                        id=section_point_id,
                        vector=section_embedding,
                        payload=section_payload
                    )
                )
            except Exception as e:
                console.log(f"[yellow]Warning: Failed to embed section {section_name} for {k_number}: {e}[/yellow]")


    # --- 6. Upsert All Vectors to Qdrant (in a single batch) ---
    try:
        if points_to_upsert:
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points_to_upsert,
                wait=True
            )
    except Exception as e:
        console.log(f"[red]Failed to upsert vectors for {k_number}: {e}[/red]")


# --- Main Execution ---

def main():
    """
    Main orchestration function.
    Initializes all services and processes all files in the directory.
    """
    console.rule("[bold green]Ingestion Pipeline Started (v3 Section-Aware)[/bold green]", style="green")
    
    # Initialize all our services
    # We need a specific LLM for ingestion
    llm = get_llm(GEMINI_MODEL, GOOGLE_API_KEY)
    embeddings = get_embeddings()
    db_conn = get_db_connection(read_only=False)
    qdrant_client = get_qdrant_client()
    
    # Setup the DBs (create tables, collections)
    setup_database(db_conn)
    setup_vector_store(qdrant_client, embeddings)
    
    # Get the agents
    extractor_agent = get_extractor_agent(llm)
    chunker = get_chunker_agent()
    
    # Find all mock files
    console.log(f"Loading files from: {DATA_DIR}")
    files_to_process = glob.glob(f"{DATA_DIR}/*.txt")
    if not files_to_process:
        console.log(f"[red]No .txt files found in {DATA_DIR}.[/red]")
        if "mock" in DATA_DIR:
            console.log("Please run `python mock_510k_data.py` first.")
        else:
            console.log("Please run `python download_corpus.py` first.")
        return

    console.log(f"Found {len(files_to_process)} total files.")

    # --- SAMPLING LOGIC ---
    if INGESTION_SAMPLE_SIZE > 0 and len(files_to_process) > INGESTION_SAMPLE_SIZE:
        console.log(f"[yellow]Randomly sampling {INGESTION_SAMPLE_SIZE} files from the total.[/yellow]")
        files_to_process = random.sample(files_to_process, INGESTION_SAMPLE_SIZE)
    # --- END SAMPLING LOGIC ---

    console.log(f"Starting ingestion for {len(files_to_process)} files...")

    # Process all files with a progress bar
    with Progress(console=console) as progress:
        task = progress.add_task("Ingesting 510(k) summaries...", total=len(files_to_process))
        
        for filepath in files_to_process:
            process_file(filepath, extractor_agent, chunker, db_conn, qdrant_client, embeddings)
            progress.update(task, advance=1)
            
    console.rule("[bold green]Ingestion Pipeline Finished[/bold green]", style="green")

    # --- Final Verification ---
    try:
        device_count = db_conn.execute("SELECT COUNT(*) FROM device_metadata").fetchone()[0]
        console.log(f"[bold]Total devices in structured DB: {device_count}[/bold]")
        
        vec_count = qdrant_client.count(collection_name=QDRANT_COLLECTION, exact=True).count
        console.log(f"[bold]Total vectors in vector DB: {vec_count}[/bold]")
        
        if device_count > 0 and vec_count > 0:
            console.log("[green]Databases populated successfully.[/green]")
        else:
            console.log("[yellow]Ingestion complete, but one or more databases are empty.[/yellow]")
            
    except Exception as e:
        console.log(f"[red]Failed to verify database counts: {e}[/red]")
        console.log("[red]Please check errors above. The Qdrant collection may not exist.[/red]")

    finally:
        db_conn.close()

if __name__ == "__main__":
    main()