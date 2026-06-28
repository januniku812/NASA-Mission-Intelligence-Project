# forcing python to look at the newest verison of 
__import__('pysqlite3')
import sys 
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from chromadb import PersistentClient
from pathlib import Path
import traceback 
import pysqlite3
import os 
import embedding_pipeline 
import json 
import logging 
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_client.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

document_snippet_display_chars = 500

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    # deferred chromadb imports
    import chromadb 
    from chromadb.config import Settings 

    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    
    # Look for ChromaDB directories
    # Create list of directories that match specific criteria (directory type and name pattern)
    chroma_directories = []
    for potential_chroma_directory in current_dir.iterdir(): 
        if potential_chroma_directory.name.startswith("chromadb") or potential_chroma_directory.name.startswith("chroma_db"):
            chroma_directories.append(potential_chroma_directory)

    logger.log(level=logging.INFO, msg=f"Step 2. Chroma Directories established {chroma_directories} ✅")

    # Loop through each discovered directory
    for chroma_directory in chroma_directories: 
        # Wrap connection attempt in try-except block for error handling
        try:
            # Initialize database client with directory path and configuration settings
            client = chromadb.PersistentClient(
                path=str(chroma_directory),
                settings=Settings(anonymized_telemetry=False)
            )
            # Retrieve list of available collections from the database
            collections = client.list_collections() 
            logger.log(level=logging.INFO, msg=f"available collections: {collections}")
            # Loop through each collection found
            for collection in collections: 
                # Create unique identifier key combining directory and collection names
                key=f"{chroma_directory}-{collection}"

                try:
                    doc_count = str(collection.count()) 

                except Exception as e:
                    doc_count = "unknown"

                info_dictionary={
                    "directory": str(chroma_directory), # Store directory path as string
                    "collection_name": collection.name, # Store collection name
                    "display_name": f"chromadb - {collection.name}",# Create user-friendly display name
                    "document_count": doc_count  # Get document count with fallback for unsupported operations
                    }

                if not key in [key for key, value in backends.items()]:
                    backends[key] = {}

                backends[key] = info_dictionary

        except Exception as e:
            logger.log(level=logging.INFO, msg=f"Initializing ChromaDB Persistent Client at chroma directory: {chroma_directory} failed with error: {traceback.format_exc()}")
            chromadb_failed_connection_key = f"{chroma_directory}-[]"
            fallback_info_dictionary = {
                "directory": str(chroma_directory),
                "collection_name": "", 
                "display_name": f"chromadb - [collection name not provided]", 
                "document_count": ""
            }

            if not chromadb_failed_connection_key in [key for key, value in backends.items()]:
                backends[chromadb_failed_connection_key] = {}

            backends[chromadb_failed_connection_key] = fallback_info_dictionary 

    # Return complete backends dictionary with all discovered collections
    return backends 

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    # Create a chomadb persistentclient
    persistent_client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False)
            )
    # Return the collection with the collection_name
    try: 
        collection = persistent_client.get_or_create_collection(name=collection_name)
        return collection
    except Exception as e: 
        traceback.print_exception(e) 
        logger.log(level=logging.INFO, msg=f"Failed to return collection name: {collection_name}")   
        return None

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # Initialize filter variable to None (represents no filtering)
    where_filter_parameter = None 

    # Check if filter parameter exists and is not set to "all" or equivalent
    if mission_filter and mission_filter.lower() not in ("all", "any", ""):
    # If filter conditions are met, create filter dictionary with appropriate field-value pairs
        where_filter_parameter = {"Mission": mission_filter}
        
    try: # Execute database query with the following parameters:
        pipeline = embedding_pipeline.ChromaEmbeddingPipelineTextOnly(openai_api_key=os.getenv("OPENAI_API_KEY", ""))

        filter_dictionary = collection.query(
            query_embeddings=pipeline.get_embedding(query, dimension=1536),
            query_texts=[query], # Pass search query in the required format
            n_results=n_results, # Set maximum number of results to return
            where=where_filter_parameter # Apply conditional filter (None for no filtering, dictionary for specific filtering)
        )

        filter_dictionary_file_path = "dict_content.txt"
        with open(filter_dictionary_file_path, 'w') as info_file: 
            info_file.write(f"info_file dictionary for query: {query}")
            json.dump(filter_dictionary, info_file)

        return filter_dictionary # Return query results to caller

    except Exception as e: 
        traceback.print_exception(e)
        logger.log(level=logging.INFO, msg=f"Retrieval failed for query: {query} mission filter: {mission_filter}")


def format_context(documents: List[str], metadatas: List[Dict], scores: List[float]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    MAX_DIVIDER_LENGTH = 50 
    
    # Initialize list with header text for context section
    contexts_list = ["Retrieved NASA Document Context"]
    file_paths = [] # keeping running list of unique file paths to avoid deduplicatoin 

    information_components = ["chunk_mission", "chunk_source", "document_category"]
    components_that_need_cleaning = ["chunk_mission", "document_category"]
    information_components_to_context_formatting_mapping = {
        "chunk_mission": "Mission",
        "chunk_source": "Source",
        "document_category": "Category"
    }

    combined = list(zip(documents, metadatas, scores))
    combined.sort(key=lambda x: x[2], reverse=True) # sorting by scores in descending order to get best context (document, metadata) pairs 

    # Loop through paired documents and their metadata using enumeration
    for index, (document, metadata, distance) in enumerate(combined):
        # Extract mission information from metadata with fallback value
        file_path = metadata["source_file_path"]
        if file_path not in file_paths: 
            context_starter = "\n"
            information_components = ["chunk_mission", "chunk_source", "document_category"]
            components_that_need_cleaning = ["chunk_mission", "document_category"]
            final_context_string_parts = [] 
            for information_component in information_components: 
                # Clean up mission name formatting (replace underscores, capitalize)
                # Extract source information from metadata with fallback value  
                # Extract category information from metadata with fallback value
                # Clean up category name formatting (replace underscores, capitalize)
                info_piece = (metadata or {}).get(information_component.lower(), "unknown") or "unknown"
                if information_component in components_that_need_cleaning: 
                    final_context_string_parts.append(info_piece.lower().replace("_", " ").title())

            # Create formatted source header with index number and extracted information
            # Add source header to context parts list
            context_parts = []
            source_header_parts = [] 
            for index, (information_component_name, information_component) in enumerate(zip(information_components, final_context_string_parts)): 
                if index == 0: 
                    source_header_parts.append(f"{context_starter}{information_components_to_context_formatting_mapping[information_component_name]}: {information_component}") 
                else: 
                    source_header_parts.append(f"{information_components_to_context_formatting_mapping[information_component_name]}: {information_component}")

                source_header_parts.append("".join(["-" for i in range(min(MAX_DIVIDER_LENGTH, len(source_header_parts[-1])))]))

            source_header_str = "\n".join(source_header_parts)
            context_parts.append(source_header_str)

            # Check document length and truncate if necessary
            # Add truncated or full document content to context parts list
            context_parts.append("Context")
            context_parts.append("".join(["-" for i in range(min(MAX_DIVIDER_LENGTH, len(context_parts[-1])))]))

            if len(document) > document_snippet_display_chars: 
                context_parts.append(f"{document.strip()[:document_snippet_display_chars].rstrip()}" + " [...]")
            else: 
                context_parts.append(f"{document.strip()}")
            context_parts.append("".join(["-" for i in range(min(MAX_DIVIDER_LENGTH, len(context_parts[-1])))]))

            # Join all context parts with newlines and return formatted string
            context_parts_seperator = "\n"
            context_piece = context_parts_seperator.join(context_parts)

            contexts_list.append(context_piece)

            # updating processed file paths
            file_paths.append(file_path)

    return contexts_list