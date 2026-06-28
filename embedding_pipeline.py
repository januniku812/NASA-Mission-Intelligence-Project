#!/usr/bin/env python3
"""
ChromaDB Embedding Pipeline for NASA Space Mission Data - Text Files Only

This script reads parsed text data from various NASA space mission folders and creates
a permanent ChromaDB collection with OpenAI embeddings for RAG applications.
Optimized to process only text files to avoid duplication with JSON versions.

Supported data sources:
- Apollo 11 extracted data (text files only)
- Apollo 13 extracted data (text files only)
- Apollo 11 Textract extracted data (text files only)
- Challenger transcribed audio data (text files only)
"""

# forcing python to look at the newest verison of 
__import__('pysqlite3')
import sys 
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import json
import logging
import re 
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
import openai
from openai import OpenAI
import hashlib
import time
import traceback
from datetime import datetime
import argparse
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChromaEmbeddingPipelineTextOnly:
    """Pipeline for creating ChromaDB collections with OpenAI embeddings - Text files only"""
    
    def __init__(self, 
                 openai_api_key: str,
                 chroma_persist_directory: str = "./chroma_db",
                 collection_name: str = "nasa_space_missions_text",
                 embedding_model: str = "text-embedding-3-small",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the embedding pipeline
        
        Args:
            openai_api_key: OpenAI API key
            chroma_persist_directory: Directory to persist ChromaDB
            collection_name: Name of the ChromaDB collection
            embedding_model: OpenAI embedding model to use
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between chunks
        """
        openai_api_key = (openai_api_key or os.getenv("OPENAI_API_KEY"))
        if openai_api_key.startswith("sk"):
            base_url_client = "https://api.openai.com/v1"
        elif openai_api_key.startswith("voc"):
            base_url_client= "https://openai.vocareum.com/v1"
        else:
            base_url_client = os.getenv("OPENAI_BASE_URL")

        # Initialize OpenAI client
        self.client = OpenAI(api_key=openai_api_key,
                            base_url=base_url_client
                            ) # client for query answer generation (G of RAG)
                        
        # Store configuration parameters
        self.chroma_persist_directory = chroma_persist_directory
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize ChromaDB client 
        self.chromadb_client = chromadb.PersistentClient(path=self.chroma_persist_directory) # client for ChromaDB (embedded vector database) based retrieval (R of RAG)
       
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Split text into chunks with metadata
        
        Args:
            text: Text to chunk
            metadata: Base metadata for the text
            
        Returns:
            List of (chunk_text, chunk_metadata) tuples
        """
        import uuid
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        document_category = metadata["document_category"]

        return_list = []
        # Handle short texts that don't need chunking
        if len(text) < self.chunk_size:
           return [(text, metadata)]

        # Implement chunking logic with overlap
        # Try to break at sentence boundaries
        # Create metadata for each chunk
        start_index = 0 
        chunk_index = 0 
        chunk_start_index_diff = max(1, self.chunk_size-self.chunk_overlap) # we set the start index step to be the difference between the chunk siez and overlap desired as overlaps in text chunking allow the retriever to preserve semantic meaning and context across text chunks, improving the quality of retrievval 
        while start_index < len(text):
            if start_index + self.chunk_size >= (len(text)):
                baseline_end = len(text)

                # implementing logic for sentence breaks if there are sentence breaks
                if re.search(r"\.\s", text[start_index: baseline_end]): 
                    latest_sentence_break_index = 0 
                    for m in re.finditer(r"\.\s", text[start_index: baseline_end]):
                        match_index = m.span()[0]
                        if match_index > latest_sentence_break_index:
                            latest_sentence_break_index = match_index

                    if latest_sentence_break_index >= (len(text) / 2): # if the latest sentence break is more than halfway through the text 
                        # split chunks 
                        return_list.append((text[start_index: latest_sentence_break_index],
                                            {
                                                "chunk_index": chunk_index,
                                                "chunk_start_position": start_index,
                                                "chunk_end_position": latest_sentence_break_index,
                                                "chunk_source": metadata["source"],
                                                "chunk_mission": metadata["mission"],
                                                "document_category": document_category,
                                                "source_file_path": metadata["file_path"] 
                                            }))

                        return_list.append((text[latest_sentence_break_index: baseline_end],
                                            {
                                                "chunk_index": chunk_index,
                                                "chunk_start_position": latest_sentence_break_index,
                                                "chunk_end_position": baseline_end,
                                                "chunk_source": metadata["source"],
                                                "chunk_mission": metadata["mission"],
                                                "document_category": document_category,
                                                "source_file_path": metadata["file_path"]
                                            }))

                    else: # logic for when the sentence break is too early in the text chunk 
                        return_list.append((text[start_index: baseline_end], # keep baseline end as if the latest sentence break is too early in the text then breaking will return incomplete sentence parts
                                            {
                                                "chunk_index": chunk_index,
                                                "chunk_start_position": start_index,
                                                "chunk_end_position": baseline_end,
                                                "chunk_source": metadata["source"],
                                                "chunk_mission": metadata["mission"],
                                                "document_category": document_category,
                                                "source_file_path": metadata["file_path"]
                                            }))


                else: # implementing normal non-sentence break logic 
                    return_list.append((text[start_index: baseline_end], # chunk 
                                        { # metadata for final chunk of text with size up to self.chunk_size
                                        "chunk_index": chunk_index,
                                        "chunk_start_position": start_index,
                                        "chunk_end_position": baseline_end,
                                        "chunk_source": metadata["source"],
                                        "chunk_mission": metadata["mission"],
                                        "document_category": document_category,
                                        "source_file_path": metadata["file_path"] 
                                    })) 

                break

            else: 
                baseline_end = start_index + self.chunk_size
                # implementing logic for sentence breaks if there are any within the chunk 
                if re.search(r"\.\s", text[start_index: baseline_end]): 
                    latest_sentence_break_index = 0 
                    for m in re.finditer(r"\.\s", text[start_index: baseline_end]):
                        match_index = m.span()[0]
                        if match_index > latest_sentence_break_index:
                            latest_sentence_break_index = match_index

                    if latest_sentence_break_index >= (len(text) / 2): # if the latest sentence break is more than halfway through the text 
                        # split chunks 
                        return_list.append((text[start_index: latest_sentence_break_index],
                                            {
                                                "chunk_index": chunk_index,
                                                "chunk_start_position": start_index,
                                                "chunk_end_position": latest_sentence_break_index,
                                                "chunk_source": metadata["source"],
                                                "chunk_mission": metadata["mission"],
                                                "document_category": document_category,
                                                "source_file_path": metadata["file_path"] 
                                            }))

                        return_list.append((text[latest_sentence_break_index: baseline_end],
                                            {
                                                "chunk_index": chunk_index,
                                                "chunk_start_position": latest_sentence_break_index,
                                                "chunk_end_position": baseline_end,
                                                "chunk_source": metadata["source"],
                                                "chunk_mission": metadata["mission"],
                                                "document_category": document_category,
                                                "source_file_path": metadata["file_path"] 
                                            })) 

                    else: # logic for when the sentence break is too early in the text chunk 
                        return_list.append((text[start_index: baseline_end], # keep baseline end as if the latest sentence break is too early in the text then breaking will return incomplete sentence parts
                                            {
                                                "chunk_index": chunk_index,
                                                "chunk_start_position": start_index,
                                                "chunk_end_position": baseline_end,
                                                "chunk_source": metadata["source"],
                                                "chunk_mission": metadata["mission"],
                                                "document_category": document_category,
                                                "source_file_path": metadata["file_path"] 
                                            }))

                else: # no sentence break within text return functionality 
                    return_list.append((text[start_index: baseline_end], # keep baseline end as if the latest sentence break is too early in the text then breaking will return incomplete sentence parts
                                            {
                                                "chunk_index": chunk_index,
                                                "chunk_start_position": start_index,
                                                "chunk_end_position": baseline_end,
                                                "chunk_source": metadata["source"],
                                                "chunk_mission": metadata["mission"],
                                                "document_category": document_category,
                                                "source_file_path": metadata["file_path"] 
                                            }))

                chunk_index += 1
                start_index += chunk_start_index_diff

        return return_list
    
    def check_document_exists(self, doc_id: str) -> bool:
        """
        Check if a document with the given ID already exists in the collection
        
        Args:
            doc_id: Document ID to check
            
        Returns:
            True if document exists, False otherwise
        """
        # Query collection for document ID
        # Return True if exists, False otherwise
        try:
            document = collection.query(
                ids=[doc_id]
            )
            return bool(document and document.get("ids"))

        except Exception as e: 
            logger.log(level=logging.INFO, msg=f"[check_document_exists] function failed to return collection for document id: {doc_id}")
            return False
    
    
    def update_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """
        Update an existing document in the collection
        
        Args:
            doc_id: Document ID to update
            text: New text content
            metadata: New metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get new embedding
            embedding = self.get_embedding(text, dimension=384)
            
            # Update the document
            self.chromadb_client.get_or_create_collection(collection_dir).update(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding]
            )
            logger.debug(f"Updated document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False
    
    def delete_documents_by_source(self, collection_dir: str, source_pattern: str) -> int:
        """
        Delete all documents from a specific source (useful for re-processing files)
        
        Args:
            source_pattern: Pattern to match source names
            
        Returns:
            Number of documents deleted
        """
        try:
            # Get all documents
            all_docs = self.chromadb_client.get_or_create_collection(collection_dir).get()
            
            # Find documents matching the source pattern
            ids_to_delete = []
            for i, metadata in enumerate(all_docs['metadatas']):
                if source_pattern in metadata.get('source', ''):
                    ids_to_delete.append(all_docs['ids'][i])
            
            if ids_to_delete:
                self.chromadb_client.get_or_create_collection(collection_dir).delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} documents matching source pattern: {source_pattern}")
                return len(ids_to_delete)
            else:
                logger.info(f"No documents found matching source pattern: {source_pattern}")
                return 0
                
        except Exception as e:
            logger.error(f"Error deleting documents by source: {e}")
            return 0
    
    def get_file_documents(self, collection_dir: str, file_path: Path | str) -> List[str]:
        """
        Get all document IDs for a specific file
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of document IDs for the file
        """
        try:
            file_path = Path(file_path)
            source = file_path.stem
            mission = self.extract_mission_from_path(file_path)
            
            # Get all documents
            all_docs = self.chromadb_client.get_or_create_collection(collection_dir).get()
            
            # Find documents from this file
            file_doc_ids = []
            for i, metadata in enumerate(all_docs['metadatas']):
                if (metadata.get('source') == source and 
                    metadata.get('mission') == mission):
                    file_doc_ids.append(all_docs['ids'][i])
            
            return file_doc_ids
            
        except Exception as e:
            logger.error(f"Error getting file documents: {e}")
            return []
    
    def get_embedding(self, text: str, dimension: int) -> List[float]:
        """
        Get OpenAI embedding for text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Call OpenAI embeddings API
        try: 
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_model,
                dimensions=dimension
            )
            # Return embedding vector
            return response.data[0].embedding 
        # Add error handling
        except Exception as e: 
            logger.log(level=logging.INFO, msg=f"[get_embedding] function received embedding error for text: {text}") # forcing logger to log the text that caused embedding error before raising exception 
            raise

    def generate_document_id(self, file_path: Path | str, metadata: Dict[str, Any]) -> str:
        """
        Generate stable document ID based on file path and chunk position
        This allows for document updates without changing IDs
        """
        # Create consistent ID format
        id_format = "{}_{}_{}"
        # Use mission, source, and chunk_index
        file_path = Path(file_path)
        # Format: mission_source_chunk_0001
        return id_format.format(metadata["chunk_mission"], (metadata.get("chunk_source", file_path.stem)), metadata["chunk_index"])
    
    def process_text_file(self, file_path: Path | str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Process plain text files with enhanced metadata extraction
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of (text, metadata) tuples
        """
        try:
            file_path = Path(file_path)

            with open(str(file_path), 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                return []
            
            # Enhanced metadata extraction --> refer to document metadata key list
            metadata = {
                'source': file_path.stem,
                'file_path': str(file_path),
                'file_type': 'text',
                'content_type': 'full_text',
                'mission': self.extract_mission_from_path(file_path),
                'data_type': self.extract_data_type_from_path(file_path),
                'document_category': self.extract_document_category_from_filename(file_path.name),
                'file_size': len(content),
                'processed_timestamp': datetime.now().isoformat()
            }
            
            return self.chunk_text(content, metadata)
            
        except Exception as e:
            traceback.print_exception(e)
            logger.error(f"Error processing text file {file_path}: {e}")
            return []
    
    def extract_mission_from_path(self, file_path: Path) -> str:
        """Extract mission name from file path"""
        path_str = str(file_path).lower()
        if 'apollo11' in path_str or 'apollo_11' in path_str:
            return 'apollo_11'
        elif 'apollo13' in path_str or 'apollo_13' in path_str:
            return 'apollo_13'
        elif 'challenger' in path_str:
            return 'challenger'
        else:
            return 'unknown'
    
    def extract_data_type_from_path(self, file_path: Path) -> str:
        """Extract data type from file path"""
        path_str = str(file_path).lower()
        if 'transcript' in path_str:
            return 'transcript'
        elif 'textract' in path_str:
            return 'textract_extracted'
        elif 'audio' in path_str:
            return 'audio_transcript'
        elif 'flight_plan' in path_str:
            return 'flight_plan'
        else:
            return 'document'
    
    def extract_document_category_from_filename(self, filename: str) -> str:
        """Extract document category from filename for better organization"""
        filename_lower = filename.lower()
        
        # Apollo transcript types
        if 'pao' in filename_lower:
            return 'public_affairs_officer'
        elif 'cm' in filename_lower:
            return 'command_module'
        elif 'tec' in filename_lower:
            return 'technical'
        elif 'flight_plan' in filename_lower:
            return 'flight_plan'
        
        # Challenger audio segments
        elif 'mission_audio' in filename_lower:
            return 'mission_audio'
        
        # NASA archive documents
        elif 'ntrs' in filename_lower:
            return 'nasa_archive'
        elif '19900066485' in filename_lower:
            return 'technical_report'
        elif '19710015566' in filename_lower:
            return 'mission_report'
        
        # General categories
        elif 'full_text' in filename_lower:
            return 'complete_document'
        else:
            return 'general_document'
    
    def scan_text_files_only(self, base_path: str) -> List[Path]:
        """
        Scan data directories for text files only (avoiding JSON duplicates)
        
        Args:
            base_path: Base directory path
            
        Returns:
            List of text file paths to process
        """
        base_path = Path(base_path)
        files_to_process = []
        
        # Define directories to scan
        data_dirs = [
            'apollo11',
            'apollo13',
            'challenger'
        ]
        
        for data_dir in data_dirs:
            dir_path = base_path / data_dir
            if dir_path.exists():
                logger.info(f"Scanning directory: {dir_path}")
                
                # Find only text files
                text_files = list(dir_path.glob('**/*.txt'))
                files_to_process.extend(text_files)
                logger.info(f"Found {len(text_files)} text files in {data_dir}")
        
        # Filter out unwanted files
        filtered_files = []
        for file_path in files_to_process:
            # Skip system files and summaries
            if (file_path.name.startswith('.') or 
                'summary' in file_path.name.lower() or
                file_path.suffix.lower() != '.txt'):
                continue
            filtered_files.append(file_path)
        
        logger.info(f"Total text files to process: {len(filtered_files)}")
        
        # Log file breakdown by mission
        mission_counts = {}
        for file_path in filtered_files:
            mission = self.extract_mission_from_path(file_path)
            mission_counts[mission] = mission_counts.get(mission, 0) + 1
        
        logger.info("Files by mission:")
        for mission, count in mission_counts.items():
            logger.info(f"  {mission}: {count} files")
        
        return filtered_files
    
    def add_documents_to_collection(self, documents: List[Tuple[str, Dict[str, Any]]], 
                                    collection_dir: str,
                                   file_path: Path, batch_size: int = 50, 
                                   update_mode: str = 'skip', 
                                   ) -> Dict[str, int]:
        """
        Add documents to ChromaDB collection in batches with update handling
        
        Args:
            documents: List of (text, metadata) tuples
            file_path: Path to the source file
            batch_size: Number of documents to process in each batch
            update_mode: How to handle existing documents:
                        'skip' - skip existing documents
                        'update' - update existing documents
                        'replace' - delete all existing documents from file and re-add
                        'add' - add all documents to database for the first time 
            
        Returns:
            Dictionary with counts of added, updated, and skipped documents
        """

        if not documents:
            return {'added': 0, 'updated': 0, 'skipped': 0}
        
        stats = {'added': 0, 'updated': 0, 'skipped': 0}

        update_mode_to_stats_keys_mapping = {
            "add": "added",
            "update": "updated",
            "replace": "replaced",
            "skip": "skipped"
        }

        if update_mode == "skipped": 
            stats["skipped"] += len(documents)
            return stats 

        with open("chunks.txt", "w") as info_file:
            info_file.write(f"chunks being written to chromadb collection database to instantiate application information through update mode: {update_mode}")
            info_file.write(str(documents))

        collection = self.chromadb_client.get_or_create_collection(collection_dir) 

        # colleciton add method works by ttakingkaing corresponding chunk ids, chunk documents, chunk metadata, and embeddings in seperate lists passed at as inputs
        ids: List[str] = []
        docs: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        # Handle different update modes (skip, update, replace)
        # skip is the default update mode so only update and replace 
        if update_mode == "replace":
            existing_document_ids = self.get_file_documents(collection_dir=collection_dir, file_path=file_path)
            if existing_document_ids: 
                collection.delete(ids=[existing_document_ids])

        # For each document:
        #   - Generate document ID
        #   - Check if exists
        #   - Get embedding
        #   - Add or update in collection
        for text, metadata in documents:
            doc_id = self.generate_document_id(file_path, metadata)
            if update_mode == "update":  
                if self.check_document_exists(doc_id):
                    ids.append(doc_id)
                    docs.append(text)
                    metadatas.append(metadata)
            else: # replace or add 
                ids.append(doc_id)
                docs.append(text)
                metadatas.append(metadata)
                
        # process corresponding id, document text, and metadata updates/adds in batches using iterator
        # process chroma db collection update for whichever update mode was selected 
        batch_start_index = 0 
        while batch_start_index < len(docs): # Process documents in batches 
            ideal_end = batch_start_index + batch_size 
            if ideal_end > len(docs): 
                batched_documents = docs[batch_start_index: len(docs)]
                if update_mode == "update": 
                    collection.update(
                        ids=ids[batch_start_index: len(docs)],
                        documents=batched_documents,
                        metadatas= metadatas[batch_start_index: len(docs)],
                        embeddings=self.get_batch_embedding(batched_documents)
                    ) 

                elif update_mode == "add" or update_mode == "replace": 
                    collection.add(
                        ids=ids[batch_start_index: len(docs)],
                        documents=batched_documents,
                        metadatas= metadatas[batch_start_index: len(docs)],
                        embeddings=self.get_batch_embedding(batched_documents)
                    ) 

                stats[update_mode_to_stats_keys_mapping[update_mode]] += len(batched_documents)
                break 

            else: 
                batched_documents = docs[batch_start_index: ideal_end]
                if update_mode =="update": 
                    collection.update(
                        ids=ids[batch_start_index: ideal_end],
                        documents=batched_documents,
                        metadatas= metadatas[batch_start_index: ideal_end],
                        embeddings=self.get_batch_embedding(batched_documents)
                    )   
                
                elif update_mode == "add" or update_mode== "replace": 
                    collection.add(
                        ids=ids[batch_start_index: ideal_end],
                        documents=batched_documents,
                        metadatas= metadatas[batch_start_index: ideal_end],
                        embeddings=self.get_batch_embedding(batched_documents)
                    )  

                stats[update_mode_to_stats_keys_mapping[update_mode]] += len(batched_documents)
                batch_start_index += batch_size 

        return stats 

    def process_all_text_data(self, base_path: str, update_mode: str = 'skip') -> Dict[str, int]:
        """
        Process all text files and add to ChromaDB
        
        Args:
            base_path: Base directory containing data folders
            update_mode: How to handle existing documents:
                        'skip' - skip existing documents (default)
                        'update' - update existing documents
                        'replace' - delete all existing documents from file and re-add
            
        Returns:
            Statistics about processed files
        """
        stats = {}
        for collection_dir in os.listdir(base_path): 
            collection_dir_stats = {
                'files_processed': 0,
                'documents_added': 0,
                'documents_updated': 0,
                'documents_skipped': 0,
                'errors': 0,
                'total_chunks': 0,
                'missions': {}
            }
            collection = self.chromadb_client.get_or_create_collection(collection_dir)
            collection_directory_compiled_path = os.path.join(base_path, collection_dir)
            # Get files to process
            files_to_process = os.listdir(collection_directory_compiled_path)
            # Loop through each file
            for file_path in files_to_process: 
                try:
                    compiled_file_path = os.path.join(collection_directory_compiled_path, file_path)
                    chunks = self.process_text_file(compiled_file_path)
                    # Process file and add to collection
                    file_stats = self.add_documents_to_collection(
                        documents=chunks,
                        collection_dir=collection_dir,
                        file_path=file_path, 
                        update_mode=update_mode, # when instantiating chroma db database for first time using embedding_pipeline.py use --update-mode "add" in args 
                    )
                    # Update statistics
                    collection_dir_stats['files_processed'] += 1 
                    collection_dir_stats['documents_added'] += file_stats['added']
                    collection_dir_stats['documents_updated'] += file_stats['updated']
                    collection_dir_stats['documents_skipped'] += file_stats['skipped']
                    collection_dir_stats['total_chunks'] += len(chunks)

                    stats[collection_dir] = collection_dir_stats



                except Exception as e:# Handle errors gracefully
                    traceback.print_exception(e)
                    collection_dir_stats["errors"] += 1 
                    logger.log(level=logging.INFO, msg=f"[process_all_text_data] faced error: {e} in processing file at file_path: {file_path}")

        return stats
    
    def get_collection_info(self, collection) -> Dict[str, Any]:
        """Get information about the ChromaDB collection"""
        # Return collection name, document count, metadata
        try: 
            return {
            "collection_name": collection.name,
            "document_count": collection.count(), 
            "metadata": collection.metadata
        }
        except Exception as e: 
            return {
                "collection_name": collection.name,
                "document_count": collection.count(), 
                "metadata": collection.metadata,
                "error": str(e)
            }

    
    def query_collection(self, collection_dir: str, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Query the collection for testing
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            
        Returns:
            Query results
        """
        # Perform test query and return results
        try: 
            return self.chromadb_client.get_or_create_collection(collection_dir).query(
                input=query_text,
                n_results = n_results
            )
        except Exception as e: 
            logger.log(level=logging.INFO, msg=f"[query_collection] function with query text: {query_text}, n_results: {n_results} ran into error: {str(e)}")

    def get_batch_embedding(self, document): 
        response = self.client.embeddings.create(
            input=document,
            model=self.embedding_model
        )
        return [d.embedding for d in response.data]
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the collection"""
        try:
            # Get all documents to analyze
            all_docs = self.chromadb_client.get_or_create_collection(collection_dir).get()
            
            if not all_docs['metadatas']:
                return {'error': 'No documents in collection'}
            
            stats = {
                'total_documents': len(all_docs['metadatas']),
                'missions': {},
                'data_types': {},
                'document_categories': {},
                'file_types': {}
            }
            
            # Analyze metadata
            for metadata in all_docs['metadatas']:
                mission = metadata.get('mission', 'unknown')
                data_type = metadata.get('data_type', 'unknown')
                doc_category = metadata.get('document_category', 'unknown')
                file_type = metadata.get('file_type', 'unknown')
                
                # Count by mission
                stats['missions'][mission] = stats['missions'].get(mission, 0) + 1
                
                # Count by data type
                stats['data_types'][data_type] = stats['data_types'].get(data_type, 0) + 1
                
                # Count by document category
                stats['document_categories'][doc_category] = stats['document_categories'].get(doc_category, 0) + 1
                
                # Count by file type
                stats['file_types'][file_type] = stats['file_types'].get(file_type, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='ChromaDB Embedding Pipeline for NASA Data')
    parser.add_argument('--data-path', default='.', help='Path to data directories')
    parser.add_argument('--openai-key', required=True, help='OpenAI API key')
    parser.add_argument('--chroma-dir', default='./chroma_db_openai', help='ChromaDB persist directory')
    parser.add_argument('--collection-name', default='nasa_space_missions_text', help='Collection name')
    parser.add_argument('--embedding-model', default='text-embedding-3-small', help='OpenAI embedding model')
    parser.add_argument('--chunk-size', type=int, default=500, help='Text chunk size')
    parser.add_argument('--chunk-overlap', type=int, default=100, help='Chunk overlap size')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--update-mode', choices=['skip', 'update', 'replace', 'add'], default='skip',
                       help='How to handle existing documents: skip, update, or replace')
    parser.add_argument('--test-query', help='Test query after processing')
    parser.add_argument('--stats-only', action='store_true', help='Only show collection statistics')
    parser.add_argument('--delete-source', help='Delete all documents from a specific source pattern')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    logger.info("Initializing ChromaDB Embedding Pipeline...")
    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key=args.openai_key,
        chroma_persist_directory=args.chroma_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    ) # forcefully setting the openai vocareum key with correct matchign base url 
    
    # Handle delete source operation
    if args.delete_source:
        deleted_count = pipeline.delete_documents_by_source(source_pattern=args.delete_source, collection_dir=args.collection_name)
        logger.info(f"Deleted {deleted_count} documents matching source pattern: {args.delete_source}")
        return
    
    # If stats only, show collection statistics and exit
    if args.stats_only:
        logger.info("Collection Statistics:")
        stats = pipeline.get_collection_stats()
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        return
    
    # Process all data ---> changed functions and pipeline to automatically create new collections within the chroma db database for each directory within the inputted (./data_text) directory 
    logger.info(f"Starting text data processing with update mode: {args.update_mode}")
    start_time = time.time()
    
    stats = pipeline.process_all_text_data(base_path=args.data_path, update_mode=args.update_mode)
    

    end_time = time.time()
    processing_time = end_time - start_time

    for key, dictionary in stats.items():
        logger.info(f"Collection: {key} - Statistics")
        # use logger to display results
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Files processed: {dictionary['files_processed']}")
        logger.info(f"Total chunks created: {dictionary['total_chunks']}")
        logger.info(f"Documents added to collection: {dictionary['documents_added']}")
        logger.info(f"Documents updated in collection: {dictionary['documents_updated']}")
        logger.info(f"Documents skipped (already exist): {dictionary['documents_skipped']}")
        logger.info(f"Errors: {dictionary['errors']}")
        
        # Mission breakdown
        logger.info("\nMission breakdown:")
        for mission, mission_stats in dictionary['missions'].items():
            logger.info(f"  {mission}: {mission_stats['files']} files, {mission_stats['chunks']} chunks")
            logger.info(f"    Added: {mission_stats['added']}, Updated: {mission_stats['updated']}, Skipped: {mission_stats['skipped']}")


        # Collection info
        for collection_dir in os.listdir("data_text"):
            collection_info = pipeline.get_collection_info(pipeline.chromadb_client.get_or_create_collection(key))
            logger.info(f"\nCollection: {collection_info.get('collection_name', 'N/A')}")
            logger.info(f"Total documents in collection: {collection_info.get('document_count', 'N/A')}")
            

    logger.info(f"Processing time: {processing_time:.2f} seconds")
    
    # Test query if provided
    if args.test_query:
        logger.info(f"\nTesting query: '{args.test_query}'")
        results = pipeline.query_collection(query_text=args.test_query, collection_dir=args.collection_name)
        if results and 'documents' in results:
            logger.info(f"Found {len(results['documents'][0])} results:")
            for i, doc in enumerate(results['documents'][0][:3]):  # Show top 3
                logger.info(f"Result {i+1}: {doc[:200]}...")

    logger.info(f"Collections created: {pipeline.chromadb_client.list_collections()}")
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
