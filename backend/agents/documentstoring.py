from typing import List, Dict, Optional, Tuple
import os
from pathlib import Path
import json
from datetime import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.agents.fileagent import FileAgent 
import backend.agents.ragchunking as rc
import argparse
import sqlite3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentStore:
    def __init__(self):
        self.embedding_model = SentenceTransformer('./backend/models/paraphrase-MiniLM-L6-v2') # Embedding model, should be altered if tasks scales, dimension is 384
        self.chunk_size = 10
        self.overlap = 10
        self.chunker = rc.SemanticChunker(self.chunk_size, self.overlap) # Chunker for the documents
        self.file_agent = FileAgent(name="FileAgent", description="Agent for file operations")
        
        # Load FAISS index
        self.index_dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Load existing state if available
        self.indexed_files = self._load_indexed_files() # This is the metadata of the documents
        self.vector_index = self._load_index(self.vector_index_path) # This is the FAISS index
        
        if self.vector_index is not None:
            self.vector_index = self.vector_index
        
        self.vector_index_path = 'vector_index' # Path in disk to store the index
        self.indexed_files_path = 'document_store.db'
        
        # Initialize storage for documents and metadata
        self.documents = []
        self.document_lookup = {}
        
    def _index_documents(self, paths: List[str]) -> Dict:
        """Index new documents with given paths"""
        try:
            self._log_document_store_state("indexing")
            indexed_count = 0
            for path in paths:
                # Use FileAgent's _get_metadata_single directly
                metadata = self.file_agent._get_metadata_single(path)
                if "error" in metadata:
                    logger.error(f"Error getting metadata for {path}: {metadata['error']}")
                    continue

                doc_id = metadata["file_hash"]
                
                # Skip if already indexed
                if doc_id in {doc["doc_id"] for doc in self.documents}:
                    logger.info(f"Document {doc_id} already indexed, skipping")
                    continue
                
                # Process document
                chunks, embeddings = self._process_document(path)
                
                # Add to FAISS index
                start_idex = self.vector_index.ntotal
                self.vector_index.add(embeddings)
                end_index = self.vector_index.ntotal
                
                # Map the new indexes to a document id
                for i, idx in enumerate(range(start_idex, end_index)):
                    self.document_lookup[idx] = {
                        "doc_id": doc_id,
                        "chunk_index": i
                    }
                
                # Store document info
                self.documents.append({
                    "doc_id": doc_id,
                    "path": path,
                    "metadata": metadata,
                    "chunks": chunks
                })
                indexed_count += 1

            self._log_document_store_state("after indexing")
            return {"message": f"Successfully indexed {indexed_count} documents"}
            
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            return {"error": str(e)}
            
    def _load_index(self, filepath: str = None) -> Optional[faiss.Index]:
        """Loads vector index from disk if exists"""
        try:
            if os.path.exists(filepath or self.vector_index_path):
                logger.info(f"Loading index from {filepath or self.vector_index_path}")
                return faiss.read_index(filepath or self.vector_index_path)
            else:
                logger.warning(f"No existing index found at {filepath or self.vector_index_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return None
            
    def _load_indexed_files(self, filepath: str = None) -> Dict:
        """Loads document metadata and lookup information from SQLite database"""
        try:
            conn = sqlite3.connect(filepath or self.indexed_files_path)
            c = conn.cursor()
            
            # Create tables if they don't exist
            c.execute('''
                CREATE TABLE IF NOT EXISTS documents 
                (doc_id TEXT PRIMARY KEY, path TEXT, metadata TEXT, chunks TEXT)
            ''')
            
            c.execute('''
                CREATE TABLE IF NOT EXISTS document_lookup
                (faiss_index INTEGER PRIMARY KEY, doc_id TEXT, chunk_index INTEGER)
            ''')
            
            # Load documents
            c.execute("SELECT * FROM documents")
            rows = c.fetchall()
            
            for row in rows:
                doc_id, path, metadata_str, chunks_str = row
                
                try:
                    metadata = json.loads(metadata_str)
                    chunks = json.loads(chunks_str)
                    
                    self.documents.append({
                        "doc_id": doc_id,
                        "path": path,
                        "metadata": metadata,
                        "chunks": chunks
                    })
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON for document {doc_id}: {str(e)}")
            
            # Load document lookup
            c.execute("SELECT * FROM document_lookup")
            lookups = c.fetchall()
            
            for lookup in lookups:
                faiss_idx, doc_id, chunk_idx = lookup
                self.document_lookup[faiss_idx] = {
                    "doc_id": doc_id,
                    "chunk_index": chunk_idx
                }
            
            conn.close()
            return {"message": "Successfully loaded indexed files"}
            
        except Exception as e:
            logger.error(f"Error loading indexed files: {str(e)}")
            return {"error": str(e)}
        
    def _save_state(self) -> Dict:
        """Save the current state of the document store to disk"""
        try:
            self._log_document_store_state("saving")
            # Save FAISS index
            faiss.write_index(self.vector_index, self.vector_index_path)
            logger.info(f"Saved FAISS index to {self.vector_index_path}")
            
            # Save metadata to SQLite database
            conn = sqlite3.connect(self.indexed_files_path)
            c = conn.cursor()
            
            c.execute('''
                CREATE TABLE IF NOT EXISTS documents 
                (doc_id TEXT PRIMARY KEY, path TEXT, metadata TEXT, chunks TEXT)
            ''')
            
            c.execute('''
                CREATE TABLE IF NOT EXISTS document_lookup
                (faiss_index INTEGER PRIMARY KEY, doc_id TEXT, chunk_index INTEGER)
            ''')
            
            for doc in self.documents:
                c.execute(
                    "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?)", 
                    (doc["doc_id"], doc["path"], 
                     json.dumps(doc["metadata"], default=serialize_datetime), 
                     json.dumps(doc["chunks"]))
                )
            
            for idx, mapping in self.document_lookup.items():
                c.execute(
                    "INSERT OR REPLACE INTO document_lookup VALUES (?, ?, ?)", 
                    (idx, mapping["doc_id"], mapping["chunk_index"])
                )
                
            conn.commit()
            conn.close()
            logger.info(f"Saved document store state to {self.indexed_files_path}")
            return {"message": "Successfully saved state"}
            
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
            return {"error": str(e)}
        
    def _load_state(self) -> Dict:
        """Load the complete state of the document store from disk"""
        try:
            self._log_document_store_state("loading")
            # Load FAISS index
            loaded_index = self._load_index(self.vector_index_path)
            if loaded_index is not None:
                self.vector_index = loaded_index
                logger.info("Successfully loaded FAISS index")
            
            # Load documents and lookup
            result = self._load_indexed_files(self.indexed_files_path)
            if "error" in result:
                return result
                
            self._log_document_store_state("after loading")
            return {"message": "Successfully loaded state"}
            
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            return {"error": str(e)}
        
    def _search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents using FAISS index"""
        try:
            # Encode query
            query_embedding = self.embedding_model.encode(query)
            
            # Input validation
            if not isinstance(top_k, int) or top_k < 1:
                raise ValueError("top_k must be a positive integer")
                
            # Ensure query embedding is 2D array
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
                
            # Perform similarity search
            logger.info(f"Index size: {self.vector_index.ntotal}")
            distances, indices = self.vector_index.search(query_embedding, top_k) 
            logger.debug(f"Distances: {distances}")
            logger.debug(f"Indices: {indices}")
            
            # Prepare results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                    
                doc_id = self.document_lookup[idx]["doc_id"] 
                chunk_idx = self.document_lookup[idx]["chunk_index"]
                document = next(doc for doc in self.documents if doc["doc_id"] == doc_id)
                logger.debug(f"Found document: {document}")
                
                result = {
                    "chunk": document["chunks"][chunk_idx],
                    "document": {
                        "doc_id": doc_id,
                        "path": document["path"],
                        "metadata": document["metadata"]
                    },
                    "score": float(1 / (1 + distance))
                }
                results.append(result)
                
            return results
        
        except Exception as e:
            logger.error(f"Error during FAISS search: {str(e)}")
            raise Exception(f"Error during FAISS search: {str(e)}")
            
    def _process_document(self, path: str, metadata: Dict = None) -> Tuple[List[str], np.ndarray]:
        """Process the document at the given path"""
        if metadata is not None:
            file_type = metadata["file_type"]
        
        with open(path, "r") as file:
            text = file.read()
        
        chunks = self.chunker.chunk_text(text)
        embeddings = self.embedding_model.encode([chunk for chunk in chunks])
        return chunks, embeddings
    
    def _view_document_store(self) -> Dict:
        """View the contents of the document store"""
        try:
            conn = sqlite3.connect(self.indexed_files_path)
            c = conn.cursor()
            
            # Get all tables from the database
            c.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = c.fetchall()
            logger.info(f"Tables in the database: {tables}")
            
            # Query document tables
            c.execute("SELECT * FROM documents")
            documents = c.fetchall()
            logger.info(f"Documents in the database: {len(documents)}")
            
            store_info = {
                "tables": [table[0] for table in tables],
                "document_count": len(documents),
                "documents": []
            }
            
            for doc in documents:
                doc_id, path, metadata_str, chunks_str = doc
                
                try:
                    metadata = json.loads(metadata_str)
                    chunks = json.loads(chunks_str)
                    
                    doc_info = {
                        "doc_id": doc_id,
                        "path": path,
                        "metadata": metadata,
                        "chunk_count": len(chunks),
                        "first_chunks": chunks[:3] if chunks else []
                    }
                    store_info["documents"].append(doc_info)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON for document {doc_id}: {str(e)}")
            
            # Query document_lookup table
            c.execute("SELECT COUNT(*) FROM document_lookup")
            lookup_count = c.fetchone()[0]
            store_info["lookup_entries"] = lookup_count
            
            conn.close()
            return store_info
            
        except Exception as e:
            logger.error(f"Error viewing document store: {str(e)}")
            return {"error": str(e)}
        
    def _log_document_store_state(self, operation: str):
        """Log the current state of the document store"""
        logger.info(f"\n=== Document Store State before {operation} ===")
        logger.info(f"Number of documents: {len(self.documents)}")
        logger.info(f"FAISS index size: {self.vector_index.ntotal}")
        logger.info(f"Document lookup entries: {len(self.document_lookup)}")
        if self.documents:
            logger.info("Documents in store:")
            for doc in self.documents:
                logger.info(f"- {doc['doc_id']}: {doc['path']}")
        logger.info("================================\n")

def serialize_datetime(obj):
    """Helper function to serialize datetime objects to JSON"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

# Example usage
if __name__ == "__main__":
    store = DocumentStore()
    # Add example usage here
