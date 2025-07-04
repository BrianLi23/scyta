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

def serialize_datetime(obj):
    """Helper function to serialize datetime objects to JSON"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

class Document:
    def __init__(self, doc_id: str, path: str, metadata: Dict, chunks: List[str]):
        self.doc_id = doc_id
        self.path = path
        self.metadata = metadata
        self.chunks = chunks

    def to_row(self) -> Tuple[str, str, str, str]:
        return (
            self.doc_id,
            self.path,
            json.dumps(self.metadata, default=serialize_datetime),
            json.dumps(self.chunks)
        )

    @classmethod
    def from_row(cls, row: Tuple[str, str, str, str]):
        doc_id, path, metadata_str, chunks_str = row
        metadata = json.loads(metadata_str)
        chunks = json.loads(chunks_str)
        return cls(doc_id, path, metadata, chunks)

class DocumentStore:
    def __init__(self):
        self.embedding_model = SentenceTransformer('./backend/models/paraphrase-MiniLM-L6-v2') # Embedding model, should be altered if tasks scales, dimension is 384
        self.chunk_size = 10
        self.overlap = 10
        self.chunker = rc.SemanticChunker(self.chunk_size, self.overlap) # Chunker for the documents
        self.file_agent = FileAgent(name="FileAgent", description="Agent for file operations")
        
        # Load FAISS index
        self.index_dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.vector_index_path = 'vector_index' # Path in disk to store the index
        self.indexed_files_path = 'document_store.db'
        
        # Load existing state if available
        loaded = self._load_index(self.vector_index_path)
        self.vector_index = loaded if loaded else faiss.IndexFlatL2(self.index_dimension)
        
        # In-memoey storage
        self.documents = []
        self.document_lookup = {}
        self.doc_id_set = set() # For fast lookup
        
        self._load_state()
    
    # def _load_indexed_files(self, filepath: str = None) -> Dict:
    #     # Loads document metadata and lookup information from SQLite database
    #     try:
    #         with sqlite3.connect(filepath) as conn:
    #             c = conn.cursor()
                
    #             # Start transaction
    #             conn.execute("BEGIN TRANSACTION")
                
    #             try:
    #                 # Create tables if they don't exist
    #                 c.execute('''
    #                     CREATE TABLE IF NOT EXISTS documents 
    #                     (doc_id TEXT PRIMARY KEY, path TEXT, metadata TEXT, chunks TEXT)
    #                 ''')
                    
    #                 c.execute('''
    #                     CREATE TABLE IF NOT EXISTS document_lookup
    #                     (faiss_index INTEGER PRIMARY KEY, doc_id TEXT, chunk_index INTEGER,
    #                      FOREIGN KEY (doc_id) REFERENCES documents(doc_id))
    #                 ''')
                    
    #                 # Create index on doc_id for better join performance
    #                 c.execute('''
    #                     CREATE INDEX IF NOT EXISTS idx_document_lookup_doc_id 
    #                     ON document_lookup(doc_id)
    #                 ''')
                    
    #                 # Load documents with pagination
    #                 batch_size = 1000
    #                 offset = 0
                    
    #                 while True:
    #                     c.execute("SELECT * FROM documents LIMIT ? OFFSET ?", (batch_size, offset))
    #                     rows = c.fetchall()
                        
    #                     if not rows:
    #                         break
                            
    #                     for row in rows:
    #                         doc_id, path, metadata_str, chunks_str = row
                            
    #                         try:
    #                             metadata = json.loads(metadata_str)
    #                             chunks = json.loads(chunks_str)
                                
    #                             self.documents.append({
    #                                 "doc_id": doc_id,
    #                                 "path": path,
    #                                 "metadata": metadata,
    #                                 "chunks": chunks
    #                             })
                                
    #                         except json.JSONDecodeError as e:
    #                             logger.error(f"Error parsing JSON for document {doc_id}: {str(e)}")
                                
    #                     offset += batch_size
                    
    #                 # Load document lookup with pagination
    #                 offset = 0
    #                 while True:
    #                     c.execute("SELECT * FROM document_lookup LIMIT ? OFFSET ?", (batch_size, offset))
    #                     lookups = c.fetchall()
                        
    #                     if not lookups:
    #                         break
                            
    #                     for lookup in lookups:
    #                         faiss_idx, doc_id, chunk_idx = lookup
                            
    #                         # Validate chunk index
    #                         doc = next((d for d in self.documents if d["doc_id"] == doc_id), None)
    #                         if doc and 0 <= chunk_idx < len(doc["chunks"]):
    #                             self.document_lookup[faiss_idx] = {
    #                                 "doc_id": doc_id,
    #                                 "chunk_index": chunk_idx
    #                             }
    #                         else:
    #                             logger.warning(f"Invalid chunk index {chunk_idx} for document {doc_id}")
                                
    #                     offset += batch_size
                    
    #                 # Commit transaction
    #                 conn.commit()
    #                 return {"message": "Successfully loaded indexed files"}
                    
    #             except Exception as e:
    #                 # Rollback transaction on error
    #                 conn.rollback()
    #                 raise e
                    
    #     except Exception as e:
    #         logger.error(f"Error loading indexed files: {str(e)}")
    #         return {"error": str(e)}
    
    def _connect_to_db(self):
        return sqlite3.connect(self.indexed_files_path)
    
    def _ensure_schema(self, conn: sqlite3.Connection):
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS documents 
            (doc_id TEXT PRIMARY KEY, path TEXT, metadata TEXT, chunks TEXT)
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS document_lookup
            (faiss_index INTEGER PRIMARY KEY, doc_id TEXT, chunk_index INTEGER, FOREIGN KEY(doc_id) REFERENCES documents(doc_id))
        ''')
        
        c.execute('''
            CREATE INDEX IF NOT EXISTS idx_document_lookup_doc_id 
            ON document_lookup(doc_id)
        ''')
        
        conn.commit()
    
    def _load_documents(self, conn: sqlite3.Connection):
        c = conn.cursor()
        docs = []
        for row in c.execute("SELECT * FROM documents"):
            docs.append(Document.from_row(row))
        return docs
    
    def _load_document_lookup(self, conn: sqlite3.Connection):
        c = conn.cursor()
        lookup = {}
        for row in c.execute("SELECT * FROM document_lookup"):
            lookup[row[0]] = {
                "doc_id": row[1],
                "chunk_index": row[2]
            }
        return lookup

    def _load_index(self, filepath: str = None) -> Optional[faiss.Index]:
        try:
            if os.path.exists(filepath):
                print(f"Loading index from {filepath}")
                return faiss.read_index(filepath)
            else:
                print(f"No existing index found at {filepath}")
                return None
            
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            return None
        
    def _index_documents(self, paths: List[str]) -> Dict:
        try:
            print("Indexing documents...")
            for path in paths:
                # Use FileAgent's _get_metadata_single directly
                metadata = self.file_agent._get_metadata_single(path)
                if "error" in metadata:
                    logger.error(f"Error getting metadata for {path}: {metadata['error']}")
                    continue

                doc_id = metadata["file_hash"]
                
                # Skip if already indexed
                if doc_id in self.doc_id_set:
                    print(f"Document {doc_id} already indexed, skipping")
                    continue
                
                # Process document
                chunks, embeddings = self._process_document(path)
                
                print(f"Chunks: {chunks}")
                print(f"Embeddings: {embeddings}")
                
                # Add to FAISS index
                start_idx = self.vector_index.ntotal
                self.vector_index.add(embeddings)
                end_idx = self.vector_index.ntotal
                
                # Map the new indexes to a document id
                for offset, idx in enumerate(range(start_idx, end_idx)):
                    self.document_lookup[idx] = {
                        "doc_id": doc_id,
                        "chunk_index": offset
                    }
                
                # Store document info
                self.documents.append(Document(doc_id, path, metadata, chunks))
                self.doc_id_set.add(doc_id)
                
            print("Documents indexed successfully")
            return {"message": "Successfully indexed documents"}
            
        except Exception as e:
            print(f"Error indexing documents: {str(e)}")
            return {"error": str(e)}
        
    def _save_state(self) -> Dict:
        # Save the current state of the document store to disk
        try:
            print("Saving state...")
            # Save FAISS index
            faiss.write_index(self.vector_index, self.vector_index_path)
            print(f"Saved FAISS index to {self.vector_index_path}")
            
            # Save metadata to SQLite database
            with self._connect_to_db() as conn:
                self._ensure_schema(conn)
                c = conn.cursor()
            
                # Save documents
                doc_rows = [doc.to_row() for doc in self.documents]
                c.executemany("INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?)", doc_rows)
                
                # Save document lookup
                lookup_rows = [(idx, lookup["doc_id"], lookup["chunk_index"]) for idx, lookup in self.document_lookup.items()]
                c.executemany("INSERT OR REPLACE INTO document_lookup VALUES (?, ?, ?)", lookup_rows)
                conn.commit()
            
            print("State saved successfully")
            return {"message": "Successfully saved state"}
            
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
            return {"error": str(e)}
        
    def _load_state(self) -> Dict:
        # Load the complete state of the document store from disk
        try:
            with self._connect_to_db() as conn:
                self._ensure_schema(conn)
                self.documents = self._load_documents(conn)
                self.document_lookup = self._load_document_lookup(conn)
            
            self.doc_id_set = {doc.doc_id for doc in self.documents}
            print("State loaded successfully")
            return {"message": "Successfully loaded state"}
            
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            return {"error": str(e)}
        
    def _search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents using FAISS index"""
        try:
            # Encode query
            query_embedding = self.embedding_model.encode(query)
            
            # Ensure query embedding is 2D array
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
                
            # Perform similarity search
            print(f"Index size: {self.vector_index.ntotal}")
            distances, indices = self.vector_index.search(query_embedding, top_k) 
            print(f"Distances: {distances}")
            print(f"Indices: {indices}")
            
            # Prepare results
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                    
                entry = self.document_lookup[idx]
                doc = next(doc for doc in self.documents if doc.doc_id == entry["doc_id"])
                chunk_idx = entry["chunk_index"]
                print(f"Found document: {doc}")
                
                results.append({
                   "document": {
                    "doc_id": doc.doc_id,
                    "path": doc.path,
                    "metadata": doc.metadata,
                   },
                   "chunk": doc.chunks[chunk_idx],
                   "score": float(1 / (1 + distance))
                })
                
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
        # Log the current state of the document store
        logger.info(f"\n=== Document Store State before {operation} ===")
        logger.info(f"Number of documents: {len(self.documents)}")
        logger.info(f"FAISS index size: {self.vector_index.ntotal}")
        logger.info(f"Document lookup entries: {len(self.document_lookup)}")
        if self.documents:
            logger.info("Documents in store:")
            for doc in self.documents:
                logger.info(f"- {doc['doc_id']}: {doc['path']}")
        logger.info("================================\n")
        
# Example usage
if __name__ == "__main__":
    store = DocumentStore()