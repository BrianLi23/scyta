from typing import List, Dict, Optional
import os
from pathlib import Path
import json
from datetime import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.mainframe import BaseAgent, Tool
from backend.agents.fileagent import FileAgent
import backend.agents.ragchunking as rc
from backend.agents.documentstoring import DocumentStore
from backend.schemas.operation_schemas import RAGAGENT_SCHEMA
from backend.prompts.ragagent_prompt import PROMPT_ROUTER, ROUTER_EXAMPLES, FEW_SHOT_EXAMPLES, PROMPT_REASONING
# from backend.tools.ragagent_tools import RAGAgentTools
import copy
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Structure
# Query preprocessing --> query vectorization --> retreival phase --> prompt augementation --> LLM generation --> response postprocessing
class RAGAgent(BaseAgent):
    def __init__(self, name, description):
        super().__init__(name, description)
        
        self.document_store = DocumentStore()
        self.file_agent = FileAgent(name="FileAgent", description="Agent for file operations")
        
        # Initialize RAG tools
        # self.tools = RAGAgentTools()
        
        # Define operation schemas
        self.ragagent_operations = copy.deepcopy(RAGAGENT_SCHEMA)
        for operation, details in self.ragagent_operations.items():
            function_name = details["function"]
            if isinstance(function_name, str):
                self.ragagent_operations[operation]["function"] = getattr(self, function_name)
        
    def _router(self, query):
        """Routes the query to the appropriate tool based on the command."""
        router_examples = ROUTER_EXAMPLES
        prompt = PROMPT_ROUTER.format(query=query, router_examples=router_examples)
        
        router_response = self.llm.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.2,
            max_tokens=100,
        )
        router_response = router_response.choices[0].message.content
        router_json_response = self.file_agent._extract_json(router_response)
        
        return router_json_response
    
    def _operations(self, json: dict, **kwargs) -> Dict:
        """
        Perform and handle execution of RAG operations.
        """
        try:
            # First parse the operation and its parameters
            operations = json.get("operation", [])
            if not isinstance(operations, list):
                operations = [operations]
            
            results = []
            for operation in operations:
                operation_type = operation.get("operation")
                parameters = operation.get("parameters", {})
                
                if operation_type not in self.ragagent_operations:
                    results.append({"error": f"Invalid operation: {operation_type}"})
                    continue
                
                # Execute the operation
                result = self.ragagent_operations[operation_type]["function"](**parameters)
                results.append(result)
            
            return {"results": results}
        
        except Exception as e:
            return {"error": str(e)}
            
    def _index_documents(self, documents):
        """Index new documents with given paths"""
        try:
            indexed_count = 0
            for path in documents:
                # Use FileAgent's _get_metadata_single for comprehensive metadata
                metadata = self.file_agent._get_metadata_single(path)
                if "error" in metadata:
                    logger.error(f"Error getting metadata for {path}: {metadata['error']}")
                    continue

                doc_id = metadata["file_hash"]
                
                # Skip if already indexed
                if doc_id in {doc["doc_id"] for doc in self.document_store.documents}:
                    logger.info(f"Document {doc_id} already indexed, skipping")
                    continue
                
                # Process document
                chunks, embeddings = self.document_store._process_document(path)
                
                # Add to FAISS index
                start_idex = self.document_store.index.ntotal
                self.document_store.index.add(embeddings)
                end_index = self.document_store.index.ntotal
                
                # Map the new indexes to a document id
                for i, idx in enumerate(range(start_idex, end_index)):
                    self.document_store.document_lookup[idx] = {
                        "doc_id": doc_id,
                        "chunk_index": i
                    }
                
                # Store document info
                self.document_store.documents.append({
                    "doc_id": doc_id,
                    "path": path,
                    "metadata": metadata,
                    "chunks": chunks
                })
                indexed_count += 1

            return {"message": f"Successfully indexed {indexed_count} documents"}
            
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            return {"error": str(e)}
    
    def _update_document(self, doc_id, path):
        return self.document_store._update_document(doc_id, path)
    
    def _save_state(self):
        try:
            self.document_store._save_state()
            return {"message": "State saved"}
        except Exception as e:
            return {"error": str(e)}
    
    def _load_state(self):   
        try:
            self.document_store._load_state()
            return {"message": "State loaded"}
        except Exception as e:
            return {"error": str(e)}
        
    def _view_document_store(self):
        return self.document_store._view_document_store()
        
    def _search_reasoning(self, query: str, k: int = 5) -> Dict:
        try:
            # Perform search on document store
            requery_count = 5 # Maximum number of requery attempts
            requery_attempts = 0
            query_history = []
            all_paths = []
            all_contexts = []
            
            while requery_attempts < requery_count:
                search_results = self.document_store._search(query, top_k=k)
                if not search_results:
                    return {"error": "No results found"}
                
                # Get the first result's context
                context = search_results[0]
                all_contexts.append(context)
                all_paths.append(context["document"]["path"])
                
                # Perform reasoning on search results
                few_shot_example = FEW_SHOT_EXAMPLES
                reasoning_prompt = PROMPT_REASONING.format(query=query, context=context, few_shot_example=few_shot_example)
                
                reasoning_response = self.llm.chat.completions.create(
                    messages=[{"role": "user", "content": reasoning_prompt}],
                    model="llama3-70b-8192",
                    temperature=0.2,
                    max_tokens=100,
                )
                
                reasoning_response = reasoning_response.choices[0].message.content
                reasoning_json_response = self.file_agent._extract_json(reasoning_response)
                reasoning_json_response["Response"]["file_path"] = context["document"]["path"]
                
                # Check if a requery is needed
                if requery_attempts < requery_count:
                    new_query = reasoning_json_response["Response"].get("Query", "")
                    if new_query:
                        query_history.append(query)
                        requery_attempts += 1
                        query = new_query
                        continue
                    else:
                        break
                else:
                    break
                
            final_response = reasoning_json_response
            final_response["Response"]["file_paths"] = list(set(all_paths))  # Remove duplicates
            final_response["Response"]["query_history"] = query_history
            final_response["Response"]["requery_attempts"] = requery_attempts
            final_response["Response"]["all_contexts"] = all_contexts
        
            return final_response
        
        except Exception as e:
            logger.error(f"Error in search reasoning: {str(e)}")
            return {"error": str(e)}

if __name__ == "__main__":
    # Example usage
    agent = RAGAgent(name="RAG Agent", description="This agent utilizes the RAG framework for question answering")
    
    # Example 1: Index documents
    test_file = "./test.txt"
    with open(test_file, "w") as f:
        f.write("This is a test document about artificial intelligence.")
    
    result = agent._index_documents([test_file])
    print("Index result:", result)
    
    # Example 2: Search and reason
    query = "What is artificial intelligence?"
    result = agent._search_reasoning(query)
    print("\nSearch result:", result)
    
    # Example 3: Save and load state
    save_result = agent._save_state()
    print("\nSave state result:", save_result)
    
    load_result = agent._load_state()
    print("Load state result:", load_result)
    
    # Clean up
    os.remove(test_file)