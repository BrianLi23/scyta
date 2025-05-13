from typing import List, Dict, Optional
import os
from pathlib import Path
import json
from datetime import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.agents.fileagent import FileAgent
import backend.ragagent.ragchunking as rc
from backend.agents.documentstoring import DocumentStore

class RAGAgentTools:
    def __init__(self):
        self.document_store = DocumentStore()
        self.file_agent = FileAgent(name="FileAgent", description="Agent for file operations")
    
    def index_documents(self, documents: List[str]) -> Dict:
        """Index new documents into the document store"""
        return self.document_store.index_documents(documents)
    
    def update_document(self, doc_id: str, path: str) -> Dict:
        """Update an existing document in the document store"""
        return self.document_store.update_document(doc_id, path)
    
    def save_state(self) -> Dict:
        """Save the current state of the document store"""
        try:
            self.document_store.save_state()
            return {"message": "State saved"}
        except Exception as e:
            return {"error": str(e)}
    
    def load_state(self) -> Dict:
        """Load the saved state of the document store"""
        try:
            self.document_store.load_state()
            return {"message": "State loaded"}
        except Exception as e:
            return {"error": str(e)}
    
    def view_document_store(self) -> Dict:
        """View the contents of the document store"""
        return self.document_store._view_document_store()
    
    def search_reasoning(self, query: str, k: int = 5) -> Dict:
        """Perform semantic search with reasoning on the document store"""
        try:
            requery_count = 5
            requery_attempts = 0
            query_history = []
            all_paths = []
            all_contexts = []
            
            while requery_attempts < requery_count:
                context = self.document_store.search(query, top_k=k)
                
                # Get reasoning from LLM
                reasoning_prompt = PROMPT_REASONING.format(
                    query=query, 
                    context=context, 
                    few_shot_example=FEW_SHOT_EXAMPLES
                )
                
                reasoning_response = self.llm.chat.completions.create(
                    messages=[{"role": "user", "content": reasoning_prompt}],
                    model="llama3-70b-8192",
                    temperature=0.2,
                    max_tokens=100,
                )
                
                reasoning_response = reasoning_response.choices[0].message.content
                reasoning_json_response = self.file_agent._extract_json(reasoning_response)
                reasoning_json_response["Response"]["file_path"] = context["path"]
                
                if requery_attempts < requery_count:
                    new_query = reasoning_json_response["Response"].get("Query", "")
                    if new_query:
                        requery_attempts += 1
                        query = new_query
                        continue
                    else:
                        break
                else:
                    break
            
            final_response = reasoning_json_response
            final_response["Response"]["query_history"] = query_history
            final_response["Response"]["requery_attempts"] = requery_attempts
            final_response["Response"]["all_contexts"] = all_contexts
            
            return final_response
            
        except Exception as e:
            return {"error": str(e)}
