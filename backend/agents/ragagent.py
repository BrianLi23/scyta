from typing import List, Dict, Optional
from backend.mainframe import BaseAgent
from backend.agents.fileagent import FileAgent
from backend.agents.documentstoring import DocumentStore
from backend.schemas.operation_schemas import RAGAGENT_SCHEMA
from backend.prompts.ragagent_prompt import PROMPT_ROUTER, ROUTER_EXAMPLES, FEW_SHOT_EXAMPLES, PROMPT_REASONING
# from backend.tools.ragagent_tools import RAGAgentTools
import copy
# import logging

# Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# Structure
# Query preprocessing --> query vectorization --> retreival phase --> prompt augementation --> LLM generation --> response postprocessing
class RAGAgent(BaseAgent):
    def __init__(self, name, description):
        super().__init__(name, description)
        
        self.document_store = DocumentStore()
        self.file_agent = FileAgent(name="FileAgent", description="Agent for file operations")
        
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
            model="llama-4-scout-17b-16e-instruct",
            temperature=0.2,
            max_completion_tokens=100,
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
        """Wrapper for DocumentStore's document indexing functionality"""
        try:
            return self.document_store._index_documents(documents)
        except Exception as e:
            # logger.error(f"Error indexing documents: {str(e)}")
            return {"error": str(e)}
    
    def _update_document(self, doc_id, path):
        """Wrapper for DocumentStore's document update functionality"""
        return self.document_store._update_document(doc_id, path)
    
    def _save_state(self):
        """Wrapper for DocumentStore's state saving functionality"""
        try:
            return self.document_store._save_state()
        except Exception as e:
            return {"error": str(e)}
    
    def _load_state(self):   
        """Wrapper for DocumentStore's state loading functionality"""
        try:
            return self.document_store._load_state()
        except Exception as e:
            return {"error": str(e)}
        
    def _view_document_store(self):
        """Wrapper for DocumentStore's view functionality with enhanced formatting and table display"""
        try:
            store_info = self.document_store._view_document_store()
            
            if "error" in store_info:
                return store_info
                
            # Format the response
            formatted_response = {
                "store_summary": {
                    "total_documents": store_info["document_count"],
                    "total_lookup_entries": store_info["lookup_entries"],
                    "tables": store_info["tables"]
                },
                "documents": []
            }
            
            # Print store summary
            print("\n=== Document Store Summary ===")
            print(f"Total Documents: {store_info['document_count']}")
            print(f"Total Lookup Entries: {store_info['lookup_entries']}")
            print(f"Tables: {', '.join(store_info['tables'])}")
            print("============================\n")
            
            # Format and print each document's information
            for doc in store_info["documents"]:
                print(f"\n=== Document: {doc['doc_id']} ===")
                print(f"Path: {doc['path']}")
                print("\nMetadata:")
                print(f"  File Type: {doc['metadata'].get('file_type', 'unknown')}")
                print(f"  File Size: {doc['metadata'].get('file_size', 'unknown')}")
                print(f"  Last Modified: {doc['metadata'].get('last_modified', 'unknown')}")
                print(f"  File Hash: {doc['metadata'].get('file_hash', 'unknown')}")
                
                print("\nChunks:")
                print("┌────────────┬────────────────────────────────────────────────────────────┐")
                print("│ Chunk #    │ Content                                                    │")
                print("├────────────┼────────────────────────────────────────────────────────────┤")
                
                # Get chunks from the correct field
                chunks = doc.get('first_chunks', [])
                for i, chunk in enumerate(chunks, 1):
                    # Truncate chunk content if too long
                    chunk_preview = chunk[:60] + "..." if len(chunk) > 60 else chunk
                    print(f"│ {i:<10} │ {chunk_preview:<60} │")
                
                print("└────────────┴────────────────────────────────────────────────────────────┘")
                print("\n" + "="*80 + "\n")
                
                # Store in formatted response
                formatted_doc = {
                    "doc_id": doc["doc_id"],
                    "path": doc["path"],
                    "metadata": {
                        "file_type": doc["metadata"].get("file_type", "unknown"),
                        "file_size": doc["metadata"].get("file_size", "unknown"),
                        "last_modified": doc["metadata"].get("last_modified", "unknown"),
                        "file_hash": doc["metadata"].get("file_hash", "unknown")
                    },
                    "chunks": chunks  # Store the chunks we retrieved
                }
                formatted_response["documents"].append(formatted_doc)
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error formatting document store view: {str(e)}")
            return {"error": str(e)}
        
    def _search_reasoning(self, query: str, k: int = 5) -> Dict:
        """Enhanced search with reasoning capabilities"""
        try:
            # Perform search on document store
            requery_count = 3 # Maximum number of requery attempts
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
                    model="llama-4-scout-17b-16e-instruct",
                    temperature=0.2,
                    max_completion_tokens=100,
                )
                
                reasoning_response = reasoning_response.choices[0].message.content
                print(reasoning_response)
                reasoning_json_response = self.file_agent._extract_json(reasoning_response)
                print(reasoning_json_response)
                
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
    
    result = agent._index_documents(["index_test/test.txt", "index_test/test2.txt", "index_test/test3.txt"])
    print("Index result:", result)
    
    # Example 2: Search and reason
    query = "What is the capital of France?"
    result = agent._search_reasoning(query)
    print("\nSearch result:", result)
    
    # Example 2.1: View document store
    view_result = agent._view_document_store()
    print("\nView document store result:", view_result)
    
    # Example 3: Save and load state
    save_result = agent._save_state()
    print("\nSave state result:", save_result)
    
    load_result = agent._load_state()
    print("Load state result:", load_result)
