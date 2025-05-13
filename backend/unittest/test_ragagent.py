import unittest
import tempfile
import shutil
import os
from backend.agents.ragagent import RAGAgent

class TestRAGAgent(unittest.TestCase):
    def setUp(self):
        self.agent = RAGAgent(name="RAG Agent", description="Test RAG Agent")
        self.test_dir = tempfile.mkdtemp()
        
        # Create test files
        self.test_file1 = os.path.join(self.test_dir, "test1.txt")
        self.test_file2 = os.path.join(self.test_dir, "test2.txt")
        
        with open(self.test_file1, "w") as f:
            f.write("This is a test document about artificial intelligence.")
        with open(self.test_file2, "w") as f:
            f.write("Machine learning is a subset of AI.")
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_index_documents(self):
        result = self.agent._index_documents([self.test_file1, self.test_file2])
        self.assertNotIn("error", result)
        
        # Verify documents are indexed
        store_view = self.agent._view_document_store()
        self.assertGreater(len(store_view), 0)
    
    def test_search_reasoning(self):
        # First index documents
        self.agent._index_documents([self.test_file1, self.test_file2])
        
        # Test search
        result = self.agent.search_reasoning("What is artificial intelligence?")
        self.assertNotIn("error", result)
        self.assertIn("Response", result["Response"])
        self.assertIn("file_paths", result["Response"])
    
    def test_operations(self):
        # Test index operation
        index_op = {
            "operation": [{
                "operation": "index",
                "parameters": {
                    "documents": [self.test_file1, self.test_file2]
                }
            }]
        }
        result = self.agent.operations(index_op)
        self.assertNotIn("error", result)
        
        # Test search operation
        search_op = {
            "operation": [{
                "operation": "search",
                "parameters": {
                    "query": "What is AI?",
                    "k": 2
                }
            }]
        }
        result = self.agent.operations(search_op)
        self.assertNotIn("error", result)
    
    def test_save_load_state(self):
        # Index documents
        self.agent._index_documents([self.test_file1, self.test_file2])
        
        # Save state
        save_result = self.agent._save_state()
        self.assertNotIn("error", save_result)
        
        # Create new agent instance
        new_agent = RAGAgent(name="New RAG Agent", description="Test RAG Agent")
        
        # Load state
        load_result = new_agent._load_state()
        self.assertNotIn("error", load_result)
        
        # Verify documents are loaded
        store_view = new_agent._view_document_store()
        self.assertGreater(len(store_view), 0)
    
    def test_update_document(self):
        # First index a document
        self.agent._index_documents([self.test_file1])
        
        # Create a new version of the document
        with open(self.test_file2, "w") as f:
            f.write("Updated content about AI and machine learning.")
        
        # Update the document
        result = self.agent._update_document(self.test_file1, self.test_file2)
        self.assertNotIn("error", result)
        
        # Verify update
        store_view = self.agent._view_document_store()
        self.assertGreater(len(store_view), 0)

if __name__ == "__main__":
    unittest.main() 