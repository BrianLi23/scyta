from collections import deque
from datetime import datetime
import json
from typing import Dict, Optional

class ChatHistory:
    _instance = None # What holds the instance of the class, we can think about it as a global object
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChatHistory, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the chat history"""
        self.max_conversations = 10
        self.conversation_history = deque(maxlen=self.max_conversations)
        self.operation_history = []
        self.operation_lookup = {}
        
    def add_conversation(self, user_input: str, response: Dict, planning: Optional[Dict] = None):
        """Add a conversation entry"""
        conversation_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "planning": planning,
            "response": response,
            "related_operations": [op["operation"] for op in self.operation_history[-len(response.get("operations", [])):]]
        }
        self.conversation_history.append(conversation_entry)
        
    def add_operation(self, response: Dict):
        """Add an operation entry"""
        for op in response.get("operations", []):
            operation_record = {
                "timestamp": datetime.now().isoformat(),
                "operation": op["operation"],
                "parameters": op.get("parameters", {}),
                # "status": status,
                # "result": result,
            }
            self.operation_history.append(operation_record)
            self.operation_lookup[operation_record["timestamp"]] = operation_record
        
    def get_recent_conversations(self, limit: int = 3) -> str:
        """Get recent conversation context"""
        recent = list(self.conversation_history)[-limit:] # Retrieve the last n conversations
        return "\n".join(
            f"User: {conv['user_input']}\n"
            f"Planning: {json.dumps(conv['planning'], indent=2)}\n"
            f"Response: {json.dumps(conv['response'], indent=2)}\n"
            f"Operations: {', '.join(conv['related_operations'])}"
            for conv in recent
        )
        
    # def save_state(self, filepath: str = "chat_history.json"):
    #     """Save history state to file"""
    #     state = {
    #         "conversations": list(self.conversation_history),
    #         "operations": self.operation_history
    #     }
    #     with open(filepath, 'w') as f:
    #         json.dump(state, f, indent=2)
            
    # def load_state(self, filepath: str = "chat_history.json"):
    #     """Load history state from file"""
    #     try:
    #         with open(filepath, 'r') as f:
    #             state = json.load(f)
    #             self.conversation_history = deque(state["conversations"], maxlen=10)
    #             self.operation_history = state["operations"]
    #             self.operation_lookup = {
    #                 op["timestamp"]: op for op in self.operation_history
    #             }
    #     except FileNotFoundError:
    #         pass

# Global instance
chat_history = ChatHistory()