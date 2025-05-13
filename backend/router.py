from llama_cpp import Llama
import os
import subprocess
from abc import ABC, abstractmethod
import time
from datetime import datetime
from pathlib import Path
import pwd
from typing import List, Dict
from backend.mainframe import BaseAgent, Tool
import hashlib
import mimetypes
from typing import Dict, List, Optional


class DecisionRouter:
    def __init__(self):
        self.chat_memory = []
        self.available_agents = {
            "internet": "InternetAgent",
            "file": "FileAgent",
            "rag": "RAGAgent"
        }

    def route_query(self, query: str) -> dict:
        self.chat_memory.append({"user": query})
        decision = None

        # Determine intent using simple keyword heuristics
        lowered = query.lower()
        if "recipe" in lowered:
            clarification = input(
                "It seems you asked about recipes. Should I search your local files or use a web search? (local/web): "
            ).strip()
            self.chat_memory.append({"clarification": clarification})
            if clarification.lower() == "local":
                decision = "file"
            elif clarification.lower() == "web":
                decision = "internet"
        elif any(word in lowered for word in ["rename", "directory", "file", "scan"]):
            decision = "file"
        elif any(word in lowered for word in ["index", "document", "update"]):
            decision = "rag"
        elif any(word in lowered for word in ["search", "google", "find"]):
            decision = "internet"

        # If ambiguous, ask for more details.
        if decision is None:
            print("I'm not sure what you're asking for.")
            clarification = input("Could you please provide more context? ").strip()
            self.chat_memory.append({"clarification": clarification})
            clarified = clarification.lower()
            if "local" in clarified or "file" in clarified:
                decision = "file"
            elif "document" in clarified or "index" in clarified:
                decision = "rag"
            elif "web" in clarified or "online" in clarified:
                decision = "internet"

        return {"agent": decision, "chat_memory": self.chat_memory}


def main():
    query = input("Enter your query: ").strip()
    router = DecisionRouter()
    decision = router.route_query(query)
    if decision["agent"]:
        print(f"Routing to the {decision['agent']} agent.")
    else:
        print("No suitable agent found based on your query.")
    print("Chat memory:", decision["chat_memory"])

