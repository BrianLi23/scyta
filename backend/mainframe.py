from llama_cpp import Llama
import os
import subprocess
from abc import ABC, abstractmethod
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.types import Tool, FunctionDeclaration
from google.genai.types import GenerateContentConfig

load_dotenv()

# Tool class definition
class Tool:
    def __init__(self, name, func, description):
        self.name = name # Name of the tool
        self.func = func # Functionality of the tool
        self.description = description # Description of the tool

# Defining the calculator function
def calculator(query):
    try:
        # Basic safety checks
        if any(word in query.lower() for word in ['import', 'eval', 'exec']):
            return "Invalid input"
        return eval(query)
    except Exception as e:
        return f"Error in calculation: {str(e)}"

class BaseAgent(ABC):
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.llm = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.memory = [] # Stores the previous querieres and responses
        self.tools = [Tool(name="Calculator", func=calculator, description="Useful for math calculations")]
