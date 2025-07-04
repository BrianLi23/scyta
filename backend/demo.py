#!/usr/bin/env python3
"""
SCYTA Demo - Multi-Agent AI System Demonstration
==============================================

This demo showcases the integration of three intelligent agents:
1. FileAgent - File and directory operations
2. RAGAgent - Document retrieval and Q&A
3. InternetAgent - Web search capabilities

Author: SCYTA Team
"""

import os
import sys
import json
import time
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import textwrap
import pprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Color codes for terminal output (ANSI escape sequences)
class Colors:
    HEADER = '\033[96m\033[1m'      # Bright Cyan
    SUCCESS = '\033[92m\033[1m'     # Bright Green  
    ERROR = '\033[91m\033[1m'       # Bright Red
    WARNING = '\033[93m\033[1m'     # Bright Yellow
    INFO = '\033[94m\033[1m'        # Bright Blue
    AGENT = '\033[95m\033[1m'       # Bright Magenta
    USER = '\033[97m\033[1m'        # Bright White
    RESET = '\033[0m'               # Reset all formatting

# Add the project root to Python path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent if current_dir.name == 'backend' else current_dir
sys.path.insert(0, str(project_root))

try:
    from backend.agents.fileagent import FileAgent
    from backend.agents.ragagent import RAGAgent  
    from backend.agents.internetagent import InternetAgent
    from backend.router import DecisionRouter
    from backend.chathistory import chat_history
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all dependencies are installed and paths are correct.")
    sys.exit(1)


class ScytaDemo:
    """Main demo class for the SCYTA multi-agent system."""
    
    def __init__(self):
        """Initialize the demo with all agents and configuration."""
        self.width = 80
        self.agents = {}
        self.router = None
        self.conversation_history = []
        self.session_start = datetime.now()
        
        # Color scheme using ANSI codes
        self.colors = {
            'header': Colors.HEADER,
            'success': Colors.SUCCESS,
            'error': Colors.ERROR,
            'warning': Colors.WARNING,
            'info': Colors.INFO,
            'agent': Colors.AGENT,
            'user': Colors.USER,
            'reset': Colors.RESET
        }
        
    def print_banner(self):
        """Display the SCYTA banner."""
        banner = f"""
{self.colors['header']}
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║    ███████╗ ██████╗██╗   ██╗████████╗ █████╗                                 ║
║    ██╔════╝██╔════╝╚██╗ ██╔╝╚══██╔══╝██╔══██╗                                ║
║    ███████╗██║      ╚████╔╝    ██║   ███████║                                ║
║    ╚════██║██║       ╚██╔╝     ██║   ██╔══██║                                ║
║    ███████║╚██████╗   ██║      ██║   ██║  ██║                                ║
║    ╚══════╝ ╚═════╝   ╚═╝      ╚═╝   ╚═╝  ╚═╝                                ║
║                                                                               ║
║              Smart Cognitive Your Task Assistant                              ║
║                   Multi-Agent AI System Demo                                  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
{self.colors['reset']}

{self.colors['info']}Welcome to SCYTA - Your intelligent multi-agent assistant!{self.colors['reset']}

{self.colors['warning']}Available Agents:{self.colors['reset']}
  🗂️  {self.colors['agent']}FileAgent{self.colors['reset']}    - File operations, directory management, file manipulation
  📚 {self.colors['agent']}RAGAgent{self.colors['reset']}     - Document retrieval, Q&A, knowledge base queries  
  🌐 {self.colors['agent']}InternetAgent{self.colors['reset']} - Web search, information gathering

{self.colors['info']}Commands:{self.colors['reset']}
  • Type your request naturally - the system will route to appropriate agents
  • 'help' - Show detailed help and examples
  • 'history' - View conversation history
  • 'agents' - Show agent status and capabilities
  • 'clear' - Clear conversation history
  • 'exit' or 'quit' - End the session

{self.colors['success']}Initializing agents...{self.colors['reset']}
"""
        print(banner)
        
    def initialize_agents(self):
        """Initialize all agents with error handling."""
        agent_configs = [
            ('FileAgent', 'file', FileAgent, 'File operations and directory management'),
            ('RAGAgent', 'rag', RAGAgent, 'Document retrieval and question answering'), 
            ('InternetAgent', 'internet', InternetAgent, 'Web search and information gathering')
        ]
        
        for agent_name, agent_key, agent_class, description in agent_configs:
            try:
                print(f"  {self.colors['info']}→{self.colors['reset']} Initializing {agent_name}...", end="")
                
                if agent_class == FileAgent:
                    agent = agent_class(
                        name=agent_name,
                        description=description,
                        directories=["~/Documents", "~/Downloads"]
                    )
                else:
                    agent = agent_class(name=agent_name, description=description)
                    
                self.agents[agent_key] = agent
                print(f" {self.colors['success']}✓{self.colors['reset']}")
                
            except Exception as e:
                print(f" {self.colors['error']}✗ Failed: {str(e)}{self.colors['reset']}")
                self.agents[agent_key] = None
        
        # Initialize router
        try:
            print(f"  {self.colors['info']}→{self.colors['reset']} Initializing Decision Router...", end="")
            self.router = DecisionRouter()
            print(f" {self.colors['success']}✓{self.colors['reset']}")
        except Exception as e:
            print(f" {self.colors['error']}✗ Failed: {str(e)}{self.colors['reset']}")
            
        print(f"\n{self.colors['success']}All agents initialized successfully!{self.colors['reset']}\n")
        
    def format_agent_response(self, response: Dict, agent_name: str) -> str:
        """Format agent response for user-friendly display."""
        formatted = f"\n{self.colors['agent']}🤖 {agent_name} Response:{self.colors['reset']}\n"
        formatted += "═" * 50 + "\n"
        
        if isinstance(response, dict):
            # Handle different response types
            if "error" in response:
                formatted += f"{self.colors['error']}❌ Error: {response['error']}{self.colors['reset']}\n"
            elif "operations" in response:
                formatted += self.format_operations(response["operations"])
            elif "results" in response:
                formatted += self.format_results(response["results"])
            elif "message" in response:
                formatted += f"{self.colors['success']}✅ {response['message']}{self.colors['reset']}\n"
            else:
                formatted += self.format_generic_response(response)
        elif isinstance(response, list):
            formatted += self.format_list_response(response)
        else:
            formatted += f"{self.colors['info']}{str(response)}{self.colors['reset']}\n"
            
        return formatted
        
    def format_operations(self, operations: List[Dict]) -> str:
        """Format operation lists in a readable way."""
        if not operations:
            return f"{self.colors['info']}No operations to display.{self.colors['reset']}\n"
            
        formatted = f"{self.colors['info']}📋 Planned Operations:{self.colors['reset']}\n\n"
        
        for i, op in enumerate(operations, 1):
            formatted += f"{self.colors['warning']}  {i}. {op.get('operation', 'Unknown').title()}{self.colors['reset']}\n"
            
            # Format parameters
            params = op.get('parameters', {})
            if params:
                formatted += f"     {self.colors['info']}Parameters:{self.colors['reset']}\n"
                for key, value in params.items():
                    if isinstance(value, str) and len(value) > 50:
                        value = value[:47] + "..."
                    formatted += f"       • {key}: {self.colors['user']}{value}{self.colors['reset']}\n"
            
            # Show post-processing if available  
            if op.get('post_processing'):
                formatted += f"     {self.colors['info']}Post-processing: {op['post_processing']}{self.colors['reset']}\n"
                
            formatted += "\n"
            
        return formatted
        
    def format_results(self, results: List[Dict]) -> str:
        """Format result lists in a readable way."""
        if not results:
            return f"{self.colors['info']}No results to display.{self.colors['reset']}\n"
            
        formatted = f"{self.colors['success']}📊 Results:{self.colors['reset']}\n\n"
        
        for i, result in enumerate(results, 1):
            if isinstance(result, dict):
                if "error" in result:
                    formatted += f"{self.colors['error']}  {i}. Error: {result['error']}{self.colors['reset']}\n"
                elif "message" in result:
                    formatted += f"{self.colors['success']}  {i}. {result['message']}{self.colors['reset']}\n"
                elif "metadatas" in result:
                    formatted += f"  {i}. {self.colors['info']}Found {len(result['metadatas'])} items{self.colors['reset']}\n"
                    if result.get('total_files'):
                        formatted += f"     Total files: {result['total_files']}\n"
                    if result.get('total_size'):
                        formatted += f"     Total size: {self._format_size(result['total_size'])}\n"
                else:
                    formatted += f"  {i}. {self.colors['info']}{str(result)[:100]}{'...' if len(str(result)) > 100 else ''}{self.colors['reset']}\n"
            else:
                formatted += f"  {i}. {self.colors['info']}{str(result)}{self.colors['reset']}\n"
                
        return formatted + "\n"
        
    def format_generic_response(self, response: Dict) -> str:
        """Format generic dictionary responses."""
        formatted = ""
        for key, value in response.items():
            if key in ['timestamp', 'operation_id']:
                continue
                
            formatted += f"{self.colors['info']}📌 {key.title().replace('_', ' ')}:{self.colors['reset']}\n"
            
            if isinstance(value, (dict, list)):
                formatted += f"   {self._format_nested_data(value, indent=3)}\n"
            else:
                formatted += f"   {self.colors['user']}{value}{self.colors['reset']}\n"
                
        return formatted
        
    def format_list_response(self, response: List) -> str:
        """Format list responses."""
        formatted = f"{self.colors['info']}📋 Results ({len(response)} items):{self.colors['reset']}\n\n"
        
        for i, item in enumerate(response, 1):
            if isinstance(item, dict):
                if "error" in item:
                    formatted += f"{self.colors['error']}  {i}. Error: {item['error']}{self.colors['reset']}\n"
                elif "message" in item:
                    formatted += f"{self.colors['success']}  {i}. {item['message']}{self.colors['reset']}\n"
                else:
                    formatted += f"  {i}. {self._format_nested_data(item, indent=5)}\n"
            else:
                formatted += f"  {i}. {self.colors['user']}{item}{self.colors['reset']}\n"
                
        return formatted
        
    def _format_nested_data(self, data, indent=0) -> str:
        """Helper to format nested dictionaries and lists."""
        spaces = " " * indent
        if isinstance(data, dict):
            formatted = ""
            for key, value in data.items():
                if isinstance(value, (dict, list)) and value:
                    formatted += f"{spaces}{key}: {self._format_nested_data(value, indent + 2)}\n"
                else:
                    formatted += f"{spaces}{key}: {self.colors['user']}{value}{self.colors['reset']}\n"
            return formatted
        elif isinstance(data, list):
            return f"[{len(data)} items]"
        else:
            return str(data)
            
    def _format_size(self, size: int) -> str:
        """Format file size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}PB"
        
    def show_help(self):
        """Display detailed help information."""
        help_text = f"""
{self.colors['header']}📖 SCYTA Help & Examples{self.colors['reset']}
═══════════════════════════════════════════════════════════════

{self.colors['agent']}🗂️  FileAgent Examples:{self.colors['reset']}
  • "scan my Documents folder"
  • "create a new file called test.txt with hello world content"
  • "move all .pdf files from Downloads to Documents" 
  • "delete the file named old_data.csv"
  • "rename photo.jpg to vacation_photo.jpg"
  • "copy report.docx to the backup folder"

{self.colors['agent']}📚 RAGAgent Examples:{self.colors['reset']}
  • "search for documents about machine learning"
  • "what does my research paper say about neural networks?"
  • "find information about project requirements"
  • "summarize the content of technical_doc.pdf"

{self.colors['agent']}🌐 InternetAgent Examples:{self.colors['reset']}
  • "search the web for latest AI news"
  • "find information about Python programming"
  • "what's the weather like today?"
  • "search for tutorial on machine learning"

{self.colors['info']}💡 Tips:{self.colors['reset']}
  • Be specific about file paths and names
  • The system automatically determines which agent to use
  • You can combine multiple operations in one request
  • Use 'history' to see previous operations
  • Type 'agents' to check agent status
"""
        print(help_text)
        
    def show_agents_status(self):
        """Display current agent status and capabilities."""
        status_text = f"""
{self.colors['header']}🤖 Agent Status & Capabilities{self.colors['reset']}
═══════════════════════════════════════════════════════════════

"""
        for agent_key, agent in self.agents.items():
            if agent:
                status = f"{self.colors['success']}✅ Online{self.colors['reset']}"
                capabilities = "Fully operational"
            else:
                status = f"{self.colors['error']}❌ Offline{self.colors['reset']}"
                capabilities = "Not available"
                
            agent_name = agent_key.title() + "Agent"
            status_text += f"{self.colors['agent']}{agent_name:<15}{self.colors['reset']} {status}\n"
            status_text += f"  └─ {capabilities}\n\n"
            
        # Router status
        router_status = f"{self.colors['success']}✅ Online{self.colors['reset']}" if self.router else f"{self.colors['error']}❌ Offline{self.colors['reset']}"
        status_text += f"{self.colors['info']}Decision Router  {router_status}{self.colors['reset']}\n"
        
        print(status_text)
        
    def show_history(self):
        """Display conversation history."""
        if not self.conversation_history:
            print(f"{self.colors['info']}📝 No conversation history yet.{self.colors['reset']}")
            return
            
        print(f"\n{self.colors['header']}📜 Conversation History{self.colors['reset']}")
        print("═" * 50)
        
        for i, entry in enumerate(self.conversation_history, 1):
            timestamp = entry['timestamp'].strftime("%H:%M:%S")
            print(f"\n{self.colors['info']}[{timestamp}] Query {i}:{self.colors['reset']}")
            print(f"  {self.colors['user']}User: {entry['query']}{self.colors['reset']}")
            print(f"  {self.colors['agent']}Agent: {entry['agent_used']}{self.colors['reset']}")
            
            if 'summary' in entry:
                print(f"  {self.colors['info']}Result: {entry['summary']}{self.colors['reset']}")
                
        print()
        
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()
        if hasattr(chat_history, 'clear_history'):
            chat_history.clear_history()
        print(f"{self.colors['success']}✅ Conversation history cleared.{self.colors['reset']}")
        
    def process_query(self, query: str) -> None:
        """Process user query and route to appropriate agent."""
        if not query.strip():
            return
            
        print(f"\n{self.colors['info']}🔄 Processing your request...{self.colors['reset']}")
        
        # Simple routing logic for demo
        query_lower = query.lower()
        agent_used = None
        response = None
        
        try:
            # Determine which agent to use based on keywords
            if any(keyword in query_lower for keyword in ['file', 'folder', 'directory', 'create', 'delete', 'move', 'copy', 'rename', 'scan']):
                agent_used = 'FileAgent'
                if self.agents['file']:
                    planning_response = self.agents['file'].router(query)
                    if 'operations' in planning_response:
                        response = self.agents['file'].operations(planning_response)
                    else:
                        response = planning_response
                else:
                    response = {"error": "FileAgent not available"}
                    
            elif any(keyword in query_lower for keyword in ['search web', 'internet', 'online', 'web search', 'google']):
                agent_used = 'InternetAgent'
                if self.agents['internet']:
                    response = self.agents['internet'].search_web(query)
                else:
                    response = {"error": "InternetAgent not available"}
                    
            elif any(keyword in query_lower for keyword in ['document', 'rag', 'knowledge', 'search document', 'find in']):
                agent_used = 'RAGAgent'  
                if self.agents['rag']:
                    router_response = self.agents['rag']._router(query)
                    response = self.agents['rag']._operations(router_response)
                else:
                    response = {"error": "RAGAgent not available"}
                    
            else:
                # Default to FileAgent for general requests
                agent_used = 'FileAgent'
                if self.agents['file']:
                    planning_response = self.agents['file'].router(query)
                    if 'operations' in planning_response:
                        response = self.agents['file'].operations(planning_response)
                    else:
                        response = planning_response
                else:
                    response = {"error": "No suitable agent available"}
                    
            # Display response
            if response:
                formatted_response = self.format_agent_response(response, agent_used)
                print(formatted_response)
                
                # Store in history
                summary = "Success" if not any(key in str(response) for key in ['error', 'Error']) else "Error occurred"
                self.conversation_history.append({
                    'timestamp': datetime.now(),
                    'query': query,
                    'agent_used': agent_used,
                    'response': response,
                    'summary': summary
                })
            else:
                print(f"{self.colors['error']}❌ No response received from agent.{self.colors['reset']}")
                
        except Exception as e:
            print(f"{self.colors['error']}❌ Error processing query: {str(e)}{self.colors['reset']}")
            
    def _determine_agent_for_query(self, query: str) -> str:
        """Determine which agent should handle the query (for demo purposes)."""
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in ['file', 'folder', 'directory', 'create', 'delete', 'move', 'copy', 'rename', 'scan']):
            return "FileAgent"
        elif any(keyword in query_lower for keyword in ['search web', 'internet', 'online', 'web search', 'google']):
            return "InternetAgent"
        elif any(keyword in query_lower for keyword in ['document', 'rag', 'knowledge', 'search document', 'find in']):
            return "RAGAgent"
        else:
            return "FileAgent (default)"
            
    def _mock_operations_for_query(self, query: str) -> str:
        """Generate mock operations for demo purposes."""
        query_lower = query.lower()
        if 'scan' in query_lower:
            return "Scan directory → List files → Format metadata"
        elif 'create' in query_lower:
            return "Validate path → Create file → Confirm creation"
        elif 'search' in query_lower and 'web' in query_lower:
            return "Parse query → Web search → Format results"
        elif 'search' in query_lower:
            return "Vectorize query → Search documents → Rank results"
        else:
            return "Parse request → Plan operations → Execute → Report results"
            
    def run(self):
        """Main demo loop."""
        self.print_banner()
        self.initialize_agents()
        
        print(f"{self.colors['success']}🚀 SCYTA is ready! Type your request or 'help' for examples.{self.colors['reset']}\n")
        
        while True:
            try:
                # Get user input
                prompt = f"{self.colors['user']}💬 You: {self.colors['reset']}"
                user_input = input(prompt).strip()
                
                if not user_input:
                    continue
                    
                # Handle special commands
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print(f"\n{self.colors['success']}👋 Thank you for using SCYTA! Goodbye!{self.colors['reset']}")
                    break
                    
                elif user_input.lower() == 'help':
                    self.show_help()
                    
                elif user_input.lower() == 'history':
                    self.show_history()
                    
                elif user_input.lower() == 'agents':
                    self.show_agents_status()
                    
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    
                else:
                    # Process the query
                    self.process_query(user_input)
                    
            except KeyboardInterrupt:
                print(f"\n\n{self.colors['warning']}⚠️  Interrupted by user. Type 'exit' to quit properly.{self.colors['reset']}")
                continue
                
            except EOFError:
                print(f"\n{self.colors['success']}👋 Session ended. Goodbye!{self.colors['reset']}")
                break
                
            except Exception as e:
                print(f"{self.colors['error']}❌ Unexpected error: {str(e)}{self.colors['reset']}")
                continue


def main():
    """Entry point for the demo."""
    try:
        demo = ScytaDemo()
        demo.run()
    except Exception as e:
        print(f"Failed to start SCYTA demo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
