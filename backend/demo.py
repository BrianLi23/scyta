#!/usr/bin/env python3
"""
SCYTA Demo - Multi-Agent AI System Demonstration
==============================================

This demo showcases the integration of three intelligent agents, orchestrated
by a central decision router:
1. FileAgent - For file and directory operations.
2. RAGAgent - For document retrieval and Q&A from a knowledge base.
3. InternetAgent - For web search and information gathering.

It provides a command-line interface for users to interact with the system.
"""

import sys
from typing import Dict, Any
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add the project root to Python path for local imports
# This ensures that 'backend' can be found from the script's location
try:
    from backend.agents.fileagent import FileAgent
    from backend.agents.ragagent import RAGAgent
    from backend.agents.internetagent import InternetAgent
    from backend.router import DecisionRouter
    from backend.chathistory import chat_history
except ImportError:
    # Adjust path if script is run from a different directory
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    sys.path.insert(0, str(project_root))
    from backend.agents.fileagent import FileAgent
    from backend.agents.ragagent import RAGAgent
    from backend.agents.internetagent import InternetAgent
    from backend.router import DecisionRouter
    from backend.chathistory import chat_history


# Load environment variables from a .env file
load_dotenv()

# --- ANSI Color Codes for Terminal Output ---
class Colors:
    HEADER = '\033[96m\033[1m'      # Bright Cyan
    SUCCESS = '\033[92m\033[1m'     # Bright Green
    ERROR = '\033[91m\033[1m'       # Bright Red
    WARNING = '\033[93m\033[1m'     # Bright Yellow
    INFO = '\033[94m\033[1m'        # Bright Blue
    AGENT = '\033[95m\033[1m'       # Bright Magenta
    USER = '\033[97m\033[1m'        # Bright White
    RESET = '\033[0m'               # Reset all formatting

class ScytaDemo:
    """Main demo class for the SCYTA multi-agent system."""

    def __init__(self):
        """Initialize the demo with all agents and configuration."""
        self.agents: Dict[str, Any] = {}
        self.router: DecisionRouter = None
        self.conversation_history: list = []
        self.operation_history: list = []
        self.colors = Colors

    def print_banner(self):
        """Display the SCYTA welcome banner and instructions."""
        banner = f"""
{self.colors.HEADER}
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║              ███████╗ ██████╗██╗   ██╗████████╗ █████╗                        ║
║              ██╔════╝██╔════╝╚██╗ ██╔╝╚══██╔══╝██╔══██╗                       ║
║              ███████╗██║      ╚████╔╝    ██║   ███████║                       ║
║              ╚════██║██║       ╚██╔╝     ██║   ██╔══██║                       ║
║              ███████║╚██████╗   ██║      ██║   ██║  ██║                       ║
║              ╚══════╝ ╚═════╝   ╚═╝      ╚═╝   ╚═╝  ╚═╝                       ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
{self.colors.RESET}

{self.colors.WARNING}Available Agents:{self.colors.RESET}
  🗂️  {self.colors.AGENT}FileAgent{self.colors.RESET}    - File operations, directory management, file manipulation
  📚 {self.colors.AGENT}RAGAgent{self.colors.RESET}     - Document retrieval, Q&A, knowledge base queries
  🌐 {self.colors.AGENT}InternetAgent{self.colors.RESET} - Web search, information gathering

{self.colors.INFO}Commands:{self.colors.RESET}
  • Type your request naturally - the system will route to the best agent.
  • 'help'       - Show detailed help and examples.
  • 'history'    - View conversation history.
  • 'operations' - View detailed operation history.
  • 'agents'     - Show agent status and capabilities.
  • 'clear'      - Clear all history for the current session.
  • 'exit'|'quit' - End the session.

{self.colors.SUCCESS}Initializing system...{self.colors.RESET}
"""
        print(banner)

    def initialize_system(self):
        """Initialize all agents and the decision router with error handling."""
        agent_configs = [
            ('file', FileAgent, "File operations and directory management"),
            ('rag', RAGAgent, "Document retrieval and question answering"),
            ('internet', InternetAgent, "Web search and information gathering")
        ]

        for key, agent_class, description in agent_configs:
            agent_name = agent_class.__name__
            try:
                print(f"  {self.colors.INFO}→{self.colors.RESET} Initializing {agent_name}...", end="", flush=True)
                # Special handling for FileAgent's directory arguments
                if agent_class == FileAgent:
                    agent = agent_class(name=agent_name, description=description, directories=["~/Documents", "~/Downloads"])
                else:
                    agent = agent_class(name=agent_name, description=description)
                self.agents[key] = agent
                print(f" {self.colors.SUCCESS}✓{self.colors.RESET}")
            except Exception as e:
                print(f" {self.colors.ERROR}✗ Failed: {e}{self.colors.RESET}")
                self.agents[key] = None

        print(f"\n{self.colors.SUCCESS}System is ready. Please enter your request.{self.colors.RESET}\n")

    def show_help(self):
        """Display detailed help information with examples for each agent."""
        help_text = f"""
{self.colors.HEADER}📖 SCYTA Help & Examples{self.colors.RESET}
═══════════════════════════════════════════════════════════════

{self.colors.AGENT}🗂️  FileAgent Examples:{self.colors.RESET}
  • "Scan my Documents folder for all python files."
  • "Create a new file in my Downloads folder named 'report.txt' with the content 'This is a test'."
  • "Move all .pdf files from Downloads to Documents/PDFs."
  • "Delete the file named 'old_data.csv' from my Desktop."

{self.colors.AGENT}📚 RAGAgent Examples:{self.colors.RESET}
  • "Search for documents about machine learning."
  • "What does my research paper say about neural networks?"
  • "Summarize the content of 'technical_spec_v1.pdf'."

{self.colors.AGENT}🌐 InternetAgent Examples:{self.colors.RESET}
  • "Search the web for the latest news on large language models."
  • "Find a good tutorial on how to use the Python requests library."
  • "What is the current weather in New York City?"

{self.colors.INFO}💡 Tips:{self.colors.RESET}
  • The system automatically determines which agent to use based on your query.
  • Be as specific as possible for better results (e.g., provide full paths).
"""
        print(help_text)

    def show_agents_status(self):
        """Display the current status and capabilities of all agents."""
        status_text = f"\n{self.colors.HEADER}🤖 Agent Status & Capabilities{self.colors.RESET}\n"
        status_text += "═══════════════════════════════════════════════════════════════\n\n"
        for key, agent in self.agents.items():
            agent_name = agent.name if agent else f"{key.title()}Agent"
            if agent:
                status = f"{self.colors.SUCCESS}✅ Online{self.colors.RESET}"
                desc = agent.description
            else:
                status = f"{self.colors.ERROR}❌ Offline{self.colors.RESET}"
                desc = "Not available due to an initialization error."

            status_text += f"{self.colors.AGENT}{agent_name:<15}{self.colors.RESET} {status}\n"
            status_text += f"  └─ {desc}\n\n"

        router_status = f"{self.colors.SUCCESS}✅ Online{self.colors.RESET}" if self.router else f"{self.colors.ERROR}❌ Offline{self.colors.RESET}"
        status_text += f"{self.colors.INFO}Decision Router  {router_status}{self.colors.RESET}\n"
        print(status_text)

    def show_history(self):
        """Display the high-level conversation history."""
        if not self.conversation_history:
            print(f"{self.colors.INFO}📝 No conversation history yet.{self.colors.RESET}")
            return

        print(f"\n{self.colors.HEADER}📜 Conversation History{self.colors.RESET}\n" + "═" * 50)
        for i, entry in enumerate(self.conversation_history, 1):
            timestamp = entry['timestamp'].strftime("%H:%M:%S")
            summary_color = self.colors.SUCCESS if "Success" in entry['summary'] else self.colors.ERROR
            print(f"\n{self.colors.INFO}[{timestamp}] Query {i}:{self.colors.RESET}")
            print(f"  {self.colors.USER}User: {entry['query']}{self.colors.RESET}")
            print(f"  {self.colors.AGENT}Agent Used: {entry['agent_used']}{self.colors.RESET}")
            print(f"  {self.colors.INFO}Result: {summary_color}{entry['summary']}{self.colors.RESET}")
        print()

    def show_operations(self):
        """Display a detailed log of all operations executed by agents."""
        if not self.operation_history:
            print(f"{self.colors.INFO}⚙️ No operations have been executed yet.{self.colors.RESET}")
            return

        print(f"\n{self.colors.HEADER}⚙️ Detailed Operation History{self.colors.RESET}\n" + "═" * 50)
        for i, op in enumerate(self.operation_history, 1):
            timestamp = op['timestamp'].strftime("%H:%M:%S")
            status_icon = "✅" if op['success'] else "❌"
            op_name = op.get('operation', 'unknown').replace('_', ' ').title()

            print(f"\n{self.colors.INFO}[{timestamp}] Operation {i}: {op_name} {status_icon}{self.colors.RESET}")
            print(f"  {self.colors.USER}Triggered by: \"{op['query']}\"{self.colors.RESET}")

            params = op.get('parameters', {})
            if params:
                param_str = ", ".join(f"{k}='{str(v)[:50]}...'" if isinstance(v, str) and len(str(v)) > 50 else f"{k}='{v}'" for k, v in params.items())
                print(f"  {self.colors.INFO}Parameters: {self.colors.USER}{param_str}{self.colors.RESET}")
        print()

    def clear_history(self):
        """Clear all conversation and operation history for the session."""
        self.conversation_history.clear()
        self.operation_history.clear()
        chat_history.clear_history()  # Clear the shared history module as well
        print(f"{self.colors.SUCCESS}✅ Conversation and operation history has been cleared.{self.colors.RESET}")

    def _process_and_display_response(self, response: Dict[str, Any], agent_name: str, query: str):
        """
        Processes a structured agent response, displays it in a user-friendly
        format, and updates all history logs. This is the primary output handler.
        """
        if not response:
            print(f"{self.colors.ERROR}❌ Agent returned an empty response.{self.colors.RESET}")
            return

        # --- Calculate operation summary metrics ---
        operations = response.get("operations_executed", [])
        total_ops = len(operations)
        success_count = sum(1 for op in operations if not (isinstance(op.get("result", {}), dict) and "error" in op.get("result", {})))
        success = not response.get("error") and (not operations or success_count == total_ops)

        # --- Build Formatted Output with Operation Summary Header ---
        formatted_output = f"\n{self.colors.HEADER}🔍 Operation Summary{self.colors.RESET}\n"
        formatted_output += "═" * 50 + "\n"
        formatted_output += f"{self.colors.INFO}📝 Request:{self.colors.RESET} {query}\n"
        formatted_output += f"{self.colors.INFO}⚙️  Operations Executed:{self.colors.RESET} {total_ops}\n"
        formatted_output += f"{self.colors.INFO}📊 Status:{self.colors.RESET} {'✅ Success' if success else '❌ Errors occurred'}\n\n"

        # --- Agent Response Section ---
        formatted_output += f"{self.colors.AGENT}🤖 {agent_name} Response:{self.colors.RESET}\n"
        formatted_output += "═" * 50 + "\n"

        if "error" in response:
            formatted_output += f"{self.colors.ERROR}❌ Error: {response['error']}{self.colors.RESET}\n"
            
        else:
            if response.get('response'):
                formatted_output += f"{self.colors.SUCCESS}✅ Response:{self.colors.RESET}\n{response['response']}\n"
            else:
                formatted_output += f"{self.colors.WARNING}⚠️ No final result provided, but operations may have run.{self.colors.RESET}\n"

        # --- Update Histories and Display Operations ---
        if operations:
            formatted_output += f"\n{self.colors.INFO}⚙️ Operations Executed:{self.colors.RESET}\n"
            for op in operations:
                op_name = op.get("operation", "unknown_op")
                op_result = op.get("result", {})
                is_error = isinstance(op_result, dict) and "error" in op_result
                status_icon = "❌" if is_error else "✅"

                # Append to operation display string
                result_str = op_result.get('error') or op_result.get('status') or "Completed"
                formatted_output += f"  • {op_name}: {status_icon} {result_str}\n"

                # Update the persistent operation history log
                self.operation_history.append({
                    'timestamp': datetime.now(),
                    'query': query,
                    'operation': op_name,
                    'parameters': op.get("parameters"),
                    'result': op_result,
                    'success': not is_error
                })

        # Update the main conversation history
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'query': query,
            'agent_used': agent_name,
            'response': response,
            'summary': "Success" if success else "Error or No Result"
        })
        
        # Finally, print the entire formatted block
        print(formatted_output)


    def process_query(self, query: str):
        """
        Processes a user query:
        1. Routes the query to the appropriate agent using the DecisionRouter.
        2. Executes the agent's logic.
        3. Hands the response to the display handler.
        """
        if not query.strip():
            # if not self.router:
            #     print(f"{self.colors.ERROR}❌ Decision Router is offline. Cannot process query.{self.colors.RESET}")
            return

        print(f"\n{self.colors.INFO}Routing request...{self.colors.RESET}")

        try:
            # 1. Use the router to determine the best agent
            # The chat history is passed for context
            # agent_key = self.router.route(query, chat_history.get_history())
            agent_to_use = self.agents.get("file")
            agent_name = agent_to_use.name if agent_to_use else "Unknown Agent"

            # if not agent_to_use:
            #     response = {"error": f"Router selected agent '{agent_key}', but it is not available."}
            # else:
            #     print(f"{self.colors.SUCCESS}→ Routed to {self.colors.AGENT}{agent_name}{self.colors.SUCCESS}. Executing...{self.colors.RESET}")
                
                # 2. Execute the selected agent's main entrypoint
                # We assume a 'plan and execute' pattern for all agents for consistency
                # The agent's internal 'router' creates a plan
            plan = agent_to_use.router(query)
                # The 'operations' method executes the plan
            response = agent_to_use.operations(plan)

            # 3. Process and display the agent's response
            self._process_and_display_response(response, agent_name, query)

        except Exception as e:
            print(f"{self.colors.ERROR}❌ An unexpected error occurred during query processing: {e}{self.colors.RESET}")
            # Add a failure record to conversation history
            self.conversation_history.append({
                'timestamp': datetime.now(), 'query': query, 'agent_used': 'System',
                'response': {'error': str(e)}, 'summary': 'Critical System Error'
            })

    def run(self):
        """Main demo loop to accept and process user input."""
        self.print_banner()
        self.initialize_system()

        while True:
            try:
                prompt = f"{self.colors.USER}💬 You: {self.colors.RESET}"
                user_input = input(prompt).strip()

                if not user_input:
                    continue

                cmd = user_input.lower()
                if cmd in ['exit', 'quit', 'bye']:
                    print(f"\n{self.colors.SUCCESS}👋 Goodbye!{self.colors.RESET}")
                    break
                elif cmd == 'help':
                    self.show_help()
                elif cmd == 'history':
                    self.show_history()
                elif cmd == 'operations':
                    self.show_operations()
                elif cmd == 'agents':
                    self.show_agents_status()
                elif cmd == 'clear':
                    self.clear_history()
                else:
                    self.process_query(user_input)

            except KeyboardInterrupt:
                print(f"\n{self.colors.WARNING}Caught interrupt. Exiting...{self.colors.RESET}")
                break
            except Exception as e:
                print(f"\n{self.colors.ERROR}A fatal error occurred in the main loop: {e}{self.colors.RESET}")
                break

def main():
    """Entry point for the SCYTA demo application."""
    try:
        demo = ScytaDemo()
        demo.run()
    except Exception as e:
        print(f"{Colors.ERROR}Failed to start SCYTA demo: {e}{Colors.RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()