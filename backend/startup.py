from typing import Dict, Optional
from backend.router import DecisionRouter
from backend.mainframe import BaseAgent
from backend.agents.fileagent import FileAgent
from backend.agents.internet_agent import InternetAgent
from backend.agents.rag_agent import RAGAgent
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AgentManager:
    def __init__(self):
        self.router = DecisionRouter()
        self.agents: Dict[str, BaseAgent] = {}
        self.initialize_agents()
        
    def initialize_agents(self):
        """Initialize all available agents"""
        try:
            # Initialize FileAgent with required parameters
            self.agents["file"] = FileAgent(
                name="File Agent",
                description="Handles file operations and management",
                directories=["~/Documents", "~/Downloads", "~/Desktop"]
            )
            self.agents["internet"] = InternetAgent()
            self.agents["rag"] = RAGAgent()
            logger.info("Successfully initialized all agents")
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            raise

    def process_user_input(self, user_input: str) -> Optional[str]:
        """Process user input through the router and appropriate agent"""
        try:
            # Route the query to determine which agent should handle it
            routing_decision = self.router.route_query(user_input)
            selected_agent = routing_decision["agent"]
            
            if not selected_agent:
                return "I couldn't determine which agent should handle your request. Please try rephrasing."
            
            if selected_agent not in self.agents:
                return f"Error: Agent {selected_agent} is not available."
            
            # Get the appropriate agent and process the request
            agent = self.agents[selected_agent]
            response = agent.process_request(user_input)
            return response
            
        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}")
            return f"An error occurred while processing your request: {str(e)}"

def main():
    """Main entry point for the application"""
    try:
        agent_manager = AgentManager()
        logger.info("Agent system initialized successfully")
        
        print("\nWelcome to the Agent System!")
        print("You can ask questions or give instructions, and I'll route them to the appropriate agent.")
        print("Type 'exit' to quit.\n")
        
        while True:
            user_input = input("What would you like me to help you with? ").strip()
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
                
            response = agent_manager.process_user_input(user_input)
            print(f"\nResponse: {response}\n")
            
    except Exception as e:
        logger.error(f"Fatal error in main loop: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
