import requests
from bs4 import BeautifulSoup
from backend.mainframe import BaseAgent, Tool
from backend.schemas.operation_schemas import FILEAGENT_SCHEMA

class InternetAgent(BaseAgent):
    def __init__(self, name="Internet Agent", description="Handles web search operations"):
        super().__init__(name, description)
        self.search_engine_url = "https://www.google.com/search?q="

        # Define operation schemas
        self.internetagent_operations = FILEAGENT_SCHEMA

        # Map operations to functions
        self.operations = {
            op_name: schema["function"]
            for op_name, schema in self.internetagent_operations.items()
        }

        # Register tool
        self.tools.append(Tool(
            name="WebSearch", 
            func=self.search_web, 
            description="Search the web using the query"
        ))
    
    def router(self, operation: str, **kwargs):
        """
        Routes the operation to the associated function.
        """
        func = self.operations.get(operation)
        if func:
            return func(**kwargs)
        return {"error": f"Unknown operation: {operation}"}

    def search_web(self, query: str):
        """
        Search the web with the given query and return parsed results.
        """
        search_url = self.search_engine_url + query
        response = requests.get(search_url)
        if response.status_code == 200:
            results = self.parse_results(response.text)
            return {"query": query, "results": results}
        return {"error": f"Request failed with status code {response.status_code}"}

    def parse_results(self, html_content: str):
        """
        Parse HTML content and return a list of result texts.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        results = []
        for g in soup.find_all('div', class_='BNeawe vvjwJb AP7Wnd'):
            results.append(g.get_text())
        return results

# Example usage
if __name__ == "__main__":
    agent = InternetAgent()
    # Using router to perform operation
    query = "Python programming"
    output = agent.router("search_web", query=query)
    print(output)