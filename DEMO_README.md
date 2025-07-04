# SCYTA Demo - Multi-Agent AI System

## Overview

SCYTA (Smart Cognitive Your Task Assistant) is a multi-agent AI system that combines three specialized agents to handle various tasks:

- **🗂️ FileAgent**: File operations, directory management, file manipulation
- **📚 RAGAgent**: Document retrieval, question answering, knowledge base queries  
- **🌐 InternetAgent**: Web search, information gathering

## Quick Start

### Prerequisites

1. Python 3.8 or higher
2. All dependencies installed (see `requirements.txt`)
3. GROQ API key set in environment

### Environment Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```
   GROQ_API=your_groq_api_key_here
   ```

### Running the Demo

**Option 1: Using the launcher (recommended)**
```bash
python run_demo.py
```

**Option 2: Direct execution**
```bash
python backend/demo.py
```

## Demo Features

### Interactive Commands

- **help** - Show detailed help and examples
- **history** - View conversation history  
- **agents** - Show agent status and capabilities
- **clear** - Clear conversation history
- **exit/quit** - End the session

### Example Queries

**FileAgent Examples:**
- "scan my Documents folder"
- "create a new file called test.txt with hello world content"
- "move all .pdf files from Downloads to Documents"
- "delete the file named old_data.csv"
- "rename photo.jpg to vacation_photo.jpg"

**RAGAgent Examples:**
- "search for documents about machine learning"
- "what does my research paper say about neural networks?"
- "find information about project requirements"

**InternetAgent Examples:**
- "search the web for latest AI news"
- "find information about Python programming"
- "what's the weather like today?"

## Demo Interface

The demo features a beautiful text-based interface with:

- **🎨 Color-coded output** for different types of information
- **📋 Formatted operation displays** showing planned actions
- **📊 Structured result presentation** with clear organization
- **🤖 Agent status indicators** showing system health
- **📜 Conversation history** tracking all interactions
- **💡 Helpful tips and examples** for user guidance

## Architecture

```
SCYTA Demo
├── Decision Router (determines which agent to use)
├── FileAgent (file system operations)
├── RAGAgent (document retrieval & QA)
├── InternetAgent (web search)
└── Chat History (conversation tracking)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root directory
2. **API Errors**: Check that your GROQ_API key is set correctly
3. **Permission Errors**: FileAgent only operates in allowed directories (~/Documents, ~/Downloads, etc.)
4. **Module Not Found**: Run `pip install -r requirements.txt`

### Debug Mode

To enable verbose output, set environment variable:
```bash
export SCYTA_DEBUG=1
python run_demo.py
```

## File Structure

```
├── backend/
│   ├── demo.py              # Main demo interface
│   ├── agents/
│   │   ├── fileagent.py     # File operations agent
│   │   ├── ragagent.py      # Document retrieval agent
│   │   └── internetagent.py # Web search agent
│   ├── router.py            # Decision routing logic
│   └── chathistory.py       # Conversation tracking
├── run_demo.py              # Demo launcher script
└── requirements.txt         # Python dependencies
```

## Security Notes

- FileAgent operations are restricted to specific directories for security
- All file operations require user confirmation for destructive actions
- Web searches are performed through safe, rate-limited APIs
- No sensitive data is logged or transmitted

## Contributing

This is a demonstration system. For production use, consider:

- Enhanced error handling and logging
- More sophisticated routing algorithms  
- Additional agent capabilities
- Security hardening
- Performance optimization

## License

This demo is part of the SCYTA project. Please refer to the main project license.
