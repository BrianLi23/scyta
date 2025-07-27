<div align="center">

# Scyta

**An intelligent multi-agent system for advanced file management and information retrieval**

[![Status](https://img.shields.io/badge/status-In%20Development-orange.svg?style=for-the-badge)](#-roadmap)

</div>

## 🔍 About

Scyta is a multi-agent file-system agent that provides intelligent file processing and information retrieval capabilities. It's designed to be a comprehensive solution for personal file management and information access, using a team of specialized AI agents to handle complex tasks.

***For context: this project was done before coding agents were cool.**

<br/>

<p align="center">
  <img width="100%" alt="Scyta UI" src="https://github.com/user-attachments/assets/8762e550-1016-410d-9c07-2a92be3250d8" />
</p>

### Early-Stage Demo

https://github.com/user-attachments/assets/a76f6cfb-46d0-4463-a8c3-17a7e1b15764

## Features

- **Multi-Agent System**: Utilizes specialized agents that work concurrently to break down and solve complex requests.
- **Intelligent File Operations**: Automates file organization, metadata extraction, and operations for secure and efficient management.
- **Dynamic Learning**: Designed for agents to improve their performance over time based on interaction history and user feedback.

## Architecture

Scyta's power comes from its multi-agent architecture, where complex user prompts are delegated to a team of specialized agents. Each agent has a distinct role and can operate concurrently, collaborating to deliver a comprehensive result.

- **Specialized Agents**: Each agent is an expert in a specific domain (e.g., file operations, information retrieval, web browsing).
- **Concurrent Processing**: Multiple agents can work in parallel, drastically speeding up tasks that involve different domains (e.g., searching the web while organizing local files).
- **Tool-Augmented Reasoning**: Agents are equipped with specific tools to perform their tasks reliably and efficiently.

### Core Agents

| Agent | Description | Key Functions |
| :--- | :--- | :--- |
| **File Agent** | Manages all interactions with the local file system. | Smart organization, permission-aware R/W/X, metadata extraction, secure file handling. |
| **RAG Agent** | Handles information retrieval from indexed documents. | FAISS vector indexing, Cohere-powered response generation, context-aware retrieval. |
| **Internet Agent** | Performs autonomous research and data gathering from the web. | Intelligent scraping, multi-source search, automatic summarization. |

---
## 🛠️ Installation

```bash
# Coming Soon...
