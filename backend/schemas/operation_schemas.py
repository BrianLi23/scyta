FILEAGENT_SCHEMA = {
    "change_directory": {
        "description": "Change the current working directory",
        "parameters": {
            "directory": {
                "type": "string",
                "required": True,
                "description": "Path of the directory to change to"
            }
        },
        "process_output": False,
        "function": "_change_directory"
    },
    
    "get_file": {
        "description": "Return file path based on provided hash",
        "parameters": {
            "hash": {
                "type": "string", 
                "required": True, 
                "description": "Hash to search for in file names"
            }
        },
        "process_output": True,
        "function": "_get_filepath"
    },
    
    "scan": {
        "description": "Scan a directory/file and return a list of file paths and metadata of all files/folders",
        "parameters": {
            "source_path": {
                "type": "string", 
                "required": True, 
                "description": "Directory (or file) path to scan"
            }
        },
        "process_output": True,
        "function": "_scan_operation"
    },
    
    "get_metadata": {
        "description": "Get detailed metadata of a specific file or directory",
        "parameters": {
            "source_path": {
                "type": "string", 
                "required": True, 
                "description": "Path of the file or directory to get metadata for"
            }
        },
        "process_output": True,
        "function": "_get_metadata_single"
    },
    
    "delete": {
        "description": "Delete a file or directory",
        "parameters": {
            "source_path": {
                "type": "Union[str, List[str]]", 
                "required": True, 
                "description": "Full path to the file/folder to delete (e.g., ~/Documents/file.pdf)"
            }
        },
        "process_output": False,
        "function": "_delete_file"
    },
    
    "rename": {
        "description": "Rename a file or directory from old_path to new_path",
        "parameters": {
            "source_path": {
                "type": "string", 
                "required": True, 
                "description": "Full path to the current file (e.g., ~/Documents/file.pdf)"
            },
            "destination_path": {
                "type": "string", 
                "required": True, 
                "description": "Full path to the new file location (e.g., ~/Documents/new_name.pdf)"
            }
        },
        "process_output": False,
        "function": "_rename_file"
    },
    
    "move": {
        "description": "Move a file or directory to a new location",
        "parameters": {
            "source_path": {
                "type": "string", 
                "required": True, 
                "description": "Full path to the source file (e.g., ~/Documents/file.pdf)"
            },
            "destination_path": {
                "type": "string", 
                "required": True, 
                "description": "Full path to the destination (e.g., ~/Downloads/file.pdf)"
            }
        },
        "process_output": False,
        "function": "_move_file"
    },
    
    "copy": {
        "description": "Copy a file or directory to a new location",
        "parameters": {
            "source_path": {
                "type": "Union[str, List[str]]", 
                "required": True, 
                "description": "Full path to the source file(s) (e.g., ~/Documents/file.pdf)"
            },
            "destination_path": {
                "type": "Union[str, List[str]]", 
                "required": True, 
                "description": "Full path to the destination (e.g., ~/Downloads/file.pdf)"
            }
        },
        "process_output": False,
        "function": "_copy_file"
    },
    
    "create_file": {
        "description": "Create a new file with specified content",
        "parameters": {
            "source_path": {
                "type": "string", 
                "required": True, 
                "description": "Full path for the new file (e.g., ~/Documents/new_file.txt)"
            },
            "content": {
                "type": "string", 
                "required": True, 
                "description": "Content to write to the new file"
            }
        },
        "process_output": False,
        "function": "_create_file"
    },
    
    "create_dir": {
        "description": "Create a new directory",
        "parameters": {
            "source_path": {
                "type": "string", 
                "required": True, 
                "description": "Full path for the new directory (e.g., ~/Documents/new_folder)"
            }
        },
        "process_output": False,
        "function": "_create_dir"
    },
    
    "revert": {
        "description": "Revert the last k operations",
        "parameters": {
            "k": {
                "type": int, 
                "required": True, 
                "description": "Number of operations to revert"
            }
        },
        "process_output": False,
        "function": "_revert_last_k_operations"
    },
    
    "get_chat_history": {
    "description": "Retrieve and return recent chat history",
    "parameters": {
        "limit": {
            "type": "integer",
            "required": False,
            "description": "Number of recent conversations to retrieve (default: 3)"
        }
    },
    "process_output": True,
    "function": "_get_chat_history"
    }
}

RAGAGENT_SCHEMA = {
    "index_documents": {
        "description": "Index new documents into the document store",
        "parameters": {
            "documents": {
                "type": "List[str]", 
                "required": True, 
                "description": "List of document paths to index"
            }
        },
        "process_output": True,
        "function": "_index_documents"
    },

    "search_documents": {
        "description": "Search indexed documents with reasoning capabilities",
        "parameters": {
            "query": {
                "type": "string", 
                "required": True, 
                "description": "Search query to find relevant documents"
            },
            "k": {
                "type": "integer", 
                "required": False, 
                "description": "Number of top results to return (default: 5)"
            }
        },
        "process_output": True,
        "function": "_search_reasoning"
    },

    "update_document": {
        "description": "Update an existing document in the store",
        "parameters": {
            "doc_id": {
                "type": "string", 
                "required": True, 
                "description": "ID of the document to update"
            },
            "path": {
                "type": "string", 
                "required": True, 
                "description": "New path for the document"
            }
        },
        "process_output": True,
        "function": "_update_document"
    },

    "save_state": {
        "description": "Save the current state of the document store to disk",
        "parameters": {},
        "process_output": True,
        "function": "_save_state"
    },

    "load_state": {
        "description": "Load the document store state from disk",
        "parameters": {},
        "process_output": True,
        "function": "_load_state"
    },

    "view_store": {
        "description": "View the current state of the document store",
        "parameters": {},
        "process_output": True,
        "function": "_view_document_store"
    }
}
    
INTERNETAGENT_SCHEMA = {
    "search_web": {
        "description": "Search the web for a given query",
        "parameters": {
            "query": {"type": "string", "required": True, "description": "Search query"}
        },
        "function": "_search_web"
    }
}
