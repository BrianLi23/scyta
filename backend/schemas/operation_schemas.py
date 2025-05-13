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
            "file_path": {
                "type": "string", 
                "required": True, 
                "description": "Directory (or file) path to scan"
            }
        },
        "process_output": True,
        "function": "_scan_operation"
    },
    
    "delete": {
        "description": "Delete a file or directory",
        "parameters": {
            "file_path": {
                "type": "Union[str, List[str]]", 
                "required": True, 
                "description": "Path of file/folder to delete"
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
                "description": "Current file path"
            },
            "destination_path": {
                "type": "string", 
                "required": True, 
                "description": "New file path"
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
                "description": "Source file path"
            },
            "destination_path": {
                "type": "string", 
                "required": True, 
                "description": "Destination file path"
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
                "description": "Source file path"
            },
            "destination_path": {
                "type": "Union[str, List[str]]", 
                "required": True, 
                "description": "Destination file path"
            }
        },
        "process_output": False,
        "function": "_copy_file"
    },
    
    "create_file": {
        "description": "Create a new file with specified content",
        "parameters": {
            "file_path": {
                "type": "string", 
                "required": True, 
                "description": "Path for the new file"
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
            "file_path": {
                "type": "string", 
                "required": True, 
                "description": "Path for the new directory"
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
    "index": {
        "description": "Index new documents",
        "parameters": {
            "documents": {
                "type": "List", 
                "required": True, 
                "description": "List of documents to index"
            }
        },
        "process_output": True,
        "function": "_index_documents"
    },

    "search": {
        "description": "Search indexed documents",
        "parameters": {
            "query": {
                "type": "string", 
                "required": True, 
                "description": "Search query"
            },
            "k": {
                "type": "integer", 
                "required": False, 
                "description": "Number of documents to return (default: 5)"
            }
        },
        "process_output": True,
        "function": "_search_reasoning"
    },

    "update": {
        "description": "Update existing documents",
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
        "description": "Save the current state of document store",
        "parameters": {},
        "process_output": True,
        "function": "_save_state"
    },

    "load_index": {
        "description": "Load the index from saved state",
        "parameters": {},
        "process_output": True,
        "function": "_load_state"
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
