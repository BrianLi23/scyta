from llama_cpp import Llama
import os
import concurrent.futures 
from pathlib import Path
from backend.mainframe import BaseAgent, Tool
import hashlib
import mimetypes
import json
from typing import Dict, List, Optional, Union, Any
import shutil
from backend.chathistory import chat_history
from backend.schemas.operation_schemas import FILEAGENT_SCHEMA
from backend.prompts.fileagent_prompt import PROMPT_PLANNING, PROMPT_OPERATION, PLANNING_EXAMPLES, OPERATION_EXAMPLES, POST_PROCESSING_PROMPT
from backend.tools.fileagent_tools import FileAgentTools
import copy
import re
import pwd
from datetime import datetime   

# FileAgent class definition
class FileAgent(BaseAgent):
    def __init__(self, name, description, llm=None, directories: List[str] = None):
        super().__init__(name, description)
        
        # Initialize file tools
        # self.tools = FileAgentTools()
        
        # Initialize metadata cache
        self.metadata_cache = {}
        
        # Initialize current directory tracking
        self.current_directory = Path("~/TestingFolder").expanduser()
        
        # Define allowed directories with their access levels
        self.allowed_directories = {
            "~/Downloads": {"read": True, "write": True, "delete": True},
            "~/Documents": {"read": True, "write": True, "delete": True},
            "~/Pictures": {"read": True, "write": False, "delete": False},
            "~/Music": {"read": True, "write": False, "delete": False},
            "~/Movies": {"read": True, "write": True, "delete": False},
            "~/TestingFolder": {"read": True, "write": True, "delete": True}
        }
        
        # Initialize other attributes
        self.mimetypes = mimetypes
        self.mimetypes.init()  # Used for getting mimetype of files
        self.files = []  # List of file metadata in directories
        self.max_history = 10  # Length of conversation history
        
        # Define operation schemas
        self.fileagent_operations = copy.deepcopy(FILEAGENT_SCHEMA)
        for operation, details in self.fileagent_operations.items():
            function_name = details["function"]
            if isinstance(function_name, str):
                self.fileagent_operations[operation]["function"] = getattr(self, function_name)
                
    def _resolve_file_path(self, file_path: str) -> Path:
        """
        Intelligently resolve a file path by searching in allowed directories if not found
        """
        # First try the path as provided
        path_obj = Path(file_path).expanduser()
        
        # If the file doesn't exist and it's a relative path, try to find it in allowed directories
        if not path_obj.exists() and not path_obj.is_absolute():
            # Try to find the file in allowed directories
            for allowed_dir in self.allowed_directories.keys():
                allowed_path = Path(allowed_dir).expanduser()
                potential_path = allowed_path / path_obj.name
                if potential_path.exists():
                    return potential_path
        
        return path_obj
    
    def _resolve_destination_path(self, destination_path: str, source_path: Path) -> Path:
        """
        Resolve destination path, using source path's parent directory if destination is relative
        """
        destination_obj = Path(destination_path).expanduser()
        
        # If destination is relative and not absolute, resolve it against the source's parent directory
        if not destination_obj.is_absolute():
            destination_obj = source_path.parent / destination_obj
            
        return destination_obj
        
    def _check_permission(self, path: Path, operation: str) -> bool:
        try:
            # Expand and resolve the path
            path = Path(path).expanduser().resolve()
            
            # For new paths that don't exist yet, check the parent directory
            if not path.exists():
                # Check if parent directory exists and has write permission
                parent_path = path.parent
                if not parent_path.exists():
                    return False
                
                # Check if parent directory is within allowed directories
                for allowed_dir, permissions in self.allowed_directories.items():
                    allowed_path = Path(allowed_dir).expanduser().resolve()
                    if allowed_path in parent_path.parents or parent_path == allowed_path:
                        return permissions.get("write", False)
                return False
            
            # For existing paths, check the path itself
            for allowed_dir, permissions in self.allowed_directories.items():
                allowed_path = Path(allowed_dir).expanduser().resolve()
                if allowed_path in path.parents or path == allowed_path:
                    return permissions.get(operation, False)
            
            return False
                
        except Exception:
            return False
        
    def _validate_operation(self, operation: str, path: Union[str, Path]) -> bool:
        path = Path(path).expanduser()
        if operation in ["scan", "get_metadata"]:
            return self._check_permission(path, "read")
        elif operation in ["create_file", "create_dir", "move", "copy", "rename"]:
            return self._check_permission(path, "write")
        elif operation == "delete":
            return self._check_permission(path, "delete")
        return False
        
    def router(self, instruction: str, **kwargs) -> Dict:
        """
        Route file operation; plan, and then handle operation
        """
        
        # Get recent conversation context
        recent_context = chat_history.get_recent_conversations(limit=3)
        
        prompt_planning = PROMPT_PLANNING.format(
            recent_context=recent_context,
            instruction=instruction,
            fileagent_operations=self.fileagent_operations,
            planning_examples=PLANNING_EXAMPLES,
            current_directory=self.get_current_directory()
        )
        
        plan_output = self.llm.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_planning,
            # config={
            # "temperature": 0.2,
            # "max_output_tokens": 4000,
            # }
        )
        plan_response = plan_output.text
        print(plan_response)
        planning_json_response = self._extract_json(plan_response)
        
        prompt_operation = PROMPT_OPERATION.format(
            fileagent_operations=self.fileagent_operations,
            planning_json_response=planning_json_response,
            instruction=instruction,
            operation_examples=OPERATION_EXAMPLES,
            current_directory=self.get_current_directory()
        ) 
       
        operation_output = self.llm.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_operation,
            # config={
            # "temperature": 0.2,
            # "max_output_tokens": 4000,
            # }
        )
        operation_response = operation_output.text
        operation_json_response = self._extract_json(operation_response)
        chat_history.add_operation(operation_json_response)
        chat_history.add_conversation(instruction, operation_json_response, planning_json_response)
        # return operation_json_response.get("operations", []) 
        return operation_json_response
        
    # TODO: In the future, we can use a loop reACT to exeucte each operation in the plan in a loop until done
    def operations(self, json: dict, **kwargs) -> Dict:
        """
        Perform and handle execution of file operations
        """
        try:
            # First parse the operation and its parameters
            operations = json.get("operations", [])
            operation_intermediate_results = [] # Used to store results passed on to next operation
            operation_results = [] # Used to store all results for final processing
            executed_operations = [] # Track what operations were actually executed
            
            for operation in operations:
                
                # print("Performing operation: \n", operation)
                operation_type = operation.get("operation")
                post_processing = operation.get("post_processing", "")
                next_operation = operation.get("next_operation", "")
                kwargs = operation.get("parameters", {})
            
                if operation_type not in self.fileagent_operations:
                    return {"error": "Invalid operation"}
                # print("Operation intermediate results: ", operation_intermediate_results)
                
                # Execute the operation
                # Check if any parameter needs to be populated from previous operation results
                if any(value == "<populated_after_execution>" for value in kwargs.values()):
                    # Get the results from the most recent operation
                    last_result = operation_intermediate_results[-1]
                    
                    # Replace placeholder values with actual data from previous operation
                    # For each parameter in kwargs:
                    # - If value is "<populated_after_execution>", replace with data from last_result
                    # - Otherwise, keep the original value unchanged
                    updated_kwargs = {}
                    for param_name, param_value in kwargs.items():
                        if param_value == "<populated_after_execution>":
                            # Try to get the corresponding value from previous operation result
                            updated_kwargs[param_name] = last_result.get(param_name, param_value)
                        else:
                            # Keep original value if not a placeholder
                            updated_kwargs[param_name] = param_value
                    
                    kwargs = updated_kwargs
                
                print("Executing operation: ", operation_type)
                returnval = self.fileagent_operations[operation_type]["function"](**kwargs)
                # print("Returnval: ", returnval)
                
                # Track the executed operation
                executed_operations.append({
                    "operation": operation_type,
                    "parameters": kwargs,
                    "result": returnval,
                    "timestamp": datetime.now().isoformat()
                })

                # Append the result to the list of results
                operation_results.append(returnval)
                
                if post_processing:
                    # Here we will do batch processing due to the input of token size, and this can help us to account for edge cases
                    batch_size = 6
                    operation_intermediate_results.append(self._batch_process(returnval, next_operation, post_processing, batch_size))
                    
            # Return structured response with operation tracking
            final_result = {}
            if operation_intermediate_results:
                final_result = operation_intermediate_results[-1].get("return_val", {})
            return {
                "response": final_result,
                "operations_executed": executed_operations,
                "results": operation_results,
                "total_operations": len(executed_operations),
                "success": all("error" not in str(result) for result in operation_results)
            }
        
        except Exception as e:
            import traceback
            print(f"Operations method failed at line {traceback.extract_tb(e.__traceback__)[-1].lineno}: {e}")
            return {"error": str(e)}
            
    def _batch_process(self, returnval: List[Dict], next_operation, post_processing, batch_size=10) -> Dict:
        """
        Process batches using ThreadPoolExecutor
        """
        if not returnval:
            return {}
        
        # Make sure to be able to count number of files
            
        metadatas = []
        if isinstance(returnval, list) and len(returnval) > 0 and isinstance(returnval[0], dict):
            # Check if this is a scan result with metadatas
            if 'metadatas' in returnval[0] and isinstance(returnval[0]['metadatas'], list):
                metadatas = returnval[0]['metadatas']
            else:
                # Process the entire result as is
                return self._batch_process_single(returnval, next_operation, 
                                                self.fileagent_operations[next_operation]["parameters"] if next_operation in self.fileagent_operations else "",
                                                post_processing)
        elif isinstance(returnval, dict):
            # Check if this is a direct metadata dictionary
            if 'metadatas' in returnval and isinstance(returnval['metadatas'], list):
                metadatas = returnval['metadatas']
            else:
                # Process the dictionary as a single batch
                return self._batch_process_single(returnval, next_operation, 
                                                self.fileagent_operations[next_operation]["parameters"] if next_operation in self.fileagent_operations else "",
                                                post_processing)
        else:
            # Assume returnval is already a list that can be batched
            metadatas = returnval
        
        batches = [metadatas[i:i + batch_size] for i in range(0, len(metadatas), batch_size)]
        # Get the schema for the next operation
        next_operation_schema = ""
        if next_operation and next_operation in self.fileagent_operations:
            next_operation_schema = self.fileagent_operations[next_operation]["parameters"]
        
        # Process batches in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            batch_results = list(executor.map(
                lambda batch: self._batch_process_single(
                    batch, 
                    next_operation, 
                    next_operation_schema, 
                    post_processing
                ), 
                batches
            ))
        
        # Merge the results from all batches
        merged_results = {}
        for result in batch_results:
            if not result:
                continue
                
            for key, value in result.items():
                if key not in merged_results:
                    merged_results[key] = []
                
                # If value is a list, extend it, otherwise append it
                if isinstance(value, list):
                    merged_results[key].extend(value)
                else:
                    merged_results[key].append(value)
        return merged_results
            
    def _batch_process_single(self, batch_data: List[Dict], next_operation, next_operation_schema, post_processing) -> Dict:
        
        try:
            # print("Next operation: ", next_operation)
            # print("Performing post-processing")
            
            post_processing_prompt = POST_PROCESSING_PROMPT.format(next_operation=next_operation, next_operation_schema=next_operation_schema, post_processing=post_processing, batch_data=batch_data)
            
            process_output = self.llm.models.generate_content(
                model="gemini-2.5-flash",
                contents=post_processing_prompt,
            #     config={
            #     "temperature": 0.2,
            #     "max_output_tokens": 4000,
            # }
        )
            
            process_response = process_output.text
            print("Post-processing response: ", process_response)
            process_json_response = self._extract_json(process_response)
            returnval = process_json_response.get("output", {})
            return returnval
        
        except Exception as e:
            return {"error": str(e)}
        
    def _get_metadata_single(self, source_path: str) -> Dict:
        """
        Get metadata of a file/folder given pathname
        """
        try:
            file_path = Path(source_path).expanduser()
            # If file path is in cache, return it
            if str(file_path) in self.metadata_cache:
                return self.metadata_cache[str(file_path)]

            stat_info = file_path.stat()
            file_hash = self._get_file_hash(file_path) if file_path.is_file() else None
            file_type, encoding = mimetypes.guess_type(str(file_path))

            metadata = {
                "file_path": str(file_path),
                "file_owner": pwd.getpwuid(stat_info.st_uid).pw_name,
                "size": stat_info.st_size,
                "parent_directory": str(file_path.parent),
                # "size_human": self._humanize_size(metadata.st_size),
                "type": "directory" if file_path.is_dir() else "file",
                "timestamps": {
                    "last_modified": datetime.fromtimestamp(stat_info.st_mtime),
                    "last_accessed": datetime.fromtimestamp(stat_info.st_atime),
                    "created": datetime.fromtimestamp(stat_info.st_birthtime)
                },
                "permissions": {
                    "mode": oct(stat_info.st_mode)[-3:],
                    "readable": os.access(file_path, os.R_OK),
                    "writable": os.access(file_path, os.W_OK),
                    "executable": os.access(file_path, os.X_OK)
                }
            }
            
            # Add file-specific metadata 
            if metadata["type"] == "file":
                metadata.update({
                    "file_hash": file_hash,
                    "mime_type": file_type,
                    "extension": file_path.suffix,
                    "size_formatted": self._format_size(stat_info.st_size)
                })
                
            else:
                # Add directory-specific metadata
                try:
                    contents = list(file_path.iterdir())
                    metadata.update({
                        "contents_count": {
                            "files": sum(1 for x in contents if x.is_file()),
                            "directories": sum(1 for x in contents if x.is_dir())
                        }
                    })
                except PermissionError:
                    metadata["error"] = "Permission denied to read directory contents"

            self.metadata_cache[str(file_path)] = metadata
            return metadata

        except Exception as e:
            return {"error": str(e)}
    
    def _format_size(self, size: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}PB"
        
    def _scan_operation(self, source_path: Optional[str] = None) -> List[Dict]:
        """
        Scan specified directory/file and return metadata (Mainly used for directory)
        """
        # Assume LLM doesn't know what the user path is so we'll expand it for them
        try:
            expanded_path = Path(source_path).expanduser()
            
            # Check permissions
            if not self._validate_operation("scan", expanded_path):
                return {"error": f"Permission denied: No read access to {source_path}"}
                
            results = {
                "type": "",
                "metadatas": [],
                "total_files": 0,
                "total_size": 0,
                "formatted_file_list": ""
            }

            if not expanded_path.exists():
                return {"error": f"Path does not exist: {file_path}"}

            if expanded_path.is_file():
                # This means for a singlefile, we process the array of files at index 0
                metadata = self._get_metadata_single(str(expanded_path))
                if "error" not in metadata:
                    results["type"] = "file"
                    results["metadatas"].append(metadata)
                    results["total_files"] = 1
                    results["total_size"] = metadata["size"]
                else:
                    return {"error": f"Failed to get metadata for file: {metadata['error']}"}

            elif expanded_path.is_dir():
                results["type"]= "directory"
                try:
                    for file_path in expanded_path.rglob('*'):
                        
                        if any(parent.name.endswith('.app') for parent in file_path.parents):
                            continue
                        
                        if file_path.name.startswith('.'):
                            continue
                        
                        if file_path.is_file():
                            metadata = self._get_metadata_single(str(file_path))
                            if "error" not in metadata:
                                results["metadatas"].append(metadata)
                                results["total_files"] += 1
                                results["total_size"] += metadata["size"]
                            else:
                                return {"error": f"Failed to get metadata for file: {metadata['error']}"}
                        else:
                            results["total_files"] += 1
                except Exception as e:
                    return {"error": str(e)}
                
            else:
                return {"error": "Invalid path"}
            
            # Add formatted file list to results
            if results["metadatas"]:
                results["formatted_file_list"] = self._format_metadata_list(results["metadatas"])
            
            return results

        except Exception as e:
            return {"error": str(e)}
        
    
    def _track_conversation(self, user_input: str, response: dict, planning: dict = None) -> None:
        """Track conversation with timestamps, user input, and agent responses"""
        conversation_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "planning": planning,  # Store the planning steps
            "response": response,
            "related_operations": [op["operation"] for op in chat_history.operation_history[-len(response.get("operations", [])):]]
        }
        
        self.conversation_history.append(conversation_entry)
        
        # Maintain history length
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
        
    
    def _get_filepath(self, **kwargs) -> List[str]:
        """
        Retrieves file path based on the file (From its hash)
        """
        try:
            file_paths = []
            for directory in self.directories:
                for file_path in Path(directory).rglob('*'):
                    
                    if file_path.is_file():
                        file_paths.append(str(file_path))
            return file_paths
        except Exception as e:
            return {"error": str(e)}
        
    def _confirm_operation(self, operation_type: str, files: List[str]) -> bool:
        """
        Prompt user for confirmation before performing destructive operations
        """
        if not files:
            return True
            
        print(f"\nThe following files will be {operation_type}d:")
        for i, file in enumerate(files, 1):
            print(f"{i}. {file}")
            
        while True:
            if operation_type == "move":
                operation_type = "mov"
            elif operation_type == "delete":
                operation_type = "delet"
            elif operation_type == "rename":
                operation_type = "renam"
                
            response = input(f"\nDo you want to proceed with {operation_type}ing these files? (yes/no): ").lower()
            if response in ['yes', 'y']:
                return True
            elif response in ['no', 'n']:
                return False
            print("Please answer 'yes' or 'no'")

    def _format_file_list(self, source_path: List[str]) -> str:
        """
        Format a list of files for better readability
        """
        if not source_path:
            return "No files found"
            
        formatted = "\nFiles:\n"
        for i, file in enumerate(source_path, 1):
            formatted += f"{i}. {file}\n"
        return formatted
    
    def _format_metadata_list(self, metadatas: List[Dict], max_display: int = 30) -> str:
        """
        Format a list of metadata objects to show file names and key info
        """
        if not metadatas:
            return "No files found"
        
        formatted = f"\n📁 Files Found ({len(metadatas)} total):\n"
        formatted += "─" * 50 + "\n"
        
        # Show first max_display files with details
        for i, metadata in enumerate(metadatas[:max_display], 1):
            file_path = Path(metadata.get('file_path', ''))
            file_name = file_path.name
            file_size = metadata.get('size_formatted', 'Unknown size')
            file_type = metadata.get('type', 'unknown')
            
            formatted += f"{i:2d}. 📄 {file_name}\n"
            formatted += f"     📂 Path: {file_path}\n"
            formatted += f"     📊 Size: {file_size}\n"
            formatted += f"     🏷️  Type: {file_type}\n"
            
            if i < len(metadatas[:max_display]):
                formatted += "\n"
        
        # If there are more files, show a summary
        if len(metadatas) > max_display:
            remaining = len(metadatas) - max_display
            formatted += f"\n... and {remaining} more files\n"
        
        return formatted

    def _delete_file(self, source_path: Union[str, List[str]]) -> Dict:
        try:
            # Check permissions for delete operation
            if isinstance(source_path, list):
                for path in source_path:
                    if not self._validate_operation("delete", path):
                        return {"error": f"Permission denied: No delete access to {path}"}
            else:
                if not self._validate_operation("delete", source_path):
                    return {"error": f"Permission denied: No delete access to {source_path}"}
                    
            trash_path = Path.home() / ".Trash"
            results = []
            
            if isinstance(source_path, list):
                if not self._confirm_operation("delete", source_path):
                    return {"message": "Operation cancelled by user"}
                    
                for path in source_path:
                    path = Path(path).expanduser()
                    if not path.exists():
                        results.append({"file": str(path), "error": "File does not exist"})
                        continue
                    shutil.move(path, trash_path / path.name)
                    results.append({"file": str(path), "message": "File moved to trash successfully"})
                return {"results": results, "formatted_output": self._format_file_list(source_path)}
            else:
                path = Path(source_path).expanduser()
                if not path.exists():
                    return {"error": "File does not exist"}
                if not self._confirm_operation("delete", [str(path)]):
                    return {"message": "Operation cancelled by user"}
                shutil.move(path, trash_path / path.name)
                return {
                    "message": f"File moved to trash successfully: {path.name}",
                    "formatted_output": f"🗑️ Deleted: {path.name}\n     Path: {path}\n     Location: Trash"
                }
        except Exception as e:
            return {"error": str(e)}
        
    def _rename_file(self, source_path: Union[str, List[str]], destination_path: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        try:
            if isinstance(source_path, list):
                if not isinstance(destination_path, list):
                    return {"error": "Source and destination paths must be lists"}
                if len(source_path) != len(destination_path):
                    return {"error": "Source and destination paths must be of equal length"}
                
                if not self._confirm_operation("rename", source_path):
                    return {"message": "Operation cancelled by user"}
                    
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = executor.map(lambda paths: self._rename_file_single(*paths), zip(source_path, destination_path))
                    return {"results": list(results), "formatted_output": self._format_file_list(source_path)}
            
            # For single file operations, also ask for confirmation
            if not self._confirm_operation("rename", [source_path]):
                return {"message": "Operation cancelled by user"}
            return self._rename_file_single(source_path, destination_path)
        except Exception as e:
            return {"error": str(e)}
        
    def _rename_file_single(self, source_path: str, destination_path: str) -> Dict:
        try:
            source_path_obj = self._resolve_file_path(source_path)
            destination_path_obj = self._resolve_destination_path(destination_path, source_path_obj)
            
            if not source_path_obj.exists():
                return {"error": f"File does not exist: {source_path_obj}"}
            
            source_path_obj.rename(destination_path_obj)
            return {
                "message": f"File renamed successfully: {source_path_obj.name} → {destination_path_obj.name}",
                "formatted_output": f"✏️ Renamed: {source_path_obj.name} → {destination_path_obj.name}\n     Path: {destination_path_obj}"
            }
        except Exception as e:
            return {"error": str(e)}
        
    def _extract_json(self, text):
        try:
            # Try parsing JSON from code blocks first
            for pattern in [r"```json\s*\n?(.*?)\n?\s*```", r"```\s*\n?(.*?)\n?\s*```"]:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    return json.loads(match.group(1))

            # Try parsing JSON between brackets
            first_brace_index = text.find('{')
            if first_brace_index == -1:
                raise ValueError("No JSON found")

            json_str = text[first_brace_index:].strip()
            
            # Add missing closing brace if needed
            if json_str.count('{') > json_str.count('}'):
                json_str += '}'
                
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Clean up common JSON formatting issues
                # Remove trailing commas before closing braces/brackets
                json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                # Add missing commas between fields
                json_str = re.sub(r'"\s*}\s*"', '", "', json_str)
                json_str = re.sub(r'"\s*}\s*}', '"}', json_str)
                return json.loads(json_str)
                
        except Exception as e:
            print(f"Error extracting JSON: {e}")
            return {"error": str(e)}
    
    def _move_file(self, source_path: Union[str, List[str]], destination_path: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        try:
            # Check permissions for move operation
            if isinstance(source_path, list):
                for path in source_path:
                    if not self._validate_operation("move", path):
                        return {"error": f"Permission denied: No write access to {path}"}
                    if not self._validate_operation("move", destination_path):
                        return {"error": f"Permission denied: No write access to {destination_path}"}
            else:
                if not self._validate_operation("move", source_path):
                    return {"error": f"Permission denied: No write access to {source_path}"}
                
                if not self._validate_operation("move", destination_path):
                    return {"error": f"Permission denied: No write access to {destination_path}"}
                    
            if isinstance(source_path, list):
                if not isinstance(destination_path, list):
                    dest_path = Path(destination_path).expanduser()
                    if not dest_path.is_dir():
                        return {"error": "Destination path must be a directory"}
                
                    # Create destination paths by joining directory with source filenames
                    dest_paths = [dest_path / Path(src).name for src in source_path]
                    
                    if not self._confirm_operation("move", source_path):
                        return {"message": "Operation cancelled by user"}
                        
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        results = executor.map(lambda paths: self._move_file_single(*paths), zip(source_path, dest_paths))
                        return {"results": list(results), "formatted_output": self._format_file_list(source_path)}
                
                # Case 2: Multiple source files to multiple destination paths
                if len(source_path) != len(destination_path):
                    return {"error": "Source and destination paths must be of equal length"}
                
                if not self._confirm_operation("move", source_path):
                    return {"message": "Operation cancelled by user"}
                    
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = executor.map(lambda paths: self._move_file_single(*paths), zip(source_path, destination_path))
                    return {"results": list(results), "formatted_output": self._format_file_list(source_path)}
            
            # Case 3: Single source to single destination
            if not self._confirm_operation("move", [source_path]):
                return {"message": "Operation cancelled by user"}
            return self._move_file_single(source_path, destination_path)
        except Exception as e:
            return {"error": str(e)}
        
    def _move_file_single(self, source_path: str, destination_path: str) -> Dict:
        try:
            source_path_obj = self._resolve_file_path(source_path)
            destination_path_obj = self._resolve_destination_path(destination_path, source_path_obj)
            
            if not source_path_obj.exists():
                return {"error": f"File does not exist: {source_path_obj}"}
            
            shutil.move(source_path_obj, destination_path_obj)
            return {
                "message": f"File moved successfully: {source_path_obj.name} → {destination_path_obj}",
                "formatted_output": f"📦 Moved: {source_path_obj.name}\n     From: {source_path_obj}\n     To: {destination_path_obj}"
            }
        except Exception as e:
            return {"error": str(e)}
        
    def _copy_file(self, source_path: Union[str, List[str]], destination_path: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        try:
            if isinstance(source_path, list):
                if not isinstance(destination_path, list):
                    dest_path = Path(destination_path).expanduser()
                    if not dest_path.is_dir():
                        return {"error": "Destination must be a directory when copying multiple files"}
                    
                    # Create destination paths by joining directory with source filenames
                    dest_paths = [dest_path / Path(src).name for src in source_path]
                    
                    if not self._confirm_operation("copy", source_path):
                        return {"message": "Operation cancelled by user"}
                        
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        results = executor.map(lambda paths: self._copy_single_file(*paths), zip(source_path, dest_paths))
                        return {"results": list(results), "formatted_output": self._format_file_list(source_path)}
                
                # Case 2: Multiple source files to multiple destination paths
                if len(source_path) != len(destination_path):
                    return {"error": "Source and destination paths must be of equal length"}
                
                if not self._confirm_operation("copy", source_path):
                    return {"message": "Operation cancelled by user"}
                    
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = executor.map(lambda paths: self._copy_single_file(*paths), zip(source_path, destination_path))
                    return {"results": list(results), "formatted_output": self._format_file_list(source_path)}
            return self._copy_single_file(source_path, destination_path)
        except Exception as e:
            return {"error": str(e)}
        
    def _copy_single_file(self, source_path: str, destination_path: str) -> Dict:
        try:
            source_path_obj = self._resolve_file_path(source_path)
            destination_path_obj = self._resolve_destination_path(destination_path, source_path_obj)
            
            if not source_path_obj.exists():
                return {"error": f"File does not exist: {source_path_obj}"}
            
            shutil.copy(source_path_obj, destination_path_obj)
            
            return {
                "message": f"File copied successfully: {source_path_obj.name} → {destination_path_obj}",
                "formatted_output": f"📋 Copied: {source_path_obj.name}\n     From: {source_path_obj}\n     To: {destination_path_obj}"
            }
        except Exception as e:
            return {"error": str(e)}
        
    def _create_file(self, source_path: str, content: Any) -> Dict:
        try:
            if not self._validate_operation("create_file", source_path):
                return {"error": f"Permission denied: No write access to {source_path}"}
            
            file_path = Path(source_path).expanduser()

            # 🔽 Handle formatting of various content types
            if isinstance(content, list):
                flattened = []
                for item in content:
                    if isinstance(item, str):
                        flattened.extend(item.splitlines())
                    else:
                        flattened.append(str(item))
                content_str = "\n".join(flattened)
            elif isinstance(content, (dict, set, tuple)):
                content_str = "\n".join(str(item) for item in content)
            else:
                content_str = str(content)

            with open(file_path, "w") as file:
                file.write(content_str)

            return {
                "message": f"File created successfully: {file_path.name}",
                "formatted_output": f"📄 Created: {file_path.name}\n     Path: {file_path}\n     Content length: {len(content_str)} characters"
            }

        except Exception as e:
            return {"error": str(e)}
        
    def _create_dir(self, source_path: str) -> Dict:
        """
        Create a directory given its path
        """
        try:
            # Check permissions for create operation
            if not self._validate_operation("create_dir", source_path):
                return {"error": f"Permission denied: No write access to {source_path}"}
                
            true_path = Path(source_path).expanduser()
            
            true_path.mkdir(parents=True, exist_ok=True)
            return {
                "message": f"Directory created successfully: {true_path.name}",
                "formatted_output": f"📁 Created: {true_path.name}\n     Path: {true_path}"
            }
        
        except Exception as e:
            return {"error": str(e)}

        
    def _revert_operation(self, operation_id: str) -> Dict:
        """
        Revert operation given timestamp id
        """
        try:
            # Get operation from lookup
            if operation_id not in chat_history.operation_lookup:
                return {"error": "Operation not found"}
                
            print(chat_history.operation_lookup)
            operation = chat_history.operation_lookup[operation_id]
            
            print("Operation to revert: ", operation)
            
            # Strip any quotes from paths before processing
            def clean_path(path_str):
                # Remove single or double quotes from the string
                # path_str = path_str.strip("'\"")
                return Path(path_str).expanduser()
            
            # Ignore for now
            # if not operation.get("reversible", False):
            #     return {"error": "Operation is not reversible"}
                
            # Handle different operation types
            op_type = operation["operation"]
            
            if op_type == "rename":
                old_name = clean_path(Path(operation["parameters"]["old_path"]))
                new_name = clean_path(Path(operation["parameters"]["new_path"]))
                new_name.rename(old_name)
                
            # For now, we can just utilize the mv function, when we reach edges cases we can use osascript
            elif op_type == "delete":
                file_path = clean_path(Path(operation["parameters"]["source_path"]))
                trash_path = Path.home() / ".Trash"
                trash_file = trash_path / file_path.name    
                
                if trash_file.exists():
                    shutil.move(trash_file, file_path)
                    return {"message": f"Restored {file_path} from trash"}
                else:
                    return {"error": "File not found in trash"}
                    
            elif op_type == "create_file":
                # Simply delete the created file
                trash_path = Path.home() / ".Trash"
                file_path = clean_path(Path(operation["parameters"]["source_path"]))
                shutil.move(file_path, trash_path / file_path.name)
                
            elif op_type == "create_dir":
                # Delete the created directory
                trash_path = Path.home() / ".Trash"
                dir_path = clean_path(Path(operation["parameters"]["path"]))
                shutil.move(dir_path, trash_path / dir_path.name)
                
            elif op_type == "move":
                # Move the file back to its original location
                print("Moving back")
                source_path = clean_path(Path(operation["parameters"]["destination_path"]))
                destination_path = clean_path(Path(operation["parameters"]["source_path"]))
                shutil.move(source_path, destination_path)
                
            elif op_type == "copy":
                # Delete the copied file
                file_path = clean_path(Path(operation["parameters"]["destination_path"]))
                file_path.unlink()
                
            # elif op_type == "modify_file":
            #     # Restore previous content
            #     if "previous_content" not in operation["parameters"]:
            #         return {"error": "Previous content not stored, cannot revert"}
            #     with open(operation["parameters"]["file_path"], 'w') as f:
            #         f.write(operation["parameters"]["previous_content"])
            
            else:
                return {"error": f"Unknown operation type: {op_type}"}
            
            # Mark operation as reverted
            operation["status"] = "reverted"
            
            # Update the operation lookup
            chat_history.operation_lookup[operation_id] = operation
            
            return {
                "message": f"Operation {op_type} reverted successfully",
                "operation": operation
            }
            
        except Exception as e:
            return {"error": f"Failed to revert operation: {str(e)}"}

    def _revert_last_k_operations(self, k: int) -> List[Dict]:
        """
        Revert the last k reversible operations
        """
        try:
            # Convert k to int if it's a string
            k = int(k) if isinstance(k, str) else k
            
            results = []
            # Filter out revert operations and get the last k operations
            recent_ops = [op for op in chat_history.operation_history if op["operation"] != "revert"][-k:][::-1]
            
            for op in recent_ops:
                result = self._revert_operation(op["timestamp"])
                results.append(result)
            
            return results
        except ValueError:
            return [{"error": "Invalid value for k: must be a number"}]
        except Exception as e:
            return [{"error": str(e)}]
        
    def _get_file_hash(self, source_path):
        """Return the SHA-256 hash of the file at the given path."""
        sha256_hash = hashlib.sha256()
        # Open file in binary mode
        with open(source_path, "rb") as f:
            # Read and update hash in chunks (to handle large files efficiently)
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        # Return the hex representation of the digest
        return sha256_hash.hexdigest()
    
    def _get_chat_history(self, limit: int = 3) -> str:
        """Get recent conversation context"""
        recent = list(self.conversation_history)[-limit:]
        return "\n".join(
            f"User: {conv['user_input']}\n"
            f"Planning: {json.dumps(conv['planning'], indent=2)}\n"
            f"Response: {json.dumps(conv['response'], indent=2)}\n"
            f"Operations: {', '.join(conv['related_operations'])}"
            for conv in recent
        )
        
    def set_current_directory(self, directory: Union[str, Path]) -> bool:
        """Set the current directory if it's within allowed paths"""
        try:
            directory = Path(directory).expanduser()
            if not directory.exists():
                return False
                
            # Check if directory is within allowed paths
            is_allowed = False
            for allowed_dir in self.allowed_directories.keys():
                allowed_path = Path(allowed_dir).expanduser()
                if allowed_path in directory.parents or allowed_path == directory:
                    is_allowed = True
                    break
            
            if not is_allowed:
                return False
                
            self.current_directory = directory
            return True
        except Exception:
            return False
            
    def get_current_directory(self) -> str:
        """Get the current directory as a string"""
        return str(self.current_directory)

    def _change_directory(self, directory: str) -> Dict:
        """
        Change the current working directory
        """
        try:
            if self.set_current_directory(directory):
                return {"message": f"Changed directory to {directory}"}
            return {"error": f"Failed to change directory to {directory}"}
        except Exception as e:
            return {"error": str(e)}
        
if __name__ == "__main__":
    agent = FileAgent(name="File Agent", description="Handles file operations")
    while True:
        instruction = input("Enter instruction: ")
        print(agent.operations(agent.router(instruction)))


