"""
Prompt Manager for SemanticX Framework.
Handles dynamic prompt loading and template management.
"""
import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages prompt templates and dynamic content injection.
    """
    
    def __init__(self, prompt_dir: str = "./prompts"):
        self.prompt_dir = Path(prompt_dir)
        self.prompts: Dict[str, str] = {}
        self.templates: Dict[str, str] = {}
        self._load_prompts()
        logger.info(f"PromptManager initialized with directory: {prompt_dir}")
    
    def _load_prompts(self):
        """Load all prompt files from the prompt directory."""
        if not self.prompt_dir.exists():
            logger.warning(f"Prompt directory {self.prompt_dir} does not exist")
            return
        
        # Load prompts recursively from subdirectories
        for prompt_file in self.prompt_dir.rglob("*.txt"):
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Use relative path as key
                key = str(prompt_file.relative_to(self.prompt_dir)).replace('\\', '/').replace('.txt', '')
                self.prompts[key] = content
                logger.info(f"Loaded prompt: {key}")
                
            except Exception as e:
                logger.error(f"Failed to load prompt {prompt_file}: {e}")
    
    def get_prompt(self, prompt_path: str) -> Optional[str]:
        """
        Get a prompt by its path.
        
        Args:
            prompt_path: Path to the prompt (e.g., "orchestrator/planner")
            
        Returns:
            Prompt content or None if not found
        """
        return self.prompts.get(prompt_path)

    def load_prompt(self, prompt_path: str) -> str:
        """Alias to match existing orchestrator usage."""
        prompt = self.get_prompt(prompt_path)
        if prompt is None:
            raise FileNotFoundError(f"Prompt not found: {prompt_path}")
        return prompt
    
    def get_prompt_with_context(self, prompt_path: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Get a prompt and inject context variables.
        
        Args:
            prompt_path: Path to the prompt
            context: Dictionary of context variables to inject
            
        Returns:
            Prompt content with context injected
        """
        prompt = self.get_prompt(prompt_path)
        if not prompt:
            return None
        
        return self.inject_context(prompt, context)
    
    def inject_context(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        Inject context variables into a prompt template.
        
        Args:
            prompt: The prompt template
            context: Dictionary of context variables
            
        Returns:
            Prompt with context injected
        """
        try:
            # Find all placeholders in the format {{variable_name}}
            placeholders = re.findall(r'\{\{(\w+)\}\}', prompt)
            
            # Replace each placeholder with its value
            for placeholder in placeholders:
                if placeholder in context:
                    value = str(context[placeholder])
                    prompt = prompt.replace(f'{{{{{placeholder}}}}}', value)
                else:
                    logger.warning(f"Placeholder {placeholder} not found in context")
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error injecting context: {e}")
            return prompt
    
    def list_prompts(self) -> List[str]:
        """List all available prompt paths."""
        return list(self.prompts.keys())

    def get_available_prompts(self) -> List[str]:
        """Alias used by main/info endpoints."""
        return self.list_prompts()
    
    def reload_prompts(self):
        """Reload all prompts from the prompt directory."""
        self.prompts.clear()
        self._load_prompts()
        logger.info("Reloaded prompts from prompt directory")
    
    def add_prompt(self, path: str, content: str):
        """
        Add a new prompt programmatically.
        
        Args:
            path: Path/key for the prompt
            content: Prompt content
        """
        self.prompts[path] = content
        logger.info(f"Added prompt: {path}")
    
    def remove_prompt(self, path: str):
        """
        Remove a prompt.
        
        Args:
            path: Path/key of the prompt to remove
        """
        if path in self.prompts:
            del self.prompts[path]
            logger.info(f"Removed prompt: {path}")
    
    def get_prompt_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded prompts."""
        return {
            "total_prompts": len(self.prompts),
            "prompt_paths": list(self.prompts.keys()),
            "total_characters": sum(len(content) for content in self.prompts.values()),
            "average_length": sum(len(content) for content in self.prompts.values()) / len(self.prompts) if self.prompts else 0
        }
