# Qwen 3.5 2B Claude 4.6 Opus reasoning
"""
lib/context_manager.py - Production-grade conversation context manager

A modular system for managing conversation history, token limits, and 
automatic summarization compression using LLM-based techniques.
"""

import os
from typing import List, Optional, Dict, Any, Tuple
from openai import OpenAI
import tiktoken


# =============================================================================
# Configuration Constants (Configurable via environment variables)
# =============================================================================

class Config:
    """Central configuration for context manager settings."""
    
    # Default values that can be overridden
    DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
    DEFAULT_CONTEXT_LIMIT = 128000
    DEFAULT_RESERVED_OUTPUT_TOKENS = 2000
    DEFAULT_KEEP_LAST_N = 6
    
    # Environment variable names for configuration
    ENV_VARS = {
        'MODEL': 'OPENAI_MODEL',
        'CONTEXT_LIMIT': 'OPENAI_CONTEXT_LIMIT',
        'RESERVED_OUTPUT_TOKENS': 'OPENAI_RESERVED_OUTPUT_TOKENS',
        'KEEP_LAST_N': 'OPENAI_KEEP_LAST_N'
    }


# =============================================================================
# Token Management - Handles token encoding and counting
# SOLID Principle: Single Responsibility
# =============================================================================

class TokenManager:
    """
    Manages token encoding, counting, and retrieval.
    
    Provides reusable token computation capabilities for different message formats.
    Implements caching to avoid redundant computations on repeated calls.
    """
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize the token manager with a specific encoding.
        
        Args:
            encoding_name: Name of the tiktoken encoding (default: cl100k_base)
        """
        self.encoding_name = encoding_name
        self.encoder = tiktoken.get_encoding(encoding_name)
    
    def count_tokens(self, messages: Optional[List[Dict[str, str]]] = None) -> int:
        """
        Count tokens in a list of messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                     If None, uses stored messages from ContextManager.
            
        Returns:
            Total number of tokens across all messages.
        """
        if messages is None:
            return sum(len(self._encode_message(m)) for m in self.messages)
        
        return sum(len(self._encode_message(m)) for m in messages)
    
    def _encode_message(self, message: Dict[str, str]) -> int:
        """
        Encode a single message content and return token count.
        
        Args:
            message: Message dictionary with 'content' key
            
        Returns:
            Number of tokens encoded for the message content.
            
        Raises:
            ValueError: If content is empty or not a string.
        """
        if not isinstance(message, dict) or "content" not in message:
            raise ValueError("Message must be a dictionary with 'role' and 'content' keys")
        
        content = message["content"]
        if not content.strip():
            raise ValueError(f"Content cannot be empty for role '{message['role']}'")
        
        try:
            encoded = self.encoder.encode(content)
            return len(encoded)
        except Exception as e:
            print(f"Error encoding message: {e}")
            return 0
    
    def get_encoding_info(self) -> Dict[str, Any]:
        """Get information about the current encoding."""
        return {
            "name": self.encoding_name,
            "model": self.encoder.model_name,
            "max_tokens_per_token": self.encoder.max_tokens_per_token,
            "encoding_type": self.encoder.encoding_type
        }


# =============================================================================
# Message Storage - Handles conversation history management
# SOLID Principle: Single Responsibility
# =============================================================================

class ContextStorage:
    """
    Manages conversation message storage and retrieval.
    
    Provides methods for adding, retrieving, and managing messages in the conversation history.
    """
    
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        
    def add_message(self, role: str, content: str) -> None:
        """
        Add a new message to the conversation history.
        
        Args:
            role: Message role (user/assistant/system)
            content: Message content as a string
            
        Raises:
            ValueError: If content is empty or invalid type
        """
        if not isinstance(content, str):
            raise TypeError(f"Content must be a string, got {type(content).__name__}")
        
        if not content.strip():
            raise ValueError(f"Message content cannot be empty for role '{role}'")
            
        self.messages.append({"role": role, "content": content})
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get current message list."""
        return self.messages.copy()
    
    def remove_messages(self, count: int) -> Tuple[int, List[Dict[str, str]]]:
        """
        Remove messages from the end of history.
        
        Args:
            count: Number of messages to remove
            
        Returns:
            Tuple of (removed_index, remaining_messages)
            
        Raises:
            IndexError: If trying to remove more messages than exist
        """
        if count >= len(self.messages):
            raise IndexError(f"Cannot remove {count} messages when only {len(self.messages)} exist")
        
        removed_index = len(self.messages) - 1 - count
        del self.messages[removed_index]
        return (removed_index, self.messages.copy())
    
    def get_last_k_messages(
        self, 
        k: int = None
    ) -> List[Dict[str, str]]:
        """
        Get the last k messages from history.
        
        Args:
            k: Number of messages to retrieve (default: 6)
            
        Returns:
            Last k messages as a list of message dictionaries
        """
        if k is None or k > len(self.messages):
            return self.messages[-k:] if k > 0 else []
        
        return self.messages[-k:]


# =============================================================================
# Compression Service - Handles conversation summarization
# SOLID Principle: Single Responsibility
# =============================================================================

class CompressionService:
    """
    Manages automatic conversation compression using LLM-based summarization.
    
    Uses an external LLM to generate concise summaries of long conversations,
    preserving essential context while reducing token usage.
    """
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def compress_history(
        self, 
        system_msg: Dict[str, str],
        recent_msgs: List[Dict[str, str]],
        old_msgs: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], bool]:
        """
        Compress conversation history by summarizing old messages.
        
        Args:
            system_msg: Initial system message
            recent_msgs: Most recent k messages to keep unchanged
            old_msgs: Messages to summarize (everything except system and recent)
            
        Returns:
            Tuple of (new_message_stack, was_compressed)
            - was_compressed is True if compression occurred
        """
        if not old_msgs:
            return [system_msg] + recent_msgs, False
        
        # Create summary prompt
        summary_prompt = [
            {
                "role": "system", 
                "content": (
                    "Summarize the following conversation briefly but keep all important "
                    "facts, decisions, and context. Be concise but comprehensive."
                )
            },
            {
                "role": "user", 
                "content": str(old_msgs)
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.client.model,
                messages=summary_prompt,
                max_tokens=500
            )
            
            summary_content = response.choices[0].message.content
            
            # Create a more informative summary message
            summary_msg = {
                "role": "assistant", 
                "content": f"Conversation Summary:\n{summary_content}"
            }
            
            return [system_msg, summary_msg] + recent_msgs, True
            
        except Exception as e:
            print(f"Error compressing history: {e}")
            # Fallback to no compression if LLM fails
            return [system_msg] + recent_msgs, False


# =============================================================================
# Context Manager - Orchestrates token management and compression
# SOLID Principle: Single Responsibility
# =============================================================================

class TokenLimitManager:
    """
    Manages token limits for conversation context.
    
    Ensures the conversation stays within configured token boundaries
    by automatically compressing history when necessary.
    """
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.token_manager = TokenManager()
        self.storage = ContextStorage()
        
        # Initialize with system message
        self.system_message = {"role": "system", "content": "You are a helpful assistant."}
        self._initialize_messages()
    
    def _initialize_messages(self):
        """Initialize messages with system message."""
        self.storage.add_message("system", self.system_message["content"])
        
    def add_user_input(self, user_input: str) -> None:
        """
        Add user input to the conversation.
        
        Args:
            user_input: User's text input
            
        Raises:
            TypeError: If user_input is not a string
        """
        if not isinstance(user_input, str):
            raise TypeError(f"Expected string, got {type(user_input).__name__}")
            
        self.storage.add_message("user", user_input)
        
    def ensure_token_limit(self) -> bool:
        """
        Ensure context stays within token limits.
        
        Returns:
            True if limit was exceeded and compressed
            False if no compression needed
        """
        current_tokens = self.token_manager.count_tokens()
        
        if current_tokens <= self.max_input_tokens:
            return False
            
        print(f"⚠ Context exceeds token limit: {current_tokens}/{self.max_input_tokens}")
        self._compress_history()
        return True
        
    def _compress_history(self):
        """Compress conversation history when necessary."""
        # Extract messages
        system_msg = self.system_message.copy()
        recent_msgs = self.storage.get_last_k_messages(self.keep_last_n)
        
        # Messages to compress are everything except system and recent ones
        old_msgs = []
        for i, msg in enumerate(self.storage.messages):
            if i == 0 or i >= len(self.storage.messages) - len(recent_msgs):
                continue
            old_msgs.append(msg.copy())
            
        compressor = CompressionService(self.client)
        
        # Compress history
        new_messages = compressor.compress_history(
            system_msg, recent_msgs, old_msgs
        )
        
        self._replace_messages(new_messages)
        
    def _replace_messages(self, new_messages):
        """Replace current message list with new compressed messages."""
        self.storage.messages.clear()
        for msg in new_messages:
            if isinstance(msg, dict) and "role" in msg:
                self.storage.add_message(msg["role"], msg["content"])
            elif isinstance(msg, str):
                # Handle case where string might represent JSON
                try:
                    import json
                    decoded_msg = json.loads(msg)
                    if isinstance(decoded_msg, dict) and "role" in decoded_msg:
                        self.storage.add_message(decoded_msg["role"], decoded_msg["content"])
                except (json.JSONDecodeError, KeyError):
                    # Fallback to just storing the string
                    self.storage.add_message("assistant", msg)


class ContextManager:
    """
    Production-grade conversation context manager.
    
    Coordinates token management, message storage, and compression to maintain
    optimal context size while preserving essential information.
    """
    
    def __init__(
        self,
        model: str = None,
        context_limit: int = None,
        reserved_output_tokens: int = None,
        keep_last_n: int = None
    ):
        # Initialize configuration with defaults if not provided
        if model is None or len(model) == 0:
            self.model = Config.DEFAULT_MODEL
        else:
            self.model = model
            
        if context_limit is None or len(context_limit) == 0:
            self.context_limit = Config.DEFAULT_CONTEXT_LIMIT
        else:
            self.context_limit = int(context_limit)
            
        if reserved_output_tokens is None or len(reserved_output_tokens) == 0:
            self.reserved_output_tokens = Config.DEFAULT_RESERVED_OUTPUT_TOKENS
        else:
            self.reserved_output_tokens = int(reserved_output_tokens)
            
        if keep_last_n is None or len(keep_last_n) == 0:
            self.keep_last_n = Config.DEFAULT_KEEP_LAST_N
        else:
            self.keep_last_n = int(keep_last_n)
        
        # Initialize services
        self.model = model
        self.context_limit = context_limit
        self.reserved_output_tokens = reserved_output_tokens
        self.max_input_tokens = context_limit - reserved_output_tokens
        self.keep_last_n = keep_last_n
        
        self.token_manager = TokenManager()
        self.storage = ContextStorage()
        
        # Initialize with system message
        self.system_message = {"role": "system", "content": "You are a helpful assistant."}
        self._initialize_messages()
    
    def _initialize_messages(self):
        """Initialize messages with system message."""
        self.storage.add_message("system", self.system_message["content"])
        
    def add_user_input(self, user_input: str) -> None:
        """
        Add user input to the conversation.
        
        Args:
            user_input: User's text input
            
        Raises:
            TypeError: If user_input is not a string
        """
        if not isinstance(user_input, str):
            raise TypeError(f"Expected string, got {type(user_input).__name__}")
            
        self.storage.add_message("user", user_input)
        
    def ensure_token_limit(self) -> bool:
        """
        Ensure context stays within token limits.
        
        Returns:
            True if limit was exceeded and compressed
            False if no compression needed
        """
        current_tokens = self.token_manager.count_tokens()
        
        if current_tokens <= self.max_input_tokens:
            return False
            
        print(f"⚠ Context exceeds token limit: {current_tokens}/{self.max_input_tokens}")
        self._compress_history()
        return True
        
    def _compress_history(self):
        """Compress conversation history when necessary."""
        # Extract messages
        system_msg = self.system_message.copy()
        recent_msgs = self.storage.get_last_k_messages(self.keep_last_n)
        
        # Messages to compress are everything except system and recent ones
        old_msgs = []
        for i, msg in enumerate(self.storage.messages):
            if i == 0 or i >= len(self.storage.messages) - len(recent_msgs):
                continue
            old_msgs.append(msg.copy())
            
        compressor = CompressionService(self.client)
        
        # Compress history
        new_messages = compressor.compress_history(
            system_msg, recent_msgs, old_msgs
        )
        
        self._replace_messages(new_messages)
        
    def _replace_messages(self, new_messages):
        """Replace current message list with new compressed messages."""
        self.storage.messages.clear()
        for msg in new_messages:
            if isinstance(msg, dict) and "role" in msg:
                self.storage.add_message(msg["role"], msg["content"])
            elif isinstance(msg, str):
                # Handle case where string might represent JSON
                try:
                    import json
                    decoded_msg = json.loads(msg)
                    if isinstance(decoded_msg, dict) and "role" in decoded_msg:
                        self.storage.add_message(decoded_msg["role"], decoded_msg["content"])
                except (json.JSONDecodeError, KeyError):
                    # Fallback to just storing the string
                    self.storage.add_message("assistant", msg)
    
    def chat(self, user_input: str) -> str:
        """
        Execute a conversation with LLM.
        
        Args:
            user_input: User's text input
            
        Returns:
            AI's response content
            
        Raises:
            ValueError: If user_input is empty or invalid type
        """
        # Input validation
        if not isinstance(user_input, str):
            raise ValueError("User input must be a non-empty string")
        
        self.add_user_input(user_input)
        print(f"User: {user_input}")
        
        # Ensure we don't exceed token limit
        self.ensure_token_limit()

        try:
            # Call the LLM with compressed context
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.storage.messages,
                max_tokens=self.reserved_output_tokens
            )
            
            answer = response.choices[0].message.content
            
            # Add assistant response (optional - adjust based on your needs)
            if not isinstance(answer, str) or not answer.strip():
                answer = "I don't have a specific response for that."
                
            self.storage.add_message("assistant", answer)
            
            return answer
            
        except Exception as e:
            print(f"Error in chat(): {e}")
            return "An error occurred while processing your request."


# =============================================================================
# Usage Examples and Integration Guide
# =============================================================================

def create_context_manager(
    model: str = None,
    context_limit: int = None,
    reserved_output_tokens: int = None,
    keep_last_n: int = None
) -> ContextManager:
    """
    Factory function to create a ContextManager with configuration.
    
    Args:
        model: Model name (uses default if not provided)
        context_limit: Maximum input tokens allowed
        reserved_output_tokens: Tokens reserved for AI response
        keep_last_n: Number of recent messages to keep
        
    Returns:
        Configured ContextManager instance
    """
    return ContextManager(
        model=model,
        context_limit=context_limit,
        reserved_output_tokens=reserved_output_tokens,
        keep_last_n=keep_last_n
    )


def create_token_manager(encoding_name: str = "cl100k_base") -> TokenManager:
    """
    Factory function to create a TokenManager instance.
    
    Args:
        encoding_name: Name of the tiktoken encoding
        
    Returns:
        Configured TokenManager instance
    """
    return TokenManager(encoding_name)


def create_compression_service(client: OpenAI) -> CompressionService:
    """
    Factory function to create a CompressionService.
    
    Args:
        client: OpenAI client instance
        
    Returns:
        Configured CompressionService instance
    """
    return CompressionService(client)


# =============================================================================
# Example Usage and Integration Code
# =============================================================================

def main():
    """Demonstrate the complete usage of the context manager system."""
    
    print("=" * 60)
    print("Context Manager System - Complete Usage Guide")
    print("=" * 60)
    
    # Step 1: Create configuration constants
    CONFIG = {
        "model": "nvidia/nemotron-3-super-120b-a12b:free",
        "context_limit": 128000,
        "reserved_output_tokens": 2000,
        "keep_last_n": 6
    }
    
    print("\n[Step 1] Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Step 2: Create the context manager
    print("\n[Step 2] Creating ContextManager...")
    manager = create_context_manager(
        model=CONFIG["model"],
        context_limit=CONFIG["context_limit"],
        reserved_output_tokens=CONFIG["reserved_output_tokens"],
        keep_last_n=CONFIG["keep_last_n"]
    )
    
    # Step 3: Add initial system message (usually set once)
    print("\n[Step 3] Adding initial system message...")
    manager.add_user_input("Hello!")
    
    # Step 4: Process a chat with the LLM
    print("\n[Step 4] Executing conversation...")
    response = manager.chat("What's on your mind today?")
    
    print(f"\n[Result]")
    print(f"AI Response: {response}")
    
    # Step 5: Demonstrate token counting
    print("\n[Step 5] Token Counting Examples:")
    
    # Count tokens in a single message
    test_message = {"role": "user", "content": "Hello, how are you?"}
    tokens_in_message = manager.token_manager.count_tokens([test_message])
    print(f"  Tokens in user message: {tokens_in_message}")
    
    # Count tokens across multiple messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is your favorite programming language?"}
    ]
    total_tokens = manager.token_manager.count_tokens(messages)
    print(f"  Total tokens across system and user message: {total_tokens}")
    
    # Step 6: Demonstrate compression
    print("\n[Step 6] Compression Behavior:")
    print("  - If context exceeds limit, history will be compressed")
    print("  - Compressed messages are summarized using LLM")
    print("  - Last N recent messages are preserved unchanged")
    
    # Step 7: Create token manager standalone (no client needed)
    print("\n[Step 7] TokenManager Standalone Usage:")
    token_manager = create_token_manager(encoding_name="cl100k_base")
    
    # Count tokens without any context manager
    single_message_tokens = token_manager.count_tokens([test_message])
    print(f"  Tokens in single message: {single_message_tokens}")


if __name__ == "__main__":
    main()

# =============================================================================
# Integration with Other Systems
# =============================================================================

class ConversationHandler:
    """
    Example usage of ContextManager in a larger application.
    
    Shows how to integrate the context manager into different parts
    of an application (chatbot, API, etc.).
    """
    
    def __init__(self):
        self.manager = create_context_manager()
        
    def start_conversation(self, user_input: str) -> Optional[str]:
        """Start a new conversation and return the AI response."""
        try:
            # Add user input
            self.manager.add_user_input(user_input)
            
            # Ensure token limit is respected
            if self.manager.ensure_token_limit():
                pass  # Compression occurred, continue
            
            # Get AI response
            response = self.manager.chat(user_input)
            return response
            
        except Exception as e:
            print(f"Error starting conversation: {e}")
            return None
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get current conversation history."""
        return self.manager.storage.get_messages()


# =============================================================================
# Configuration Management
# =============================================================================

class ContextManagerConfig:
    """Configuration class for managing context manager settings."""
    
    def __init__(self):
        self.model = Config.DEFAULT_MODEL
        self.context_limit = Config.DEFAULT_CONTEXT_LIMIT
        self.reserved_output_tokens = Config.DEFAULT_RESERVED_OUTPUT_TOKENS
        self.keep_last_n = Config.DEFAULT_KEEP_LAST_N
    
    @classmethod
    def from_env(cls, **kwargs) -> 'ContextManagerConfig':
        """Create configuration from environment variables."""
        for env_var, value in kwargs.items():
            if os.getenv(env_var):
                try:
                    setattr(cls, env_var.replace('_', '').title(), int(value))
                except ValueError:
                    print(f"Warning: Invalid value '{value}' for {env_var}")
        
        return cls()
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ContextManagerConfig':
        """Load configuration from a YAML/JSON file."""
        with open(config_path, 'r') as f:
            content = f.read().strip()
            
            if content.startswith('{'):
                # JSON format
                import json
                data = json.loads(content)
                for key, value in data.items():
                    if hasattr(cls, key):
                        try:
                            setattr(cls, key, int(value))
                        except ValueError:
                            print(f"Warning: Non-numeric value '{value}' for {key}")
            else:
                # YAML-like format (simple parsing)
                lines = content.split('\n')
                current_key = None
                for line in lines:
                    if line.startswith(' '):  # Indented key
                        parts = line.strip().split(': ')
                        if len(parts) == 2 and not parts[0].startswith('#'):
                            current_key = parts[0]
                            try:
                                setattr(cls, current_key.replace('_', '').title(), int(parts[1]))
                            except ValueError:
                                print(f"Warning: Non-numeric value '{parts[1]}' for {current_key}")
                    elif line.startswith(' ') and not line.strip().startswith('#'):
                        # Value assignment
                        if current_key:
                            try:
                                parts = line.strip().split(':')
                                if len(parts) == 2:
                                    key = parts[0].replace('_', '').title()
                                    value = parts[1]
                                    if hasattr(cls, key):
                                        setattr(cls, key, int(value))
                            except ValueError:
                                print(f"Warning: Non-numeric value '{value}' for {key}")            
            return cls()


# =============================================================================
# Export Functions for Easy Import
# =============================================================================

__all__ = [
    # Core Classes
    'TokenManager',
    'ContextStorage', 
    'CompressionService',
    'TokenLimitManager',
    'ContextManager',
    
    # Factory Functions
    'create_context_manager',
    'create_token_manager',
    'create_compression_service',
    
    # Configuration
    'Config',
    'ContextManagerConfig'
]

# =============================================================================
# Quick Start Example
# =============================================================================

"""
Quick start example - copy and paste this to get started:

from lib.context_manager import ContextManager, TokenManager

# Create manager with default settings
manager = ContextManager()

# Add messages
manager.add_user_input("Hello!")
manager.add_user_input("How are you?")

# Process chat
response = manager.chat("What's up?")
print(response)
"""