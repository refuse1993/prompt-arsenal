"""
ConversationMemory - Centralized conversation history management

Inspired by PyRIT's memory system for multi-turn jailbreak attacks
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json


class ConversationMessage:
    """Single message in a conversation"""

    def __init__(
        self,
        role: str,  # "user" or "assistant"
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        self.role = role
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMessage':
        """Create from dictionary"""
        return cls(
            role=data["role"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else None
        )


class ConversationMemory:
    """
    Centralized conversation history manager

    Manages conversation state for multi-turn attacks:
    - Stores messages from both Strategy LLM and Target LLM
    - Provides history in correct format for each LLM provider
    - Supports persistence and retrieval
    """

    def __init__(self, conversation_id: str = None):
        """
        Args:
            conversation_id: Unique identifier for this conversation
        """
        self.conversation_id = conversation_id or self._generate_id()
        self.messages: List[ConversationMessage] = []
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now(),
            "goal": None,
            "strategy": None
        }

    def _generate_id(self) -> str:
        """Generate unique conversation ID"""
        import uuid
        return str(uuid.uuid4())

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationMessage:
        """
        Add message to conversation

        Args:
            role: "user" or "assistant"
            content: Message content
            metadata: Optional metadata (turn number, images, etc.)

        Returns:
            Created message
        """
        message = ConversationMessage(role, content, metadata)
        self.messages.append(message)
        return message

    def add_user_message(self, content: str, **metadata) -> ConversationMessage:
        """Convenience method to add user message"""
        return self.add_message("user", content, metadata)

    def add_assistant_message(self, content: str, **metadata) -> ConversationMessage:
        """Convenience method to add assistant message"""
        return self.add_message("assistant", content, metadata)

    def get_messages(self, limit: Optional[int] = None) -> List[ConversationMessage]:
        """
        Get conversation messages

        Args:
            limit: Maximum number of recent messages (None = all)

        Returns:
            List of messages
        """
        if limit is None:
            return self.messages.copy()
        return self.messages[-limit:] if len(self.messages) > limit else self.messages.copy()

    def get_history_for_llm(
        self,
        provider: str = "openai",
        limit: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Get conversation history in LLM-specific format

        Args:
            provider: "openai", "anthropic", or "google"
            limit: Maximum number of recent messages

        Returns:
            List of messages in provider format
        """
        messages = self.get_messages(limit)

        if provider in ["openai", "anthropic"]:
            # OpenAI and Anthropic use {"role": "user|assistant", "content": "..."}
            return [{"role": msg.role, "content": msg.content} for msg in messages]

        elif provider == "google":
            # Google Gemini uses {"role": "user|model", "parts": ["..."]}
            return [{
                "role": "model" if msg.role == "assistant" else msg.role,
                "parts": [msg.content]
            } for msg in messages]

        else:
            # Default to OpenAI format
            return [{"role": msg.role, "content": msg.content} for msg in messages]

    def clear(self):
        """Clear all messages (keeps metadata)"""
        self.messages = []

    def set_metadata(self, key: str, value: Any):
        """Set metadata field"""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata field"""
        return self.metadata.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "conversation_id": self.conversation_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMemory':
        """Deserialize from dictionary"""
        memory = cls(conversation_id=data["conversation_id"])
        memory.messages = [ConversationMessage.from_dict(msg) for msg in data["messages"]]
        memory.metadata = data["metadata"]
        return memory

    def save_to_file(self, filepath: str):
        """Save conversation to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, default=str)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'ConversationMemory':
        """Load conversation from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __len__(self) -> int:
        """Return number of messages"""
        return len(self.messages)

    def __repr__(self) -> str:
        return f"ConversationMemory(id={self.conversation_id}, messages={len(self.messages)})"
