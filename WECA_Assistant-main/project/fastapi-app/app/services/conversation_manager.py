"""
Conversation Manager using Redis for session-based memory.

Handles storing and retrieving conversation history with TTL-based expiration.
"""

import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime

import redis.asyncio as redis

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class PendingAction:
    """Tracks a multi-turn tool action being collected."""
    tool_name: str
    collected_params: Dict[str, Any]
    missing_params: List[str]
    started_at: str = ""
    
    def __post_init__(self):
        if not self.started_at:
            self.started_at = datetime.now().isoformat()


@dataclass
class ConversationSession:
    """Full conversation session data."""
    session_id: str
    messages: List[Message]
    pending_action: Optional[PendingAction] = None
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


class ConversationManager:
    """
    Manages conversation sessions using Redis.
    
    Features:
    - Session-based conversation history
    - TTL-based auto-expiration (default 10 minutes)
    - Pending action tracking for multi-turn tool calls
    - Async Redis operations
    """
    
    def __init__(
        self,
        host: str = settings.redis_host,
        port: int = settings.redis_port,
        ttl: int = settings.session_ttl,
        max_history: int = settings.max_conversation_history,
    ):
        self.host = host
        self.port = port
        self.ttl = ttl
        self.max_history = max_history
        self._redis: Optional[redis.Redis] = None
    
    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection."""
        if self._redis is None:
            self._redis = redis.Redis(
                host=self.host,
                port=self.port,
                decode_responses=True,
            )
        return self._redis
    
    def _session_key(self, session_id: str) -> str:
        """Generate Redis key for a session."""
        return f"conversation:{session_id}"
    
    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        Retrieve a conversation session from Redis.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            ConversationSession if found, None otherwise
        """
        try:
            r = await self._get_redis()
            data = await r.get(self._session_key(session_id))
            
            if not data:
                return None
            
            session_data = json.loads(data)
            
            # Reconstruct Message objects
            messages = [
                Message(**msg) for msg in session_data.get("messages", [])
            ]
            
            # Reconstruct PendingAction if exists
            pending_action = None
            if session_data.get("pending_action"):
                pending_action = PendingAction(**session_data["pending_action"])
            
            return ConversationSession(
                session_id=session_data["session_id"],
                messages=messages,
                pending_action=pending_action,
                created_at=session_data.get("created_at", ""),
                updated_at=session_data.get("updated_at", ""),
            )
            
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    async def save_session(self, session: ConversationSession) -> bool:
        """
        Save a conversation session to Redis with TTL.
        
        Args:
            session: ConversationSession to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            r = await self._get_redis()
            
            # Update timestamp
            session.updated_at = datetime.now().isoformat()
            
            # Trim messages to max history
            if len(session.messages) > self.max_history:
                session.messages = session.messages[-self.max_history:]
            
            # Serialize to JSON
            session_data = {
                "session_id": session.session_id,
                "messages": [asdict(msg) for msg in session.messages],
                "pending_action": asdict(session.pending_action) if session.pending_action else None,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
            }
            
            # Save with TTL
            await r.setex(
                self._session_key(session.session_id),
                self.ttl,
                json.dumps(session_data),
            )
            
            logger.debug(f"Saved session {session.session_id} with {len(session.messages)} messages")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
            return False
    
    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> ConversationSession:
        """
        Add a message to a session (creates session if doesn't exist).
        
        Args:
            session_id: Session identifier
            role: "user" or "assistant"
            content: Message content
            
        Returns:
            Updated ConversationSession
        """
        session = await self.get_session(session_id)
        
        if session is None:
            session = ConversationSession(
                session_id=session_id,
                messages=[],
            )
        
        session.messages.append(Message(role=role, content=content))
        await self.save_session(session)
        
        return session
    
    async def set_pending_action(
        self,
        session_id: str,
        tool_name: str,
        collected_params: Dict[str, Any],
        missing_params: List[str],
    ) -> bool:
        """
        Set a pending multi-turn action for a session.
        
        Args:
            session_id: Session identifier
            tool_name: Name of the tool being collected for
            collected_params: Parameters already collected
            missing_params: Parameters still needed
            
        Returns:
            True if successful
        """
        session = await self.get_session(session_id)
        
        if session is None:
            session = ConversationSession(
                session_id=session_id,
                messages=[],
            )
        
        session.pending_action = PendingAction(
            tool_name=tool_name,
            collected_params=collected_params,
            missing_params=missing_params,
        )
        
        return await self.save_session(session)
    
    async def clear_pending_action(self, session_id: str) -> bool:
        """Clear any pending action for a session."""
        session = await self.get_session(session_id)
        
        if session is None:
            return True
        
        session.pending_action = None
        return await self.save_session(session)
    
    async def get_conversation_context(
        self,
        session_id: str,
        max_messages: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Get conversation history formatted for LLM.
        
        Args:
            session_id: Session identifier
            max_messages: Optional limit on messages (uses default if not specified)
            
        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        session = await self.get_session(session_id)
        
        if session is None:
            return []
        
        limit = max_messages or self.max_history
        messages = session.messages[-limit:]
        
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session from Redis."""
        try:
            r = await self._get_redis()
            await r.delete(self._session_key(session_id))
            logger.debug(f"Deleted session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check if Redis is reachable."""
        try:
            r = await self._get_redis()
            await r.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None