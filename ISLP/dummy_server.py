"""
Mock Backend Server for Chat Session Management
This is a simple FastAPI backend for testing the Gradio chat interface
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import uvicorn

app = FastAPI(title="Chat Session Backend")

# In-memory storage (replace with actual database in production)
sessions_db: Dict[str, Dict] = {}
chat_histories: Dict[str, List[Dict]] = {}


class CreateSessionRequest(BaseModel):
    user_id: str
    session_id: str


class DeleteSessionRequest(BaseModel):
    user_id: str
    session_id: str


class RespawnSessionRequest(BaseModel):
    user_id: str
    session_id: str


class LoadSessionsRequest(BaseModel):
    user_id: str


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    history_of_chat: List[Tuple[str, str]]
    message: str


class GetHistoryRequest(BaseModel):
    user_id: str
    session_id: str


@app.post("/create_session")
async def create_session(request: CreateSessionRequest):
    """Create a new chat session"""
    session_key = f"{request.user_id}:{request.session_id}"
    
    if session_key in sessions_db:
        return {
            "status": "exists",
            "message": "Session already exists",
            "session_id": request.session_id
        }
    
    sessions_db[session_key] = {
        "user_id": request.user_id,
        "session_id": request.session_id,
        "created_at": datetime.now().isoformat(),
        "message_count": 0,
        "last_updated": datetime.now().isoformat()
    }
    
    chat_histories[session_key] = []
    
    return {
        "status": "success",
        "message": "Session created successfully",
        "session_id": request.session_id
    }


@app.post("/delete_session")
async def delete_session(request: DeleteSessionRequest):
    """Delete a chat session"""
    session_key = f"{request.user_id}:{request.session_id}"
    
    if session_key not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions_db[session_key]
    if session_key in chat_histories:
        del chat_histories[session_key]
    
    return {
        "status": "success",
        "message": "Session deleted successfully"
    }


@app.post("/respawn_session")
async def respawn_session(request: RespawnSessionRequest):
    """Respawn a chat session (clear history but keep session)"""
    session_key = f"{request.user_id}:{request.session_id}"
    
    if session_key not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Clear chat history but keep session metadata
    chat_histories[session_key] = []
    sessions_db[session_key]["message_count"] = 0
    sessions_db[session_key]["last_updated"] = datetime.now().isoformat()
    
    return {
        "status": "success",
        "message": "Session respawned successfully"
    }


@app.post("/load_all_sessions")
async def load_all_sessions(request: LoadSessionsRequest):
    """Load all sessions for a user"""
    user_sessions = []
    
    for session_key, session_data in sessions_db.items():
        if session_data["user_id"] == request.user_id:
            user_sessions.append({
                "session_id": session_data["session_id"],
                "created_at": session_data["created_at"],
                "message_count": session_data["message_count"],
                "last_updated": session_data["last_updated"]
            })
    
    # Sort by last updated (most recent first)
    user_sessions.sort(key=lambda x: x["last_updated"], reverse=True)
    
    return user_sessions


@app.post("/chat_with_session")
async def chat_with_session(request: ChatRequest):
    """Send a message and get a response"""
    print(f"=== Backend Chat Request ===")
    print(f"User ID: {request.user_id}")
    print(f"Session ID: {request.session_id}")
    print(f"Message: {request.message}")
    print(f"History type: {type(request.history_of_chat)}")
    print(f"History length: {len(request.history_of_chat)}")
    if request.history_of_chat:
        print(f"First history item: {request.history_of_chat[0]}")
        print(f"First history item type: {type(request.history_of_chat[0])}")
    
    session_key = f"{request.user_id}:{request.session_id}"
    
    if session_key not in sessions_db:
        # Auto-create session if it doesn't exist
        print(f"Creating new session: {session_key}")
        sessions_db[session_key] = {
            "user_id": request.user_id,
            "session_id": request.session_id,
            "created_at": datetime.now().isoformat(),
            "message_count": 0,
            "last_updated": datetime.now().isoformat()
        }
        chat_histories[session_key] = []
    
    # Store the message in history
    chat_histories[session_key].append({
        "user": request.message,
        "assistant": "",  # Will be filled after response
        "timestamp": datetime.now().isoformat()
    })
    
    # Generate a simple response (replace with actual AI model)
    response = generate_mock_response(request.message, len(request.history_of_chat))
    
    print(f"Generated response: {response[:100]}...")
    
    # Update the last entry with the response
    chat_histories[session_key][-1]["assistant"] = response
    
    # Update session metadata
    sessions_db[session_key]["message_count"] += 1
    sessions_db[session_key]["last_updated"] = datetime.now().isoformat()
    
    print(f"=== End Backend Chat Request ===")
    
    return {
        "status": "success",
        "response": response,
        "session_id": request.session_id
    }


@app.post("/get_history_of_session")
async def get_history_of_session(request: GetHistoryRequest):
    """Get chat history for a session"""
    session_key = f"{request.user_id}:{request.session_id}"
    
    if session_key not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    history = chat_histories.get(session_key, [])
    
    # Convert to tuple format for Gradio
    formatted_history = [
        (msg["user"], msg["assistant"]) 
        for msg in history 
        if msg["user"] and msg["assistant"]
    ]
    
    return {
        "status": "success",
        "history": formatted_history,
        "message_count": len(formatted_history)
    }


def generate_mock_response(message: str, conversation_length: int) -> str:
    """Generate a mock AI response"""
    responses = [
        f"I received your message: '{message}'. This is response #{conversation_length + 1} in our conversation.",
        f"That's interesting! You said: '{message}'. How can I help you further?",
        f"Thank you for sharing that. Regarding '{message}', I'd be happy to discuss this more.",
        f"I understand. About '{message}' - let me provide some insights on that topic.",
        f"Great question! Regarding '{message}', here's what I think...",
    ]
    
    # Simple response selection based on message length
    response_idx = len(message) % len(responses)
    return responses[response_idx]


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "total_sessions": len(sessions_db),
        "endpoints": [
            "/create_session",
            "/delete_session",
            "/respawn_session",
            "/load_all_sessions",
            "/chat_with_session",
            "/get_history_of_session"
        ]
    }


@app.get("/stats")
async def stats():
    """Get backend statistics"""
    return {
        "total_sessions": len(sessions_db),
        "total_messages": sum(len(hist) for hist in chat_histories.values()),
        "active_users": len(set(s["user_id"] for s in sessions_db.values()))
    }


if __name__ == "__main__":
    print("🚀 Starting Chat Session Backend Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )