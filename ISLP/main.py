import gradio as gr
import requests
import uuid
from typing import List, Tuple, Optional, Dict
import json

# Backend API Configuration
BACKEND_URL = "http://localhost:8000"  # Update this to your backend URL

# In-memory cache for chat histories (Gradio 6.5.1 format: list of dicts)
chat_history_cache: Dict[str, List] = {}


class ChatSessionManager:
    """Manages chat sessions and backend API calls"""
    
    def __init__(self, backend_url: str):
        self.backend_url = backend_url
    
    def create_session(self, user_id: str, session_id: str) -> dict:
        """Create a new session"""
        try:
            response = requests.post(
                f"{self.backend_url}/create_session",
                json={"user_id": user_id, "session_id": session_id},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to backend at {self.backend_url}")
            return {"status": "error", "message": "Backend server not running"}
        except Exception as e:
            print(f"Error creating session: {e}")
            return {"status": "error", "message": str(e)}
    
    def delete_session(self, user_id: str, session_id: str) -> dict:
        """Delete a session"""
        try:
            response = requests.post(
                f"{self.backend_url}/delete_session",
                json={"user_id": user_id, "session_id": session_id}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error deleting session: {e}")
            return {"status": "error", "message": str(e)}
    
    def respawn_session(self, user_id: str, session_id: str) -> dict:
        """Respawn a session"""
        try:
            response = requests.post(
                f"{self.backend_url}/respawn_session",
                json={"user_id": user_id, "session_id": session_id}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error respawning session: {e}")
            return {"status": "error", "message": str(e)}
    
    def load_all_sessions(self, user_id: str) -> List[dict]:
        """Load all sessions for a user"""
        try:
            response = requests.post(
                f"{self.backend_url}/load_all_sessions",
                json={"user_id": user_id},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to backend at {self.backend_url}")
            return []
        except Exception as e:
            print(f"Error loading sessions: {e}")
            return []
    
    def chat_with_session(self, user_id: str, session_id: str, 
                         history: List[Tuple[str, str]], message: str) -> str:
        """Send a chat message to the session"""
        try:
            response = requests.post(
                f"{self.backend_url}/chat_with_session",
                json={
                    "user_id": user_id,
                    "session_id": session_id,
                    "history_of_chat": history if history else [],
                    "message": message
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "No response from server")
        except requests.exceptions.ConnectionError:
            error_msg = f"❌ Cannot connect to backend at {self.backend_url}. Make sure the backend server is running!"
            print(error_msg)
            return error_msg
        except requests.exceptions.Timeout:
            error_msg = "❌ Request timed out. The backend server is taking too long to respond."
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(f"Error chatting with session: {e}")
            return error_msg
    
    def get_history_of_session(self, user_id: str, session_id: str) -> List[Tuple[str, str]]:
        """Get chat history for a session"""
        try:
            response = requests.post(
                f"{self.backend_url}/get_history_of_session",
                json={"user_id": user_id, "session_id": session_id}
            )
            response.raise_for_status()
            return response.json().get("history", [])
        except Exception as e:
            print(f"Error getting session history: {e}")
            return []


# Initialize session manager
session_manager = ChatSessionManager(BACKEND_URL)


def get_user_id(request: gr.Request) -> str:
    """Extract user ID from request headers or use default"""
    if request and hasattr(request, 'headers'):
        return request.headers.get('uid', 't12345')
    return 't12345'


def create_new_session(user_id: str, current_session_id: str) -> Tuple[str, List, str, List]:
    """Create a new chat session"""
    new_session_id = str(uuid.uuid4())
    
    # Save current session to cache if it has content
    if current_session_id and current_session_id in chat_history_cache:
        pass  # Already cached
    
    # Create new session on backend
    result = session_manager.create_session(user_id, new_session_id)
    
    # Initialize empty cache for new session
    chat_history_cache[new_session_id] = []
    
    # Refresh session list
    sessions = session_manager.load_all_sessions(user_id)
    session_data = [[s.get('session_id', ''), s.get('created_at', ''), s.get('message_count', 0)] 
                    for s in sessions]
    
    status_msg = f"✅ New session created: {new_session_id[:8]}..."
    return new_session_id, [], status_msg, session_data


def delete_current_session(user_id: str, session_id: str) -> Tuple[str, List, str, List]:
    """Delete the current active session"""
    if not session_id:
        return "", [], "⚠️ No active session to delete", []
    
    # Delete from backend
    session_manager.delete_session(user_id, session_id)
    
    # Remove from cache
    if session_id in chat_history_cache:
        del chat_history_cache[session_id]
    
    # Create a new session automatically
    new_session_id = str(uuid.uuid4())
    session_manager.create_session(user_id, new_session_id)
    chat_history_cache[new_session_id] = []
    
    # Refresh session list
    sessions = session_manager.load_all_sessions(user_id)
    session_data = [[s.get('session_id', ''), s.get('created_at', ''), s.get('message_count', 0)] 
                    for s in sessions]
    
    status_msg = f"🗑️ Session deleted. New session: {new_session_id[:8]}..."
    return new_session_id, [], status_msg, session_data


def delete_all_sessions(user_id: str, current_session_id: str) -> Tuple[str, List, str, List]:
    """Delete all sessions for the user"""
    # Get all sessions
    sessions = session_manager.load_all_sessions(user_id)
    
    # Delete each session
    deleted_count = 0
    for session in sessions:
        sid = session.get('session_id')
        if sid:
            session_manager.delete_session(user_id, sid)
            if sid in chat_history_cache:
                del chat_history_cache[sid]
            deleted_count += 1
    
    # Create a new session
    new_session_id = str(uuid.uuid4())
    session_manager.create_session(user_id, new_session_id)
    chat_history_cache[new_session_id] = []
    
    status_msg = f"🗑️ Deleted {deleted_count} session(s). New session: {new_session_id[:8]}..."
    return new_session_id, [], status_msg, []


def load_session_history(user_id: str, session_id: str, evt: gr.SelectData) -> Tuple[str, List, str]:
    """Load a session's chat history when clicked in the dataset"""
    print(f"SelectData event: index={evt.index}, value={evt.value}")  # Debug
    
    # Get the selected session ID from the dataset
    # evt.value could be the cell value or row data depending on click
    selected_session_id = None
    
    if isinstance(evt.value, list) and len(evt.value) > 0:
        # If it's a row (list), first element is session_id
        selected_session_id = evt.value[0]
    elif isinstance(evt.value, str):
        # If it's a single cell value
        selected_session_id = evt.value
    
    print(f"Extracted session_id: {selected_session_id}")  # Debug
    
    if not selected_session_id:
        return session_id, [], "⚠️ No session selected"
    
    # Check cache first
    if selected_session_id in chat_history_cache:
        history = chat_history_cache[selected_session_id]
        status_msg = f"📂 Loaded session {selected_session_id[:8]}... from cache ({len(history)} messages)"
    else:
        # Fetch from backend
        print(f"Fetching history for session: {selected_session_id}")  # Debug
        backend_history = session_manager.get_history_of_session(user_id, selected_session_id)
        print(f"Backend returned: {len(backend_history)} message pairs")  # Debug
        
        # Convert from backend format (tuples) to Gradio 6.5.1 format (dicts with content blocks)
        history = []
        if backend_history:
            for user_msg, assistant_msg in backend_history:
                if user_msg:
                    history.append({
                        "role": "user",
                        "content": [{"text": user_msg, "type": "text"}]
                    })
                if assistant_msg:
                    history.append({
                        "role": "assistant",
                        "content": [{"text": assistant_msg, "type": "text"}]
                    })
        
        chat_history_cache[selected_session_id] = history
        status_msg = f"📂 Loaded session {selected_session_id[:8]}... from backend ({len(history)} messages)"
    
    return selected_session_id, history, status_msg


def chat_response(message: str, history: List, 
                  user_id: str, session_id: str) -> Tuple[List, str]:
    """Handle chat message and get response"""
    # Ensure history is a list
    if history is None:
        history = []
    
    print(f"=== Chat Response Debug ===")
    print(f"User ID: {user_id}")
    print(f"Session ID: {session_id}")
    print(f"Message: {message}")
    print(f"History length: {len(history)}")
    print(f"History type: {type(history)}")
    if history:
        print(f"First history item: {history[0]}")
    
    if not session_id:
        # Create a new session if none exists
        session_id = str(uuid.uuid4())
        print(f"Creating new session: {session_id}")
        session_manager.create_session(user_id, session_id)
        chat_history_cache[session_id] = []
    
    try:
        # Convert Gradio format (dict) to backend format (list of tuples)
        # Gradio 6.5.1 uses: {'role': 'user', 'content': [{'text': 'message', 'type': 'text'}]}
        backend_history = []
        i = 0
        while i < len(history):
            user_msg = ""
            assistant_msg = ""
            
            # Get user message
            if i < len(history) and isinstance(history[i], dict) and history[i].get('role') == 'user':
                content = history[i].get('content', [])
                # Extract text from content blocks
                if isinstance(content, list) and len(content) > 0:
                    user_msg = content[0].get('text', '') if isinstance(content[0], dict) else str(content[0])
                elif isinstance(content, str):
                    user_msg = content
                i += 1
            
            # Get assistant message
            if i < len(history) and isinstance(history[i], dict) and history[i].get('role') == 'assistant':
                content = history[i].get('content', [])
                # Extract text from content blocks
                if isinstance(content, list) and len(content) > 0:
                    assistant_msg = content[0].get('text', '') if isinstance(content[0], dict) else str(content[0])
                elif isinstance(content, str):
                    assistant_msg = content
                i += 1
            
            # Add the pair if we have at least one message
            if user_msg or assistant_msg:
                backend_history.append((user_msg, assistant_msg))
        
        print(f"Converted to backend format: {len(backend_history)} pairs")
        if backend_history:
            print(f"Last pair: {backend_history[-1]}")
        
        # Get response from backend
        bot_response = session_manager.chat_with_session(user_id, session_id, backend_history, message)
        
        print(f"Bot response received: {bot_response[:100]}...")
        
        # Update history - create new list in Gradio 6.5.1 format
        # Gradio 6.5.1 expects content as a list of content blocks
        new_history = history.copy() if history else []
        new_history.append({
            "role": "user", 
            "content": [{"text": message, "type": "text"}]
        })
        new_history.append({
            "role": "assistant", 
            "content": [{"text": bot_response, "type": "text"}]
        })
        
        print(f"New history length: {len(new_history)}")
        print(f"=== End Chat Response Debug ===")
        
        # Update cache
        chat_history_cache[session_id] = new_history
        
        return new_history, session_id
    except Exception as e:
        print(f"Error in chat_response: {e}")
        import traceback
        traceback.print_exc()
        error_response = f"Error: {str(e)}"
        new_history = history.copy() if history else []
        new_history.append({
            "role": "user",
            "content": [{"text": message, "type": "text"}]
        })
        new_history.append({
            "role": "assistant",
            "content": [{"text": error_response, "type": "text"}]
        })
        return new_history, session_id


def load_initial_sessions(user_id: str) -> List:
    """Load initial session list"""
    sessions = session_manager.load_all_sessions(user_id)
    return [[s.get('session_id', ''), s.get('created_at', ''), s.get('message_count', 0)] 
            for s in sessions]


def refresh_sessions(user_id: str) -> List:
    """Refresh the session list"""
    return load_initial_sessions(user_id)


# Custom CSS for better styling
custom_css = """
#session-controls {
    display: flex;
    gap: 10px;
    margin-bottom: 10px;
}

#session-info {
    background-color: #f0f0f0;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
    font-family: monospace;
}

#status-box {
    padding: 10px;
    border-radius: 5px;
    margin-top: 10px;
}

.session-header {
    font-weight: bold;
    margin-bottom: 5px;
}
"""


# Create Gradio Interface
with gr.Blocks(css=custom_css, title="Chat Session Manager") as demo:
    # State variables
    user_id_state = gr.State(value="t12345")
    session_id_state = gr.State(value=None)
    
    gr.Markdown("# 💬 Chat Session Manager")
    gr.Markdown("Multi-session chat interface with persistent history")
    
    with gr.Row():
        with gr.Column(scale=3):
            # Chat interface
            chatbot = gr.Chatbot(
                label="Chat",
                height=500
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Message",
                    placeholder="Type your message here...",
                    scale=4,
                    show_label=False
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            # Session controls
            with gr.Row():
                new_session_btn = gr.Button("🆕 New Session", variant="secondary")
                delete_session_btn = gr.Button("🗑️ Delete Session", variant="secondary")
                delete_all_btn = gr.Button("🗑️ Delete All Sessions", variant="stop")
                refresh_btn = gr.Button("🔄 Refresh Sessions", variant="secondary")
            
            # Status display
            status_text = gr.Textbox(
                label="Status",
                interactive=False
            )
            
            # Session info
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Current Session:**")
                    current_session_display = gr.Textbox(
                        value="No active session",
                        interactive=False,
                        show_label=False
                    )
        
        with gr.Column(scale=1):
            # Session history list
            gr.Markdown("### 📜 Previous Chats")
            session_list = gr.Dataframe(
                headers=["Session ID", "Created At", "Messages"],
                datatype=["str", "str", "number"],
                label="Click to load session",
                interactive=False,
                wrap=True
            )
    
    # Initialize user_id from request
    def init_user(request: gr.Request):
        uid = get_user_id(request)
        sessions = load_initial_sessions(uid)
        
        # Create initial session if no sessions exist
        if not sessions:
            initial_session_id = str(uuid.uuid4())
            session_manager.create_session(uid, initial_session_id)
            chat_history_cache[initial_session_id] = []
            sessions = load_initial_sessions(uid)
            return uid, initial_session_id, [], f"Session: {initial_session_id[:8]}...", sessions, "✅ Initial session created"
        
        return uid, None, [], "No active session", sessions, "Ready"
    
    # Event handlers
    demo.load(
        init_user,
        inputs=None,
        outputs=[user_id_state, session_id_state, chatbot, current_session_display, session_list, status_text]
    )
    
    # Update session display when session changes
    def update_session_display(session_id):
        if session_id:
            return f"Session: {session_id[:8]}..."
        return "No active session"
    
    session_id_state.change(
        update_session_display,
        inputs=[session_id_state],
        outputs=[current_session_display]
    )
    
    # Chat functionality
    def respond(message, history, user_id, session_id):
        history, new_session_id = chat_response(message, history, user_id, session_id)
        return "", history, new_session_id
    
    msg.submit(
        respond,
        inputs=[msg, chatbot, user_id_state, session_id_state],
        outputs=[msg, chatbot, session_id_state]
    )
    
    send_btn.click(
        respond,
        inputs=[msg, chatbot, user_id_state, session_id_state],
        outputs=[msg, chatbot, session_id_state]
    )
    
    # New session
    new_session_btn.click(
        create_new_session,
        inputs=[user_id_state, session_id_state],
        outputs=[session_id_state, chatbot, status_text, session_list]
    )
    
    # Delete current session
    delete_session_btn.click(
        delete_current_session,
        inputs=[user_id_state, session_id_state],
        outputs=[session_id_state, chatbot, status_text, session_list]
    )
    
    # Delete all sessions
    delete_all_btn.click(
        delete_all_sessions,
        inputs=[user_id_state, session_id_state],
        outputs=[session_id_state, chatbot, status_text, session_list]
    )
    
    # Refresh sessions
    refresh_btn.click(
        refresh_sessions,
        inputs=[user_id_state],
        outputs=[session_list]
    )
    
    # Load session from history
    session_list.select(
        load_session_history,
        inputs=[user_id_state, session_id_state],
        outputs=[session_id_state, chatbot, status_text]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        share=False
    )