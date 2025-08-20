from typing import Any, Dict

import requests
import streamlit as st

# Configure page
st.set_page_config(
    page_title="RAG AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal CSS for sidebar and core styling
st.markdown(
    """
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 15, 35, 0.95) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.2) !important;
        width: 280px !important;
        display: block !important;
        min-height: 100vh !important;
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    .sidebar-header {
        padding: 10px 16px;
        border-bottom: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    .new-chat-btn {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 8px 16px !important;
        width: 100% !important;
        font-weight: 600 !important;
        margin-bottom: 10px !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    .chat-item {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 8px;
        padding: 8px;
        margin-bottom: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .chat-item:hover {
        background: rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    .chat-item.active {
        background: rgba(99, 102, 241, 0.2);
        border-color: #6366f1;
    }
    
    .chat-title {
        font-size: 14px;
        font-weight: 500;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: 180px;
    }
    
    .delete-btn {
        color: #ef4444 !important;
        background: none;
        border: none;
        cursor: pointer;
        padding: 4px;
        border-radius: 4px;
        transition: all 0.3s ease;
        font-size: 12px;
    }
    
    .delete-btn:hover {
        background: rgba(239, 68, 68, 0.2);
    }
    
    /* Custom header */
    .custom-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 20px;
        border-bottom: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    .logo {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 24px;
        font-weight: 600;
        color: #60a5fa;
    }
    
    .model-badge {
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 14px;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    .status-badge {
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
    }
    
    .status-online {
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .status-offline {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .status-mock {
        background: rgba(251, 191, 36, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
    
    /* Welcome screen */
    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 40vh;
        text-align: center;
    }
    
    .welcome-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 10px;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 20px rgba(96, 165, 250, 0.3)); }
        to { filter: drop-shadow(0 0 30px rgba(167, 139, 250, 0.5)); }
    }
    
    .welcome-subtitle {
        font-size: 1.2rem;
        color: #94a3b8;
        margin-bottom: 20px;
        max-width: 600px;
    }
    
    /* Chat interface */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    
    .message {
        margin: 10px 0;
        padding: 15px;
        border-radius: 16px;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin-left: 60px;
    }
    
    .assistant-message {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.3);
        margin-right: 60px;
    }
    
    .error-message {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.3);
        margin-right: 60px;
    }
    
    .message-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 5px;
        font-weight: 600;
    }
    
    .user-avatar {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 14px;
    }
    
    .assistant-avatar {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #10b981, #059669);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 16px;
    }
    
    .error-avatar {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #ef4444, #dc2626);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 16px;
    }
    
    /* Citation styling */
    .citations {
        margin-top: 10px;
        padding: 10px;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
        border-left: 3px solid #6366f1;
    }
    
    .citation {
        font-size: 12px;
        color: #94a3b8;
        margin: 2px 0;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 2px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-size: 16px !important;
        padding: 12px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.3) !important;
        color: #ffffff !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #94a3b8 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Loading animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(99, 102, 241, 0.3);
        border-radius: 50%;
        border-top-color: #6366f1;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .welcome-title {
            font-size: 2.5rem;
        }
        
        .user-message {
            margin-left: 20px;
        }
        
        .assistant-message {
            margin-right: 20px;
        }
        
        .error-message {
            margin-right: 20px;
        }
    }
    
    .stTextInput input {
        color: #ffffff !important;
        caret-color: #ffffff !important;
    }

    .stTextInput input::placeholder {
        color: #ffffff !important;
        opacity: 0.6 !important;
    }

    .message {
        color: #ffffff !important;
    }

    .user-avatar, .assistant-avatar, .error-avatar {
        color: #ffffff !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "first_interaction" not in st.session_state:
    st.session_state.first_interaction = True
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 0
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "backend_status" not in st.session_state:
    st.session_state.backend_status = None

# Backend API configuration
BACKEND_URL = "http://52.3.105.20:8001"


def check_backend_status() -> Dict[str, Any]:
    """Check if the backend is available and get its status"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            return {
                "available": True,
                "status": "online",
                "bedrock_available": health_data.get("bedrock_available", False),
                "data": health_data,
            }
        else:
            return {
                "available": False,
                "status": "error",
                "error": f"HTTP {response.status_code}",
            }
    except requests.exceptions.ConnectionError:
        return {"available": False, "status": "offline", "error": "Connection refused"}
    except requests.exceptions.Timeout:
        return {"available": False, "status": "timeout", "error": "Request timeout"}
    except Exception as e:
        return {"available": False, "status": "error", "error": str(e)}


def call_rag_api(message: str) -> Dict[str, Any]:
    """
    Call the RAG backend API
    """
    try:
        payload = {"message": message, "session_id": st.session_state.session_id}

        response = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=30)

        if response.status_code == 200:
            data = response.json()
            # Update session ID if provided by backend
            if data.get("session_id"):
                st.session_state.session_id = data["session_id"]
            return data
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
                "answer": None,
            }

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to backend server. Please ensure the FastAPI server is running on http://localhost:8000",
            "answer": None,
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timeout. The backend took too long to respond.",
            "answer": None,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "answer": None,
        }


def create_new_chat():
    """Create a new chat and switch to it"""
    if st.session_state.messages and st.session_state.current_chat_id:
        st.session_state.chats[st.session_state.current_chat_id] = {
            "title": get_chat_title(st.session_state.messages[0]["content"]),
            "messages": st.session_state.messages.copy(),
            "session_id": st.session_state.session_id,
        }
    st.session_state.chat_counter += 1
    new_id = f"chat_{st.session_state.chat_counter}"
    st.session_state.current_chat_id = new_id
    st.session_state.messages = []
    st.session_state.first_interaction = True
    st.session_state.session_id = None  # Reset session for new chat


def load_chat(chat_id: str):
    """Load a specific chat"""
    if chat_id in st.session_state.chats:
        if st.session_state.messages and st.session_state.current_chat_id:
            st.session_state.chats[st.session_state.current_chat_id] = {
                "title": get_chat_title(st.session_state.messages[0]["content"]),
                "messages": st.session_state.messages.copy(),
                "session_id": st.session_state.session_id,
            }
        st.session_state.current_chat_id = chat_id
        chat_data = st.session_state.chats[chat_id]
        st.session_state.messages = chat_data["messages"].copy()
        st.session_state.session_id = chat_data.get("session_id")
        st.session_state.first_interaction = False


def delete_chat(chat_id: str):
    """Delete a specific chat"""
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
        if st.session_state.current_chat_id == chat_id:
            create_new_chat()


def get_chat_title(first_message: str) -> str:
    """Generate a chat title from the first message"""
    title = first_message[:30]
    if len(first_message) > 30:
        title += "..."
    return title


def render_message(
    role: str,
    content: str,
    is_streaming: bool = False,
    is_error: bool = False,
    citations: list = None,
):
    """Render a chat message with proper styling"""
    if role == "user":
        st.markdown(
            f"""
        <div class="message user-message">
            <div class="message-header">
                <div class="user-avatar">U</div>
                <span style="color: #a5b4fc;">You</span>
            </div>
            <div>{content}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    elif is_error:
        st.markdown(
            f"""
        <div class="message error-message">
            <div class="message-header">
                <div class="error-avatar">⚠</div>
                <span style="color: #f87171;">Error</span>
            </div>
            <div>{content}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        avatar_content = "🤖" if not is_streaming else '<div class="loading"></div>'
        citations_html = ""

        if citations and len(citations) > 0:
            citations_html = '<div class="citations"><strong>Sources:</strong><br>'
            for i, citation in enumerate(citations[:3]):  # Limit to 3 citations
                if isinstance(citation, dict):
                    try:
                        # Safe extraction of citation info with multiple fallback options
                        text = "N/A"
                        source = "Unknown source"

                        # Try different possible citation structures
                        retrieved_refs = citation.get("retrievedReferences", [])
                        if retrieved_refs and len(retrieved_refs) > 0:
                            ref = retrieved_refs[0]
                            if isinstance(ref, dict):
                                # Try to get text content
                                content_data = ref.get("content", {})
                                if isinstance(content_data, dict):
                                    text = content_data.get("text", "N/A")
                                elif isinstance(content_data, str):
                                    text = content_data

                                # Try to get source location
                                location = ref.get("location", {})
                                if isinstance(location, dict):
                                    s3_location = location.get("s3Location", {})
                                    if isinstance(s3_location, dict):
                                        source = s3_location.get(
                                            "uri", "Unknown source"
                                        )

                        # Alternative citation structure (if the above doesn't work)
                        if text == "N/A":
                            text = citation.get("content", citation.get("text", "N/A"))
                        if source == "Unknown source":
                            source = citation.get(
                                "source", citation.get("uri", "Unknown source")
                            )

                        # Truncate text safely
                        if isinstance(text, str) and len(text) > 100:
                            text = text[:100] + "..."
                        elif not isinstance(text, str):
                            text = str(text)[:100] + "..."

                        citations_html += f'<div class="citation">{i+1}. {text} (Source: {source})</div>'
                    except Exception:
                        # Fallback for any citation parsing error
                        citations_html += f'<div class="citation">{i+1}. Citation available (parsing error)</div>'
                else:
                    # Handle non-dict citations
                    citations_html += (
                        f'<div class="citation">{i+1}. {str(citation)[:100]}...</div>'
                    )
            citations_html += "</div>"

        st.markdown(
            f"""
        <div class="message assistant-message">
            <div class="message-header">
                <div class="assistant-avatar">{avatar_content}</div>
                <span style="color: #6ee7b7;">RAG AI</span>
            </div>
            <div>{content}</div>
            {citations_html}
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_sidebar():
    """Render the sidebar with chat management and backend status"""
    with st.sidebar:
        st.markdown(
            """
        <div class="sidebar-header">
            <h3 style="margin-bottom: 16px; font-size: 18px;">RAG AI</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Backend status
        if st.session_state.backend_status is None:
            st.session_state.backend_status = check_backend_status()

        status = st.session_state.backend_status
        if status["available"]:
            if status.get("data", {}).get("bedrock_available"):
                status_class = "status-online"
                status_text = "🟢 Backend Online (Bedrock)"
            else:
                status_class = "status-mock"
                status_text = "🟡 Backend Online (Mock Mode)"
        else:
            status_class = "status-offline"
            status_text = "🔴 Backend Offline"

        st.markdown(
            f"""
        <div class="{status_class} status-badge" style="margin-bottom: 10px; text-align: center;">
            {status_text}
        </div>
        """,
            unsafe_allow_html=True,
        )

        if st.button("🔄 Check Status", key="check_status", use_container_width=True):
            st.session_state.backend_status = check_backend_status()
            st.rerun()

        st.markdown("---")

        if st.button("➕ New chat", key="new_chat", use_container_width=True):
            create_new_chat()
            st.rerun()

        if st.session_state.chats:
            st.markdown("### 💬 Chat History")
            for chat_id, chat_data in reversed(list(st.session_state.chats.items())):
                col1, col2 = st.columns([4, 1])
                with col1:
                    is_active = chat_id == st.session_state.current_chat_id
                    button_style = "🔵 " if is_active else "💬 "
                    if st.button(
                        f"{button_style}{chat_data['title']}",
                        key=f"load_{chat_id}",
                        use_container_width=True,
                    ):
                        load_chat(chat_id)
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"delete_{chat_id}", help="Delete chat"):
                        delete_chat(chat_id)
                        st.rerun()
        else:
            st.markdown("### 💬 Chat History")
            st.markdown("*No previous chats*")

        st.markdown("---")
        st.markdown("### 📊 System Info")
        if status["available"] and "data" in status:
            data = status["data"]
            st.markdown(
                f"""
            - **Status**: {'✅ Online' if status['available'] else '❌ Offline'}
            - **Bedrock**: {'✅ Available' if data.get('bedrock_available') else '❌ Mock Mode'}
            - **Sessions**: {data.get('active_sessions', 0)}
            - **Region**: {data.get('aws_region', 'N/A')}
            """
            )
        else:
            st.markdown(
                f"""
            - **Status**: ❌ Offline
            - **Error**: {status.get('error', 'Unknown')}
            """
            )


def main():
    # Render sidebar (always visible)
    render_sidebar()

    # Auto-create first chat if none exists
    if not st.session_state.current_chat_id and not st.session_state.chats:
        create_new_chat()

    # Custom header
    status = st.session_state.backend_status or {"available": False}
    if status["available"] and status.get("data", {}).get("bedrock_available"):
        model_text = "Claude Sonnet 4 + Bedrock ✅"
    elif status["available"]:
        model_text = "Mock Mode (Configure Bedrock) ⚠️"
    else:
        model_text = "Backend Offline ❌"

    st.markdown(
        f"""
    <div class="custom-header">
        <div class="logo">
            🤖 RAG AI Chatbot
        </div>
        <div class="model-badge">
            {model_text}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Show welcome screen if no messages
    if st.session_state.first_interaction and not st.session_state.messages:
        st.markdown(
            """
        <div class="welcome-container">
            <h1 class="welcome-title">Hello!</h1>
            <p class="welcome-subtitle">
                How can I help you today? Ask me anything and I'll search through my knowledge base to provide you with accurate, contextual answers.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            user_input = st.text_input(
                "",
                placeholder="Ask me anything...",
                key="welcome_input",
                label_visibility="collapsed",
            )
            if st.button("Send", key="welcome_send", use_container_width=True):
                if user_input.strip():
                    st.session_state.first_interaction = False
                    st.session_state.messages.append(
                        {"role": "user", "content": user_input}
                    )
                    # Save chat to history immediately
                    if st.session_state.current_chat_id:
                        st.session_state.chats[st.session_state.current_chat_id] = {
                            "title": get_chat_title(user_input),
                            "messages": st.session_state.messages.copy(),
                            "session_id": st.session_state.session_id,
                        }
                    st.rerun()

    else:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        # Display chat history
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                render_message(message["role"], message["content"])
            else:
                citations = message.get("citations", [])
                is_error = message.get("is_error", False)
                render_message(
                    message["role"],
                    message["content"],
                    is_error=is_error,
                    citations=citations,
                )

        # Handle new user message
        if (
            st.session_state.messages
            and st.session_state.messages[-1]["role"] == "user"
        ):
            response_placeholder = st.empty()

            # Show loading state
            with response_placeholder.container():
                render_message("assistant", "Thinking...", is_streaming=True)

            # Call the API
            response = call_rag_api(st.session_state.messages[-1]["content"])

            # Display the response
            if response["success"]:
                full_response = response["answer"] or "No response received"
                citations = response.get("citations", [])
                with response_placeholder.container():
                    render_message("assistant", full_response, citations=citations)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                        "citations": citations,
                        "is_error": False,
                    }
                )
            else:
                error_message = response.get("error", "Unknown error occurred")
                with response_placeholder.container():
                    render_message("assistant", error_message, is_error=True)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message, "is_error": True}
                )

            # Update chat in history after assistant response
            if st.session_state.current_chat_id:
                st.session_state.chats[st.session_state.current_chat_id] = {
                    "title": get_chat_title(st.session_state.messages[0]["content"]),
                    "messages": st.session_state.messages.copy(),
                    "session_id": st.session_state.session_id,
                }

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input(
                "",
                placeholder="Type your message here...",
                key="chat_input",
                label_visibility="collapsed",
            )
        with col2:
            send_button = st.button("Send", key="chat_send", use_container_width=True)

        if send_button and user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input})
            # Save chat to history immediately
            if st.session_state.current_chat_id:
                st.session_state.chats[st.session_state.current_chat_id] = {
                    "title": get_chat_title(st.session_state.messages[0]["content"]),
                    "messages": st.session_state.messages.copy(),
                    "session_id": st.session_state.session_id,
                }
            st.rerun()


if __name__ == "__main__":
    main()
