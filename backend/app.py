import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    logging.warning(
        "python-dotenv not installed. Using system environment variables only."
    )

# Try to import AWS Bedrock
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    HAS_BEDROCK = True
except ImportError:
    HAS_BEDROCK = False
    logging.warning("boto3 not installed. Bedrock functionality will be disabled.")

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

BAKU_TZ = timezone(timedelta(hours=4))

# Configuration from environment variables
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID", "JGMPKF6VEI")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
CLAUDE_MODEL_ID = os.getenv(
    "CLAUDE_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"
)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Application settings
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
SESSION_CLEANUP_HOURS = int(os.getenv("SESSION_CLEANUP_HOURS", "24"))

# Rate limiting settings
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "5"))  # requests per minute
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "2"))  # seconds between retries
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

logger.info(f"Starting application with AWS Region: {AWS_REGION}")
logger.info(f"Knowledge Base ID: {KNOWLEDGE_BASE_ID}")
logger.info(
    f"Rate limiting: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds"
)
logger.info(
    f"AWS credentials configured: {bool(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)}"
)

# Initialize FastAPI app
app = FastAPI(
    title="AI Chatbot API with AWS Bedrock Knowledge Base",
    description="REST API for AI chatbot using AWS Bedrock Knowledge Base and Claude Sonnet with rate limiting",
    version="2.1.0",
    docs_url="/docs",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    session_id: Optional[str] = None
    citations: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    timestamp: Optional[str] = None
    rate_limited: Optional[bool] = False
    retry_after: Optional[int] = None


class ChatSession(BaseModel):
    session_id: str
    created_at: str
    last_activity: str
    message_count: int


# Initialize Bedrock client with explicit credentials
bedrock_client = None
if HAS_BEDROCK and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    try:
        bedrock_client = boto3.client(
            "bedrock-agent-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
        logger.info(
            "AWS Bedrock client initialized successfully with explicit credentials"
        )
    except (NoCredentialsError, Exception) as e:
        logger.warning(
            f"Failed to initialize Bedrock client with explicit credentials: {e}"
        )
        bedrock_client = None
elif HAS_BEDROCK:
    try:
        # Try with default credential chain (IAM roles, environment, etc.)
        bedrock_client = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)
        logger.info(
            "AWS Bedrock client initialized successfully with default credentials"
        )
    except (NoCredentialsError, Exception) as e:
        logger.warning(
            f"Failed to initialize Bedrock client with default credentials: {e}"
        )
        bedrock_client = None
else:
    logger.warning("boto3 not available or AWS credentials not configured")

# In-memory session storage and rate limiting (use Redis/database for production)
chat_sessions = {}
rate_limit_storage = {}  # {session_id: [timestamp1, timestamp2, ...]}


class RateLimiter:
    """Simple in-memory rate limiter"""

    def __init__(
        self,
        max_requests: int = RATE_LIMIT_REQUESTS,
        window_seconds: int = RATE_LIMIT_WINDOW,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def is_allowed(self, key: str) -> tuple[bool, int]:
        """Check if request is allowed, return (allowed, retry_after_seconds)"""
        now = datetime.now()

        # Clean old entries for this key
        if key in rate_limit_storage:
            rate_limit_storage[key] = [
                timestamp
                for timestamp in rate_limit_storage[key]
                if (now - timestamp).total_seconds() < self.window_seconds
            ]
        else:
            rate_limit_storage[key] = []

        # Check if we're within limits
        current_requests = len(rate_limit_storage[key])

        if current_requests >= self.max_requests:
            # Calculate retry after time
            oldest_request = min(rate_limit_storage[key])
            retry_after = self.window_seconds - int(
                (now - oldest_request).total_seconds()
            )
            return False, max(retry_after, 1)

        # Add current request
        rate_limit_storage[key].append(now)
        return True, 0


rate_limiter = RateLimiter()


async def query_knowledge_base_with_retry(
    query: str, session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Query the AWS Bedrock Knowledge Base with retry logic and rate limiting."""
    if not bedrock_client:
        logger.info("Bedrock client not available, using mock response")
        return create_mock_chat_response(query)

    # Validate configuration
    if not KNOWLEDGE_BASE_ID or not AWS_REGION or not CLAUDE_MODEL_ID:
        logger.error(
            "Missing required environment variables: KNOWLEDGE_BASE_ID, AWS_REGION, or CLAUDE_MODEL_ID"
        )
        return {
            "success": False,
            "error": "Missing required configuration parameters",
            "timestamp": datetime.now(BAKU_TZ).isoformat(),
        }

    # Check rate limiting
    rate_key = session_id or "anonymous"
    allowed, retry_after = rate_limiter.is_allowed(rate_key)

    if not allowed:
        logger.warning(
            f"Rate limit exceeded for session {rate_key}, retry after {retry_after} seconds"
        )
        return {
            "success": False,
            "error": f"Rate limit exceeded. Please wait {retry_after} seconds before making another request.",
            "rate_limited": True,
            "retry_after": retry_after,
            "timestamp": datetime.now(BAKU_TZ).isoformat(),
        }

    # Prepare the request
    request_body = {
        "input": {"text": query},
        "retrieveAndGenerateConfiguration": {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                "modelArn": f"arn:aws:bedrock:{AWS_REGION}::foundation-model/{CLAUDE_MODEL_ID}",
            },
        },
    }

    # Only include sessionId if it was previously returned by Bedrock
    if (
        session_id
        and session_id in chat_sessions
        and chat_sessions[session_id].get("bedrock_session", False)
    ):
        request_body["sessionId"] = session_id
        logger.info(f"Using existing Bedrock session ID: {session_id}")
    else:
        logger.info("Starting new Bedrock session (no session ID provided or invalid)")

    # Retry logic for AWS Bedrock requests
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Querying Bedrock (attempt {attempt + 1}/{MAX_RETRIES})")

            # Call Bedrock
            response = bedrock_client.retrieve_and_generate(**request_body)

            logger.info("Successfully received response from Bedrock")

            # Store the Bedrock session ID for future requests
            returned_session_id = response.get("sessionId")
            if returned_session_id:
                chat_sessions[returned_session_id] = {
                    "created_at": datetime.now(BAKU_TZ).isoformat(),
                    "last_activity": datetime.now(BAKU_TZ).isoformat(),
                    "message_count": 1,
                    "bedrock_session": True,  # Mark as a valid Bedrock session
                }

            return {
                "success": True,
                "answer": response["output"]["text"],
                "session_id": returned_session_id,
                "citations": response.get("citations", []),
                "timestamp": datetime.now(BAKU_TZ).isoformat(),
            }

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]
            logger.error(
                f"Bedrock ClientError (attempt {attempt + 1}): {error_code} - {error_message}"
            )

            # Handle throttling specifically
            if error_code == "ThrottlingException" or "rate" in error_message.lower():
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2**attempt)  # Exponential backoff
                    logger.info(
                        f"Rate limited, waiting {wait_time} seconds before retry..."
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    return {
                        "success": False,
                        "error": "AWS Bedrock rate limit exceeded. Please wait a few minutes before trying again.",
                        "rate_limited": True,
                        "retry_after": 120,  # Suggest waiting 2 minutes
                        "timestamp": datetime.now(BAKU_TZ).isoformat(),
                    }

            # Handle other AWS errors
            if error_code == "ValidationException":
                return {
                    "success": False,
                    "error": f"Invalid request parameters: {error_message}",
                    "timestamp": datetime.now(BAKU_TZ).isoformat(),
                }
            elif error_code == "ResourceNotFoundException":
                return {
                    "success": False,
                    "error": f"Resource not found: {error_message}",
                    "timestamp": datetime.now(BAKU_TZ).isoformat(),
                }
            elif error_code == "AccessDeniedException":
                return {
                    "success": False,
                    "error": f"Access denied: {error_message}",
                    "timestamp": datetime.now(BAKU_TZ).isoformat(),
                }
            else:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    logger.info(f"AWS error, retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    return {
                        "success": False,
                        "error": f"AWS error after {MAX_RETRIES} attempts: {error_message}",
                        "timestamp": datetime.now(BAKU_TZ).isoformat(),
                    }

        except Exception as e:
            logger.error(
                f"Unexpected error querying knowledge base (attempt {attempt + 1}): {str(e)}"
            )
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (attempt + 1)
                logger.info(f"Unexpected error, retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                continue
            else:
                return {
                    "success": False,
                    "error": f"Unexpected error after {MAX_RETRIES} attempts: {str(e)}",
                    "timestamp": datetime.now(BAKU_TZ).isoformat(),
                }

    # This shouldn't be reached, but just in case
    return {
        "success": False,
        "error": "Maximum retry attempts exceeded",
        "timestamp": datetime.now(BAKU_TZ).isoformat(),
    }


def create_mock_chat_response(query: str) -> Dict[str, Any]:
    """Create a mock chat response when Bedrock is not available."""
    mock_responses = {
        "hello": "Hello! I'm your AI assistant. I'm currently running in demo mode since AWS Bedrock is not configured.",
        "help": "I can help you with various questions. However, I'm currently in mock mode - please configure AWS Bedrock for full functionality.",
        "status": "I'm running in mock mode. The knowledge base integration requires proper AWS Bedrock configuration.",
        "test": "This is a test response from mock mode. Configure AWS Bedrock to connect to your knowledge base.",
        "rate": "In mock mode, there are no rate limits. This simulates a successful response without calling AWS Bedrock.",
        "default": f"I received your message: '{query}'. I'm currently running in mock mode since the AWS Bedrock knowledge base is not available. Please configure your AWS credentials and ensure the knowledge base is properly set up.",
    }

    # Simple keyword matching for mock responses
    query_lower = query.lower()
    response = mock_responses.get("default")

    for keyword, mock_response in mock_responses.items():
        if keyword != "default" and keyword in query_lower:
            response = mock_response
            break

    return {
        "success": True,
        "answer": response,
        "session_id": str(uuid.uuid4()),
        "citations": [],
        "timestamp": datetime.now(BAKU_TZ).isoformat(),
        "is_mock": True,
    }


def manage_session(session_id: Optional[str]) -> str:
    """Manage chat sessions."""
    current_time = datetime.now(BAKU_TZ).isoformat()

    if session_id and session_id in chat_sessions:
        # Update existing session
        chat_sessions[session_id]["last_activity"] = current_time
        chat_sessions[session_id]["message_count"] += 1
        return session_id
    else:
        # Create new session
        new_session_id = str(uuid.uuid4())
        chat_sessions[new_session_id] = {
            "created_at": current_time,
            "last_activity": current_time,
            "message_count": 1,
        }
        return new_session_id


def cleanup_old_sessions():
    """Clean up sessions older than configured hours."""
    cutoff_time = datetime.now(BAKU_TZ) - timedelta(hours=SESSION_CLEANUP_HOURS)
    sessions_to_remove = []

    for session_id, session_data in chat_sessions.items():
        session_time = datetime.fromisoformat(session_data["last_activity"])
        if session_time < cutoff_time:
            sessions_to_remove.append(session_id)

    for session_id in sessions_to_remove:
        del chat_sessions[session_id]
        # Also clean up rate limiting data
        if session_id in rate_limit_storage:
            del rate_limit_storage[session_id]

    if sessions_to_remove:
        logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")


@app.get("/health")
def health() -> Dict[str, Any]:
    """Health check endpoint."""
    cleanup_old_sessions()

    utc_time = datetime.now(timezone.utc).isoformat()
    baku_time = datetime.now(BAKU_TZ).isoformat()

    return {
        "status": "healthy",
        "service": "AI Chatbot API",
        "utc_time": utc_time,
        "baku_time": baku_time,
        "has_bedrock": HAS_BEDROCK,
        "bedrock_available": bedrock_client is not None,
        "active_sessions": len(chat_sessions),
        "knowledge_base_id": KNOWLEDGE_BASE_ID,
        "aws_region": AWS_REGION,
        "rate_limiting": {
            "max_requests": RATE_LIMIT_REQUESTS,
            "window_seconds": RATE_LIMIT_WINDOW,
            "max_retries": MAX_RETRIES,
            "retry_delay": RETRY_DELAY,
        },
        "environment": {
            "debug": DEBUG,
            "log_level": log_level,
            "session_cleanup_hours": SESSION_CLEANUP_HOURS,
            "aws_credentials_configured": bool(
                AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
            ),
        },
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Main chat endpoint with rate limiting."""
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Manage session
        session_id = manage_session(request.session_id)

        # Query knowledge base with retry logic
        result = await query_knowledge_base_with_retry(request.message, session_id)

        return ChatResponse(
            success=result["success"],
            answer=result.get("answer"),
            session_id=session_id,
            citations=result.get("citations", []),
            error=result.get("error"),
            timestamp=result.get("timestamp"),
            rate_limited=result.get("rate_limited", False),
            retry_after=result.get("retry_after"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        return ChatResponse(
            success=False,
            error=f"Internal server error: {str(e)}",
            timestamp=datetime.now(BAKU_TZ).isoformat(),
        )


@app.get("/sessions")
def get_sessions() -> Dict[str, Any]:
    """Get active chat sessions."""
    cleanup_old_sessions()

    sessions_info = []
    for session_id, session_data in chat_sessions.items():
        sessions_info.append(
            {
                "session_id": session_id,
                "created_at": session_data["created_at"],
                "last_activity": session_data["last_activity"],
                "message_count": session_data["message_count"],
            }
        )

    return {"total_sessions": len(sessions_info), "sessions": sessions_info}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> Dict[str, Any]:
    """Delete a specific chat session."""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        # Also clean up rate limiting data
        if session_id in rate_limit_storage:
            del rate_limit_storage[session_id]
        return {
            "success": True,
            "message": f"Session {session_id} deleted successfully",
        }
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.delete("/sessions")
def clear_all_sessions() -> Dict[str, Any]:
    """Clear all chat sessions."""
    session_count = len(chat_sessions)
    chat_sessions.clear()
    rate_limit_storage.clear()
    return {
        "success": True,
        "message": f"All {session_count} sessions cleared successfully",
    }


@app.get("/rate-limits/{session_id}")
def get_rate_limit_status(session_id: str) -> Dict[str, Any]:
    """Get rate limit status for a session."""
    allowed, retry_after = rate_limiter.is_allowed(session_id)
    current_requests = len(rate_limit_storage.get(session_id, []))

    return {
        "session_id": session_id,
        "allowed": allowed,
        "retry_after": retry_after,
        "current_requests": current_requests,
        "max_requests": RATE_LIMIT_REQUESTS,
        "window_seconds": RATE_LIMIT_WINDOW,
    }


@app.post("/rate-limits/reset/{session_id}")
def reset_rate_limit(session_id: str) -> Dict[str, Any]:
    """Reset rate limit for a specific session (admin function)."""
    if session_id in rate_limit_storage:
        del rate_limit_storage[session_id]
        return {
            "success": True,
            "message": f"Rate limit reset for session {session_id}",
        }
    else:
        return {
            "success": True,
            "message": f"No rate limit data found for session {session_id}",
        }


@app.get("/bedrock/status")
def bedrock_status() -> Dict[str, Any]:
    """Check Bedrock configuration status."""
    status = {
        "boto3_installed": HAS_BEDROCK,
        "client_initialized": bedrock_client is not None,
        "knowledge_base_id": KNOWLEDGE_BASE_ID,
        "aws_region": AWS_REGION,
        "model_id": CLAUDE_MODEL_ID,
        "aws_access_key_configured": bool(AWS_ACCESS_KEY_ID),
        "aws_secret_key_configured": bool(AWS_SECRET_ACCESS_KEY),
        "rate_limiting": {
            "enabled": True,
            "max_requests": RATE_LIMIT_REQUESTS,
            "window_seconds": RATE_LIMIT_WINDOW,
            "max_retries": MAX_RETRIES,
            "retry_delay": RETRY_DELAY,
        },
    }

    if bedrock_client:
        try:
            # Test basic connectivity
            status["connection_test"] = "client_initialized"
        except Exception as e:
            status["connection_test"] = f"failed: {str(e)}"
    else:
        status["connection_test"] = "not_attempted"

    return status


@app.post("/bedrock/test")
async def test_bedrock() -> Dict[str, Any]:
    """Test Bedrock knowledge base with a simple query."""
    test_query = "Hello, can you help me?"

    result = await query_knowledge_base_with_retry(test_query)

    return {
        "test_query": test_query,
        "result": result,
        "bedrock_available": bedrock_client is not None,
        "timestamp": datetime.now(BAKU_TZ).isoformat(),
    }


@app.get("/config")
def get_config() -> Dict[str, Any]:
    """Get current configuration (without sensitive data)."""
    return {
        "knowledge_base_id": KNOWLEDGE_BASE_ID,
        "aws_region": AWS_REGION,
        "claude_model_id": CLAUDE_MODEL_ID,
        "debug": DEBUG,
        "session_cleanup_hours": SESSION_CLEANUP_HOURS,
        "has_bedrock": HAS_BEDROCK,
        "bedrock_client_available": bedrock_client is not None,
        "allowed_origins": ALLOWED_ORIGINS,
        "rate_limiting": {
            "max_requests": RATE_LIMIT_REQUESTS,
            "window_seconds": RATE_LIMIT_WINDOW,
            "max_retries": MAX_RETRIES,
            "retry_delay": RETRY_DELAY,
        },
    }


# Root endpoint
@app.get("/")
def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": "AI Chatbot API with AWS Bedrock Knowledge Base and Rate Limiting",
        "version": "2.1.0",
        "docs": "/docs",
        "health": "/health",
        "status": "/bedrock/status",
        "test": "/bedrock/test",
        "features": [
            "Rate limiting",
            "Retry logic",
            "Session management",
            "Error handling",
        ],
    }
