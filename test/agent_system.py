from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langgraph_checkpoint import create_file_checkpoint
import re
import logging
import os
import json
import sys
import time
from typing import Dict, Any, Optional, List, Union
from dotenv import load_dotenv

# Import assistant manager functions
from assistant_manager import (
    create_assistant_tool, create_generation_assistant_tool, retrieve_assistant_tool,
    modify_system_prompt_tool, delete_assistant_tool, create_thread_tool,
    send_message_to_thread_tool, run_assistant_on_thread_tool, get_thread_messages_tool
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
logger.info("Loading environment variables")
load_dotenv()

# Get API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.critical("OPENAI_API_KEY not found in environment variables")
    sys.exit(1)
else:
    logger.info("API key loaded successfully")

# Initialize the model
logger.info("Initializing ChatOpenAI model")
try:
    model = ChatOpenAI(model="gpt-4o", api_key=api_key)
    logger.info("Model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing model: {str(e)}", exc_info=True)
    raise

# ======= Email Writing Tool =======

def draft_email_tool(recipient: str, subject: str, content: str) -> str:
    """
    Draft an email with the given subject, recipient, and content.
    
    Args:
        recipient: Email recipient
        subject: Email subject
        content: Email content
        
    Returns:
        A formatted email string
    """
    logger.info(f"Drafting email to: {recipient}")
    start_time = time.time()
    
    email_template = f"""
To: {recipient}
Subject: {subject}

{content}

Best regards,
AI Assistant
"""
    
    duration = time.time() - start_time
    logger.info(f"Email drafted in {duration:.4f} seconds")
    
    return email_template

# ======= Utility Functions =======

def classify_user_intent(text: str) -> str:
    """
    Classify the user's intent based on their message
    
    Args:
        text: The user's input message
        
    Returns:
        The classified intent: "assistant_management", "email_creation", 
        "system_improvement", or "general_conversation"
    """
    logger.debug(f"Classifying intent for query: '{text[:50]}...' (truncated)")
    start_time = time.time()
    
    text_lower = text.lower()
    
    # Assistant management keywords
    assistant_mgmt_keywords = [
        "create assistant", "make assistant", "new assistant",
        "delete assistant", "remove assistant", 
        "modify assistant", "update assistant", "change assistant",
        "system prompt", "instructions", "assistant settings"
    ]
    
    # Thread management keywords
    thread_mgmt_keywords = [
        "create thread", "new thread", "start conversation",
        "send message", "message to thread", "run assistant",
        "get messages", "conversation history"
    ]
    
    # Email creation keywords
    email_keywords = [
        "write email", "draft email", "create email", "compose email",
        "email to ", "message to ", "reply to ", "respond to ",
        "formal email", "informal email", "professional email",
        "follow up email", "introduction email", "thank you email",
        "cold email", "sales email", "marketing email",
        "help with email", "email template", "email format"
    ]
    
    # System improvement keywords
    improvement_keywords = [
        "improve", "enhance", "upgrade", "better", "smarter",
        "learn to", "teach you", "adjust your", "change your style",
        "be more", "sound more", "write more", "different tone",
        "feedback", "suggestion", "preference", "like it when you"
    ]
    
    # Check for assistant management intent
    if any(keyword in text_lower for keyword in assistant_mgmt_keywords):
        intent = "assistant_management"
    # Check for thread management intent
    elif any(keyword in text_lower for keyword in thread_mgmt_keywords):
        intent = "thread_management"
    # Check for email creation intent
    elif any(keyword in text_lower for keyword in email_keywords):
        intent = "email_creation"
    # Check for system improvement intent
    elif any(keyword in text_lower for keyword in improvement_keywords):
        intent = "system_improvement"
    # Default intent
    else:
        intent = "general_conversation"
    
    duration = time.time() - start_time
    logger.debug(f"Intent classification completed in {duration:.4f} seconds. Intent: {intent}")
    return intent

def extract_assistant_id(text: str) -> Optional[str]:
    """Extract assistant ID from text if present"""
    logger.debug(f"Extracting assistant ID from text: '{text[:50]}...' (truncated)")
    start_time = time.time()
    
    patterns = [
        r'assistant[_\s]?id[:\s]?\s*["\'`]?([a-zA-Z0-9_-]+)["\'`]?',
        r'asst_[a-zA-Z0-9]+'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            assistant_id = match.group(0) if "asst_" in match.group(0) else match.group(1)
            duration = time.time() - start_time
            logger.debug(f"Assistant ID extracted in {duration:.4f} seconds: {assistant_id}")
            return assistant_id
    
    duration = time.time() - start_time
    logger.debug(f"No assistant ID found in {duration:.4f} seconds")
    return None

def extract_campaign_id(text: str) -> str:
    """Extract or generate campaign ID from text"""
    logger.debug(f"Extracting campaign ID from text: '{text[:50]}...' (truncated)")
    start_time = time.time()
    
    campaign_patterns = [
        r'campaign[_\s]?id[:\s]?\s*["\'`]?([a-zA-Z0-9_-]+)["\'`]?',
        r'campaign[:\s]?\s*["\'`]?([a-zA-Z0-9_-]+)["\'`]?',
        r'for\s+campaign\s+["\'`]?([a-zA-Z0-9_-]+)["\'`]?',
    ]
    
    for pattern in campaign_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            campaign_id = match.group(1)
            duration = time.time() - start_time
            logger.debug(f"Campaign ID extracted in {duration:.4f} seconds: {campaign_id}")
            return campaign_id
    
    # Extract a name if one is mentioned
    name_match = re.search(r'for\s+([a-zA-Z0-9_-]+\s?[a-zA-Z0-9_-]*)', text, re.IGNORECASE)
    if name_match:
        campaign_name = name_match.group(1).strip().replace(" ", "_")
        campaign_id = f"Campaign_{campaign_name}"
        duration = time.time() - start_time
        logger.debug(f"Campaign name extracted and ID generated in {duration:.4f} seconds: {campaign_id}")
        return campaign_id
    
    # Default campaign ID with timestamp
    campaign_id = f"Campaign_{int(time.time())}"
    duration = time.time() - start_time
    logger.debug(f"Default campaign ID generated in {duration:.4f} seconds: {campaign_id}")
    return campaign_id

def extract_thread_id(text: str) -> Optional[str]:
    """Extract thread ID from text if present"""
    logger.debug(f"Extracting thread ID from text: '{text[:50]}...' (truncated)")
    start_time = time.time()
    
    patterns = [
        r'thread[_\s]?id[:\s]?\s*["\'`]?([a-zA-Z0-9_-]+)["\'`]?',
        r'thread_[a-zA-Z0-9]+'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            thread_id = match.group(0) if "thread_" in match.group(0) else match.group(1)
            duration = time.time() - start_time
            logger.debug(f"Thread ID extracted in {duration:.4f} seconds: {thread_id}")
            return thread_id
    
    duration = time.time() - start_time
    logger.debug(f"No thread ID found in {duration:.4f} seconds")
    return None

# ======= Set up Checkpointing =======

# Create a checkpoint for the agent workflows
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

logger.info(f"Setting up checkpointing in {CHECKPOINT_DIR}")
checkpoint = create_file_checkpoint(CHECKPOINT_DIR)

# ======= Create Agents =======

# Create assistant management agent
logger.info("Creating assistant management agent")
try:
    assistant_mgmt_agent = create_react_agent(
        model=model,
        tools=[
            create_assistant_tool, 
            create_generation_assistant_tool,
            retrieve_assistant_tool,
            modify_system_prompt_tool,
            delete_assistant_tool
        ],
        name="assistant_manager",
        prompt=(
            "You are an expert in managing OpenAI assistants. "
            "You can create, retrieve, modify, and delete assistants. "
            "When working with user inputs, analyze their needs and preferences. "
            "\n\n"
            "Use these tools based on user requests:\n"
            "- create_assistant_tool: When a user wants to create a basic email writing assistant\n"
            "- create_generation_assistant_tool: When a user wants a specialized assistant for generating marketing emails and messages\n"
            "- retrieve_assistant_tool: When a user wants information about an existing assistant\n"
            "- modify_system_prompt_tool: When a user wants to update an assistant's instructions\n"
            "- delete_assistant_tool: When a user wants to remove an assistant\n"
        )
    )
    logger.info("Assistant management agent created successfully")
except Exception as e:
    logger.error(f"Error creating assistant management agent: {str(e)}", exc_info=True)
    raise

# Create thread management agent
logger.info("Creating thread management agent")
try:
    thread_mgmt_agent = create_react_agent(
        model=model,
        tools=[
            create_thread_tool,
            send_message_to_thread_tool,
            run_assistant_on_thread_tool,
            get_thread_messages_tool
        ],
        name="thread_manager",
        prompt=(
            "You are an expert in managing OpenAI assistant threads and interactions. "
            "You can create threads, send messages to threads, run assistants on threads, and get messages from threads. "
            "\n\n"
            "Use these tools based on user requests:\n"
            "- create_thread_tool: When a user wants to start a new conversation thread\n"
            "- send_message_to_thread_tool: When a user wants to send a message to a thread\n"
            "- run_assistant_on_thread_tool: When a user wants to get an assistant's response on a thread\n"
            "- get_thread_messages_tool: When a user wants to see the history of messages in a thread\n"
        )
    )
    logger.info("Thread management agent created successfully")
except Exception as e:
    logger.error(f"Error creating thread management agent: {str(e)}", exc_info=True)
    raise

# Create email agent
logger.info("Creating email agent")
try:
    email_agent = create_react_agent(
        model=model,
        tools=[draft_email_tool],
        name="email_expert",
        prompt=(
            "You are an email writing expert. You specialize in drafting professional emails. "
            "When asked to create an email, use the draft_email_tool with appropriate "
            "subject, recipient, and content parameters. Be professional and concise."
        )
    )
    logger.info("Email agent created successfully")
except Exception as e:
    logger.error(f"Error creating email agent: {str(e)}", exc_info=True)
    raise

# Create supervisor workflow with checkpointing
logger.info("Creating supervisor workflow")
try:
    supervisor = create_supervisor(
        [assistant_mgmt_agent, thread_mgmt_agent, email_agent],
        model=model,
        checkpoint=checkpoint,
        prompt=(
            "You are a team supervisor managing three experts:\n"
            "1. The assistant_manager handles creating, updating, and deleting OpenAI assistants\n"
            "2. The thread_manager manages conversation threads and interactions with assistants\n"
            "3. The email_expert specializes in drafting professional emails\n\n"
            "For tasks related to creating or managing OpenAI assistants, use assistant_manager.\n"
            "For tasks related to conversation threads and interactions with assistants, use thread_manager.\n"
            "For tasks related to drafting emails directly, use email_expert.\n"
            "Make sure to determine the user's intent correctly and choose the appropriate expert."
        )
    )
    logger.info("Supervisor created successfully")
except Exception as e:
    logger.error(f"Error creating supervisor: {str(e)}", exc_info=True)
    raise

# Compile the workflow
logger.info("Compiling workflow")
try:
    workflow = supervisor.compile()
    logger.info("Workflow compiled successfully")
except Exception as e:
    logger.error(f"Error compiling workflow: {str(e)}", exc_info=True)
    raise

# ======= Process User Input =======

def process_query(query: str, user_id: str = "default_user", session_id: str = None, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Process a user query through the agent system.
    
    Args:
        query: The user's input message
        user_id: A unique identifier for the user
        session_id: Optional session ID for checkpointing
        history: Optional conversation history
        
    Returns:
        A dictionary containing the processed response and metadata
    """
    logger.info(f"Processing query for user {user_id}: '{query[:50]}...' (truncated)")
    start_time = time.time()
    
    try:
        # Classify the query to determine intent
        intent = classify_user_intent(query)
        logger.info(f"Query classified with intent: {intent}")
        
        # Extract relevant IDs if present
        assistant_id = extract_assistant_id(query)
        thread_id = extract_thread_id(query)
        
        # Set up checkpoint key if session_id is provided
        config_dict = {}
        if session_id:
            config_dict["checkpointer"] = {"config": {"key": f"session_{session_id}"}}
            logger.info(f"Using checkpoint key: session_{session_id}")
        
        # Prepare messages with history if provided
        messages = []
        if history:
            logger.info(f"Using conversation history with {len(history)} messages")
            messages.extend(history[-5:])  # Use last 5 messages for context
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        # Process via supervisor workflow
        logger.info("Invoking supervisor workflow")
        result = workflow.invoke({"messages": messages}, config=config_dict)
        
        # Extract the response content from the result
        response_content = ""
        for message in reversed(result["messages"]):
            if hasattr(message, 'content') and message.content:
                response_content = message.content
                break
        
        if not response_content and len(result["messages"]) > 0:
            response_content = str(result["messages"][-1])
        
        # Format the response as a dictionary
        response = {
            "success": True,
            "intent": intent,
            "response": response_content,
            "assistant_id": assistant_id,
            "thread_id": thread_id,
            "processing_time": round(time.time() - start_time, 4)
        }
        
        logger.info(f"Query processed successfully in {response['processing_time']} seconds")
        return response
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error processing query after {duration:.4f} seconds: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "intent": "unknown",
            "processing_time": round(duration, 4)
        }

# ======= Human-in-the-Loop Functions =======

def process_query_with_human_feedback(query: str, user_id: str = "default_user", session_id: str = None, feedback_callback=None) -> Dict[str, Any]:
    """
    Process a query with optional human feedback for critical operations.
    
    Args:
        query: The user's input message
        user_id: A unique identifier for the user
        session_id: Optional session ID for checkpointing
        feedback_callback: Function to call for human feedback
        
    Returns:
        A dictionary containing the processed response and metadata
    """
    logger.info(f"Processing query with human feedback option for user {user_id}")
    
    # Check if this operation requires human approval
    intent = classify_user_intent(query)
    requires_approval = intent == "assistant_management" and any(x in query.lower() for x in ["delete", "create", "modify"])
    
    # Process the query normally first
    response = process_query(query, user_id, session_id)
    
    # If the operation requires approval and we have a feedback callback
    if requires_approval and feedback_callback and response.get("success"):
        logger.info("Operation requires human approval, requesting feedback")
        
        # Prepare information for human reviewer
        operation_info = {
            "intent": intent,
            "query": query,
            "proposed_response": response.get("response", ""),
            "assistant_id": response.get("assistant_id"),
            "thread_id": response.get("thread_id")
        }
        
        # Get human approval
        approval = feedback_callback(operation_info)
        
        if not approval:
            logger.info("Operation rejected by human reviewer")
            return {
                "success": False,
                "intent": intent,
                "response": "This operation was not approved by the system administrator.",
                "processing_time": response.get("processing_time", 0)
            }
        
        logger.info("Operation approved by human reviewer")
    
    return response

# ======= Main Function =======

if __name__ == "__main__":
    """Test the agent system independently"""
    # Test assistant creation
    print("\n--- Testing Assistant Creation ---")
    campaign_id = f"TestCampaign_{int(time.time())}"
    create_query = f"Create a new assistant for campaign {campaign_id}"
    create_result = process_query(create_query)
    print(f"Query: {create_query}")
    print(f"Result: {json.dumps(create_result, indent=2)}")
    
    # Extract assistant ID from the response
    assistant_id = None
    if create_result.get("success", False):
        # Try to extract assistant ID from the response
        response_text = create_result.get("response", "")
        match = re.search(r'asst_[a-zA-Z0-9]+', response_text)
        if match:
            assistant_id = match.group(0)
            print(f"\nExtracted assistant ID: {assistant_id}")
    
    if assistant_id:
        # Test thread creation
        print("\n--- Testing Thread Creation ---")
        thread_query = "Create a new conversation thread"
        thread_result = process_query(thread_query)
        print(f"Query: {thread_query}")
        print(f"Result: {json.dumps(thread_result, indent=2)}")
        
        # Extract thread ID from the response
        thread_id = None
        if thread_result.get("success", False):
            response_text = thread_result.get("response", "")
            match = re.search(r'thread_[a-zA-Z0-9]+', response_text)
            if match:
                thread_id = match.group(0)
                print(f"\nExtracted thread ID: {thread_id}")
        
        if thread_id:
            # Test sending a message and running the assistant
            print("\n--- Testing Message and Assistant Run ---")
            message_query = f"Send a message to thread {thread_id} asking about email best practices and run assistant {assistant_id} on it"
            message_result = process_query(message_query)
            print(f"Query: {message_query}")
            print(f"Result: {json.dumps(message_result, indent=2)}")
        
        # Cleanup - delete assistant
        print("\n--- Cleaning Up ---")
        delete_query = f"Delete assistant {assistant_id}"
        delete_result = process_query(delete_query)
        print(f"Query: {delete_query}")
        print(f"Result: {json.dumps(delete_result, indent=2)}")
    
    # Test with human-in-the-loop feedback
    print("\n--- Testing Human-in-the-Loop ---")
    
    # Define a simple feedback callback for testing
    def mock_feedback_callback(operation_info):
        print("\nHuman approval requested for:")
        print(f"Intent: {operation_info['intent']}")
        print(f"Query: {operation_info['query']}")
        print(f"Proposed response: {operation_info['proposed_response']}")
        response = input("\nApprove this operation? (y/n): ")
        return response.lower() == 'y'
    
    human_loop_query = "Create a new assistant for email marketing campaign"
    human_loop_result = process_query_with_human_feedback(
        human_loop_query, 
        user_id="test_user", 
        session_id="test_session", 
        feedback_callback=mock_feedback_callback
    )
    print(f"Human-in-the-loop result: {json.dumps(human_loop_result, indent=2)}")
