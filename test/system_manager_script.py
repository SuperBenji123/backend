from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import openai
import os
import json
import logging
import sys
import re
import time
import requests
import asyncio
from typing import Dict, Any, Optional, List, Union

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

# Get API keys from environment variables
api_key = os.getenv("OPENAI_API_KEY")
assistants_api_key = os.getenv("OPENAI_ASSISTANTS_API_KEY", api_key)  # Use main API key as fallback
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000/api/response")

if not api_key:
    logger.critical("OPENAI_API_KEY not found in environment variables")
    sys.exit(1)
else:
    logger.info("API keys loaded successfully")

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=assistants_api_key)

# Initialize the model for LangChain
logger.info("Initializing ChatOpenAI model")
try:
    model = ChatOpenAI(model="gpt-4o", api_key=api_key)
    logger.info("Model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing model: {str(e)}", exc_info=True)
    raise

# In-memory storage for user contexts
conversation_context = {}

# ======= Assistant Manager Functions as Tools =======

def create_assistant_tool(campaign_id: str) -> str:
    """
    Creates an assistant using the Campaign ID.
    
    Args:
        campaign_id: The ID of the campaign to create an assistant for
        
    Returns:
        A string with the assistant ID or error message
    """
    logger.info(f"Creating assistant for campaign: {campaign_id}")
    start_time = time.time()
    
    try:
        # Create the assistant using the OpenAI API
        new_assistant = openai_client.beta.assistants.create(
            instructions=(
                "You are an email writing assistant that will work with a client to understand "
                "their style and how they would like their emails to look."
            ),
            name=f"{campaign_id} Brain",
            model="gpt-4o"
        )
        
        duration = time.time() - start_time
        logger.info(f"Assistant created in {duration:.4f} seconds. ID: {new_assistant.id}")
        return f"Assistant created successfully with ID: {new_assistant.id}"
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error creating assistant after {duration:.4f} seconds: {str(e)}", exc_info=True)
        return f"Error creating assistant: {str(e)}"

def create_generation_assistant_tool(campaign_id: str) -> str:
    """
    Creates a specialized assistant for generating emails and LinkedIn messages.
    
    Args:
        campaign_id: The ID of the campaign to create an assistant for
        
    Returns:
        A string with the assistant ID or error message
    """
    logger.info(f"Creating generation assistant for campaign: {campaign_id}")
    start_time = time.time()
    
    try:
        instructions = (
            "You generate Emails and LinkedIn messages for [first_name], the [job_title] at [company]. "
            "You will then be asked to generate one email which will be email number [email_number] of "
            "a four-step sequence. Here's an overview of what [company] does: [company_overview]. "
            "And here's the product/solution/event [company] is promoting: [product_offered]. "
            "Start emails/messages with a brief reference to the content provided which you found interesting. "
            "Use the reference to show that we are interested in them, have researched their company, and "
            "understand how we can help them. Ensure the comment is positive, short, succinct, and avoids flattery. "
            "End the paragraph by saying it's interesting, congratulating them, or noting something specific "
            "that they said. Keep this section short, limited to 1-7 words. "
            "Start a new paragraph mentioning an exciting opportunity. Explain that we're hosting a networking "
            "lunch in [current_month+2], bringing together [types of leaders] in [nearest city to [LOCATION]]. "
            "Highlight that it's a great chance to meet other professionals and delve into 'Scaling growth in 2025'. "
            "Conclude with a brief call to action, encouraging [contact_first_name] to let you know if they are "
            "interested in joining, so you can send details. "
            "Start all emails/messages with 'Hi [contact_first_name]' and end with 'Best, [first_name]'. "
            "Ensure messages are written in [language]. Do not include headings, subject lines, or titles. "
            "Use a professional but confident tone. Keep LinkedIn messages shorter and less formal. "
            "Sequence-specific instructions: [email_number_text]."
        )
        
        new_assistant = openai_client.beta.assistants.create(
            instructions=instructions,
            name=f"{campaign_id} Message Generator",
            model="gpt-4o"
        )
        
        duration = time.time() - start_time
        logger.info(f"Generation assistant created in {duration:.4f} seconds. ID: {new_assistant.id}")
        return f"Generation assistant created successfully with ID: {new_assistant.id}"
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error creating generation assistant after {duration:.4f} seconds: {str(e)}", exc_info=True)
        return f"Error creating generation assistant: {str(e)}"

def retrieve_assistant_tool(assistant_id: str) -> str:
    """
    Retrieves information about an assistant.
    
    Args:
        assistant_id: The ID of the assistant to retrieve
        
    Returns:
        A string with the assistant info or error message
    """
    logger.info(f"Retrieving assistant: {assistant_id}")
    start_time = time.time()
    
    try:
        assistant = openai_client.beta.assistants.retrieve(assistant_id)
        
        # Format the assistant info as a readable string
        assistant_info = (
            f"Assistant ID: {assistant.id}\n"
            f"Name: {assistant.name}\n"
            f"Model: {assistant.model}\n"
            f"Instructions: {assistant.instructions[:200]}..." if len(assistant.instructions) > 200 
                                                            else assistant.instructions
        )
        
        duration = time.time() - start_time
        logger.info(f"Assistant retrieved in {duration:.4f} seconds")
        return assistant_info
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error retrieving assistant after {duration:.4f} seconds: {str(e)}", exc_info=True)
        return f"Error retrieving assistant: {str(e)}"

def modify_system_prompt_tool(assistant_id: str, new_system_prompt: str) -> str:
    """
    Updates the system prompt for an assistant.
    
    Args:
        assistant_id: The ID of the assistant to update
        new_system_prompt: The new system prompt to set
        
    Returns:
        A string with confirmation or error message
    """
    logger.info(f"Modifying system prompt for assistant: {assistant_id}")
    start_time = time.time()
    
    try:
        updated_assistant = openai_client.beta.assistants.update(
            assistant_id,
            {"instructions": new_system_prompt}
        )
        
        duration = time.time() - start_time
        logger.info(f"System prompt modified in {duration:.4f} seconds")
        logger.debug(f"New system prompt: {new_system_prompt[:100]}...")
        return f"Assistant {assistant_id} has been updated with new instructions"
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error modifying system prompt after {duration:.4f} seconds: {str(e)}", exc_info=True)
        return f"Error modifying system prompt: {str(e)}"

def delete_assistant_tool(assistant_id: str) -> str:
    """
    Deletes an assistant by ID.
    
    Args:
        assistant_id: The ID of the assistant to delete
        
    Returns:
        A string with confirmation or error message
    """
    logger.info(f"Deleting assistant: {assistant_id}")
    start_time = time.time()
    
    try:
        deletion_response = openai_client.beta.assistants.delete(assistant_id)
        
        duration = time.time() - start_time
        logger.info(f"Assistant deleted in {duration:.4f} seconds")
        
        if deletion_response.deleted:
            return f"Assistant {assistant_id} has been successfully deleted"
        else:
            return f"Failed to delete assistant {assistant_id}"
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error deleting assistant after {duration:.4f} seconds: {str(e)}", exc_info=True)
        return f"Error deleting assistant: {str(e)}"

def create_thread_tool() -> str:
    """
    Create a new thread for conversation with an assistant.
    
    Returns:
        A string with the thread ID or error message
    """
    logger.info("Creating new thread")
    start_time = time.time()
    
    try:
        new_thread = openai_client.beta.threads.create()
        
        duration = time.time() - start_time
        logger.info(f"Thread created in {duration:.4f} seconds. ID: {new_thread.id}")
        return f"Thread created successfully with ID: {new_thread.id}"
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error creating thread after {duration:.4f} seconds: {str(e)}", exc_info=True)
        return f"Error creating thread: {str(e)}"

def send_message_to_thread_tool(thread_id: str, message_content: str, role: str = "user") -> str:
    """
    Send a message to a thread.
    
    Args:
        thread_id: The ID of the thread to send the message to
        message_content: The message content
        role: The role of the message sender (user or system)
        
    Returns:
        A string with confirmation or error message
    """
    logger.info(f"Sending {role} message to thread: {thread_id}")
    start_time = time.time()
    
    try:
        message = openai_client.beta.threads.messages.create(
            thread_id,
            {"role": role, "content": message_content}
        )
        
        duration = time.time() - start_time
        logger.info(f"{role.capitalize()} message sent in {duration:.4f} seconds")
        logger.debug(f"Message content: {message_content[:100]}...")
        return f"{role.capitalize()} message sent successfully to thread {thread_id}"
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error sending {role} message after {duration:.4f} seconds: {str(e)}", exc_info=True)
        return f"Error sending {role} message: {str(e)}"

def run_assistant_on_thread_tool(assistant_id: str, thread_id: str) -> str:
    """
    Run an assistant on a thread and get the response.
    
    Args:
        assistant_id: The ID of the assistant to run
        thread_id: The ID of the thread to run the assistant on
        
    Returns:
        A string with the response or error message
    """
    logger.info(f"Running assistant {assistant_id} on thread {thread_id}")
    start_time = time.time()
    
    try:
        # Create a run
        run = openai_client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )
        
        logger.info(f"Created run {run.id}, waiting for completion...")
        
        # Poll for the run to complete (in a real app, you might use a more sophisticated approach)
        while True:
            run_status = openai_client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            
            if run_status.status == "completed":
                logger.info(f"Run {run.id} completed")
                break
            elif run_status.status in ["failed", "cancelled", "expired"]:
                logger.error(f"Run {run.id} {run_status.status} with reason: {run_status.last_error}")
                return f"Run {run_status.status}: {run_status.last_error}"
            
            logger.debug(f"Run {run.id} status: {run_status.status}, waiting...")
            time.sleep(1)  # Wait before polling again
        
        # Get the latest message
        messages = openai_client.beta.threads.messages.list(thread_id=thread_id)
        
        if not messages.data:
            return "No messages found after run completion"
        
        latest_message = messages.data[0]
        message_content = latest_message.content[0].text.value
        
        duration = time.time() - start_time
        logger.info(f"Assistant response received in {duration:.4f} seconds")
        logger.debug(f"Response content: {message_content[:100]}...")
        
        return f"Assistant response: {message_content}"
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error running assistant after {duration:.4f} seconds: {str(e)}", exc_info=True)
        return f"Error running assistant: {str(e)}"

def get_messages_from_thread_tool(thread_id: str) -> str:
    """
    Get all messages from a thread.
    
    Args:
        thread_id: The ID of the thread to get messages from
        
    Returns:
        A string with the messages or error message
    """
    logger.info(f"Getting messages from thread: {thread_id}")
    start_time = time.time()
    
    try:
        messages = openai_client.beta.threads.messages.list(thread_id=thread_id)
        
        if not messages.data:
            return "No messages found in thread"
        
        # Format messages as a readable string
        formatted_messages = "Thread messages:\n\n"
        
        for i, msg in enumerate(messages.data):
            content = msg.content[0].text.value if msg.content else "No content"
            formatted_messages += f"{i+1}. {msg.role}: {content[:200]}...\n\n" if len(content) > 200 else f"{i+1}. {msg.role}: {content}\n\n"
        
        duration = time.time() - start_time
        logger.info(f"Retrieved {len(messages.data)} messages in {duration:.4f} seconds")
        
        return formatted_messages
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error getting messages after {duration:.4f} seconds: {str(e)}", exc_info=True)
        return f"Error getting messages: {str(e)}"

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

def send_response_to_frontend(response_data: Dict[str, Any]) -> bool:
    """Send the response data to the frontend via a POST request."""
    logger.info("Sending response to frontend")
    start_time = time.time()
    
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(frontend_url, json=response_data, headers=headers)
        
        if response.status_code == 200:
            duration = time.time() - start_time
            logger.info(f"Response successfully sent to frontend in {duration:.4f} seconds")
            return True
        else:
            duration = time.time() - start_time
            logger.error(f"Failed to send response after {duration:.4f} seconds. Status code: {response.status_code}, Response: {response.text}")
            return False
    
    except requests.exceptions.RequestException as e:
        duration = time.time() - start_time
        logger.error(f"Error sending response to frontend after {duration:.4f} seconds: {str(e)}", exc_info=True)
        return False

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
            get_messages_from_thread_tool
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
            "- get_messages_from_thread_tool: When a user wants to see the history of messages in a thread\n"
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

# Create supervisor workflow
logger.info("Creating supervisor workflow")
try:
    supervisor = create_supervisor(
        [assistant_mgmt_agent, thread_mgmt_agent, email_agent],
        model=model,
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

def process_query(query: str, user_id: str = "default_user", history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Process a user query through the agent system.
    
    Args:
        query: The user's input message
        user_id: A unique identifier for the user
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
        
        # Prepare messages with history if provided
        messages = []
        if history:
            logger.info(f"Using conversation history with {len(history)} messages")
            messages.extend(history[-5:])  # Use last 5 messages for context
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        # Process via supervisor workflow
        logger.info("Invoking supervisor workflow")
        result = workflow.invoke({"messages": messages})
        
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

# ======= Main Function =======

if __name__ == "__main__":
    # Test assistant creation
    print("\n--- Testing Assistant Creation ---")
    campaign_id = f"TestCampaign_{int(time.time())}"
    assistant_response = create_assistant_tool(campaign_id)
    print(f"Creation result: {assistant_response}")
    
    # Extract assistant ID from response
    assistant_id = None
    match = re.search(r'asst_[a-zA-Z0-9]+', assistant_response)
    if match:
        assistant_id = match.group(0)
        print(f"Created assistant ID: {assistant_id}")
    
    if assistant_id:
        # Test thread creation
        print("\n--- Testing Thread Creation ---")
        thread_response = create_thread_tool()
        thread_id = None
        match = re.search(r'thread_[a-zA-Z0-9]+', thread_response)
        if match:
            thread_id = match.group(0)
            print(f"Created thread ID: {thread_id}")
        
        if thread_id:
            # Test sending a message
            print("\n--- Testing Message Sending ---")
            message = "I need help drafting a professional email to introduce our services"
            message_response = send_message_to_thread_tool(thread_id, message)
            print(f"Message result: {message_response}")
            
            # Test running the assistant
            print("\n--- Testing Assistant Run ---")
            run_response = run_assistant_on_thread_tool(assistant_id, thread_id)
            print(f"Run result: {run_response}")
            
            # Cleanup
            print("\n--- Cleanup ---")
            # delete_response = delete_assistant_tool(assistant_id)
            # print(f"Deletion result: {delete_response}")
