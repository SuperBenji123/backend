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
import aiohttp
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

#TEMP_DATABASE
users = {
    "default_user_id": ["Hello, bot message", "Hey there, user message"]
}

email = ""

# Get API keys from environment variables
api_key = os.getenv("OPENAI_API_KEY")
assistants_api_key = os.getenv("OPENAI_ASSISTANTS_API_KEY", api_key)  # Use main API key as fallback
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000/api/response")

if not api_key:
    logger.critical("OPENAI_API_KEY not found in environment variables")
    sys.exit(1)
else:
    logger.info("API keys loaded successfully")
    logger.info(api_key)

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
                 "Output a randomly generated email sent by user to oscar about selling Super Benji"
             ),
             name=f"{campaign_id} Brain",
             model="gpt-4o"
         )
 
         duration = time.time() - start_time
         logger.info(f"Assistant created in {duration:.4f} seconds. ID: {new_assistant.id}")
         return f"Assistant created successfully with ID: {new_assistant.id}{api_key}"
     except Exception as e:
         duration = time.time() - start_time
         logger.error(f"Error creating assistant after {duration:.4f} seconds: {str(e)}", exc_info=True)
         return f"Error creating assistant: {str(e)}"

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
        new_instructions = {"instructions": new_system_prompt}
        
        updated_assistant = openai_client.beta.assistants.update(
            assistant_id=assistant_id,  # Must be passed as a keyword argument
            instructions=new_system_prompt  # Directly pass the value instead of a dict
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

def create_email_generation_thread_tool() -> str:
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

def send_message_to_email_generation_thread_tool(thread_id: str, message_content: str, role: str = "user") -> str:
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

    message_to_be_sent = {"role": role, "content": message_content}
    logger.info({message_to_be_sent})
    
    try:
        message = openai_client.beta.threads.messages.create(
            thread_id,
            message_to_be_sent
        )
        
        duration = time.time() - start_time
        logger.info(f"{role.capitalize()} message sent in {duration:.4f} seconds")
        logger.debug(f"Message content: {message_content[:100]}...")
        return f"{role.capitalize()} message sent successfully to thread {thread_id}"
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error sending {role} message after {duration:.4f} seconds: {str(e)}", exc_info=True)
        return f"Error sending {role} message: {str(e)}"

def run_email_assistant_on_thread_tool(assistant_id: str, thread_id: str) -> str:
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
        global email
        
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
        logger.error(f"Response content: {message_content[:100]}...")
        email = message_content

        logger.error(f"{email}")
        
        return f"Assistant response: {message_content}"
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error running assistant after {duration:.4f} seconds: {str(e)}", exc_info=True)
        return f"Error running assistant: {str(e)}"

def get_messages_from_email_generation_thread_tool(thread_id: str) -> str:
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


#Find Prospects Tool
def find_prospects_tool(prompt: str) -> dict:
    """
    Call Make Webhook to retrieve prospects for a given client.
    
    Args:
        prompt: User inputted prompt to send to the webhook
        
    Returns:
        A dictionary containing a series of prospects and the information about them
    """
    logger.info(f"Finding prospects based on this prompt: '{prompt[:50]}...' (truncated)")
    start_time = time.time()
    
    url = "https://hook.eu2.make.com/ytq23ywoyd4x2ve23fjdxhc1n1wluj6a"
    headers = {"Content-Type": "application/json"}
    payload = {"prompt": prompt}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for HTTP failure responses (4xx, 5xx)
        data = response.json()
        
        duration = time.time() - start_time
        logger.info(f"Prospects retrieved successfully in {duration:.4f} seconds")
        return data
    
    except requests.exceptions.RequestException as e:
        duration = time.time() - start_time
        logger.error(f"Error finding prospects in {duration:.4f} seconds: {str(e)}", exc_info=True)
        return {"error": str(e)}


# ======= Utility Functions =======

def classify_user_intent(text: str) -> str:
    """
    Classify the user's intent based on their message
    
    Args:
        text: The user's input message
        
    Returns:
        The classified intent: "assistant_management", "email_creation", 
        "system_improvement", "prospecting" or "general_conversation"
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
        "help with email", "email template", "email format", "email", "write an email"
    ]
    
    # System improvement keywords
    improvement_keywords = [
        "improve", "enhance", "upgrade", "better", "smarter",
        "learn to", "teach you", "adjust your", "change your style",
        "be more", "sound more", "write more", "different tone",
        "feedback", "suggestion", "preference", "like it when you"
    ]

    # Prospecting Keywords
    prospecting_keywords = [
        "prospect", "prospects", "contacts", "people"
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
    # Check for prospecting intent
    elif any(keyword in text_lower for keyword in prospecting_keywords):
        intent = "prospecting"
    # Default intent
    else:
        intent = "general_conversation"
    
    duration = time.time() - start_time
    logger.debug(f"Intent classification completed in {duration:.4f} seconds. Intent: {intent}")
    return intent

#CHANGE THIS TO JUST TAKE AN INPUT FROM THE FRONTEND OR FROM THE DATABASE
def extract_assistant_id(text: str) -> Optional[str]:
    """Extract assistant ID from text if present"""
    logger.debug(f"Extracting assistant ID from text: '{text[:50]}...' (truncated)")
    start_time = time.time()
    
    patterns = [
        r'assistant[_\s]?id[:\s]?\s*["\'`]?([a-zA-Z0-9_-]+)["\'`]?',
        r'asst_[a-zA-Z0-9]{24}'
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

#CHANGE THIS TO JUST TAKE AN INPUT FROM THE FRONTEND OR FROM THE DATABASE
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

#CHANGE THIS TO JUST TAKE AN INPUT FROM THE FRONTEND OR FROM THE DATABASE
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

# ======= Create Agents =======

# Create assistant management agent
logger.info("Creating assistant management agent")
try:
    assistant_mgmt_agent = create_react_agent(
        model=model,
        tools=[
            create_assistant_tool,
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
            "- create_assistant_tool: When a user or the system wants to make an new email generation assistant for a user's campaign\n"
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
logger.info("Creating message generation management agent")
try:
    message_generation_mgmt_agent = create_react_agent(
        model=model,
        tools=[
            send_message_to_email_generation_thread_tool,
            run_email_assistant_on_thread_tool,
            get_messages_from_email_generation_thread_tool
        ],
        name="message_generation_mgmt_agent",
        prompt=(
            "You are an expert in managing OpenAI assistant threads and interactions to create emails. "
            "You use these threads to create emails for the user"
            "You can create threads, send messages to threads, run assistants on threads, and get messages from threads. "
            "\n\n"
            "Use these tools based on user requests:\n"
            "- send_message_to_email_generation_thread_tool: When a user wants to send a message to a thread to generate an email\n"
            "- run_email_assistant_on_thread_tool: When a user or agent has sent a message to the thread and then wants to generate an email on a thread\n"
            "- get_messages_from_email_generation_thread_tool: When a user wants to see the history of messages in a thread\n"
        )
    )
    logger.info("Thread management agent created successfully")
except Exception as e:
    logger.error(f"Error creating thread management agent: {str(e)}", exc_info=True)
    raise

# Create thread management agent
logger.info("Creating prospecting agent")
try:
    prospecting_agent = create_react_agent(
        model=model,
        tools=[
            find_prospects_tool
        ],
        name="prospecting_agent",
        prompt=(
            "You are an expert in finding sales prospects for a given user. "
            "You can find prospects, filter them, and evaluate them"
            "\n\n"
            "Use these tools based on user requests:\n"
            "- find_prospcts_tool: When a user wants to find new prospects\n"
        )
    )
    logger.info("Prospecting agent created successfully")
except Exception as e:
    logger.error(f"Error creating prospecting agent: {str(e)}", exc_info=True)
    raise

# Create supervisor workflow
logger.info("Creating supervisor workflow")
try:
    supervisor = create_supervisor(
        [assistant_mgmt_agent, message_generation_mgmt_agent, prospecting_agent],
        model=model,
        prompt=(
            "You are a team supervisor managing three experts:\n"
            "1. The assistant_manager handles creating, updating, and deleting OpenAI assistants\n"
            "2. The message_generation_mgmt_agent manages generating emails on a thread using the user's assistant\n"
            "3. The prospecting_agent handles finding and evaluating new prospects\n\n"
            "For tasks related to creating or managing OpenAI assistants, use assistant_manager.\n"
            "For tasks related to email generation use message_generation_mgmt_agent.\n"
            "For tasks related to finding prospects, use prospecting expert.\n"
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

    if user_id not in users:
        assistant_id = extract_assistant_id(create_assistant_tool(user_id))
        thread_id = extract_thread_id(create_email_generation_thread_tool())
        
        users[user_id] = [[{"role": "ai", "content": "Are you ready?"}],[],assistant_id, thread_id]
        
        logger.info(f"User added to memory")

    try:
        logs = []
        for logmessages in users[user_id][1]:
            logs.append(logmessages)
        

        # Classify the query to determine intent
        intent = classify_user_intent(query)
        logger.info(f"Query classified with intent: {intent}")
        
        
        # Prepare messages with history if provided
        messages = []
        for chats in users[user_id][0]:
            messages.append(chats)
        
        # Add current query
        messages.append({"role": "user", "content": query})

        global email
        email = ""

        assistant_id = users[user_id][2]
        thread_id = users[user_id][3]
        logger.info(f"{assistant_id} and {thread_id}")
        messages.append({"role": "user", "content": f"My Assistant ID is {assistant_id} and my Thread ID is {thread_id}"})
    
        
        # Process via supervisor workflow
        logger.info("Invoking supervisor workflow")
        result = workflow.invoke({"messages": messages, "logs": logs, "assistant_id": assistant_id, "thread_id": thread_id})

        #Any Logs we want to remember we do this 
        logs.append("Invoking supervisor workflow")

        logger.error(f"{email}")
        
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
            "processing_time": round(time.time() - start_time, 4),
            "email": email,
        }

        #Adds agent response message to messages
        messages.append({"role": "ai", "content": response_content})
        
        logger.error(messages)
        logger.error(logs)
        
        #replacing messages and logs with up to date versions of each
        users[user_id][0] = messages
        users[user_id][1] = logs


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
