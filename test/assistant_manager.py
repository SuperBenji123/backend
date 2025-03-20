import openai
import logging
import time
import os
from typing import Dict, Any, Optional, List, Union
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API keys from environment variables
api_key = os.getenv("OPENAI_API_KEY")
assistants_api_key = os.getenv("OPENAI_ASSISTANTS_API_KEY", api_key)  # Use main API key as fallback

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=assistants_api_key)

# ======= Assistant Management Functions =======

def create_assistant(campaign_id: str) -> Dict[str, Any]:
    """
    Creates an assistant using the Campaign ID.
    
    Args:
        campaign_id: The ID of the campaign to create an assistant for
        
    Returns:
        A dictionary with success status, assistant ID, and message
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
        
        return {
            "success": True,
            "assistant_id": new_assistant.id,
            "message": f"Assistant created successfully with ID: {new_assistant.id}"
        }
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error creating assistant after {duration:.4f} seconds: {str(e)}", exc_info=True)
        
        return {
            "success": False,
            "assistant_id": None,
            "message": f"Error creating assistant: {str(e)}"
        }

def create_generation_assistant(campaign_id: str) -> Dict[str, Any]:
    """
    Creates a specialized assistant for generating emails and LinkedIn messages.
    
    Args:
        campaign_id: The ID of the campaign to create an assistant for
        
    Returns:
        A dictionary with success status, assistant ID, and message
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
        
        return {
            "success": True,
            "assistant_id": new_assistant.id,
            "message": f"Generation assistant created successfully with ID: {new_assistant.id}"
        }
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error creating generation assistant after {duration:.4f} seconds: {str(e)}", exc_info=True)
        
        return {
            "success": False,
            "assistant_id": None,
            "message": f"Error creating generation assistant: {str(e)}"
        }

def retrieve_assistant(assistant_id: str) -> Dict[str, Any]:
    """
    Retrieves information about an assistant.
    
    Args:
        assistant_id: The ID of the assistant to retrieve
        
    Returns:
        A dictionary with success status, assistant data, and message
    """
    logger.info(f"Retrieving assistant: {assistant_id}")
    start_time = time.time()
    
    try:
        assistant = openai_client.beta.assistants.retrieve(assistant_id)
        
        # Format the assistant info
        assistant_data = {
            "id": assistant.id,
            "name": assistant.name,
            "model": assistant.model,
            "instructions": assistant.instructions
        }
        
        duration = time.time() - start_time
        logger.info(f"Assistant retrieved in {duration:.4f} seconds")
        
        return {
            "success": True,
            "assistant": assistant_data,
            "message": f"Assistant {assistant_id} retrieved successfully"
        }
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error retrieving assistant after {duration:.4f} seconds: {str(e)}", exc_info=True)
        
        return {
            "success": False,
            "assistant": None,
            "message": f"Error retrieving assistant: {str(e)}"
        }

def modify_system_prompt(assistant_id: str, new_system_prompt: str) -> Dict[str, Any]:
    """
    Updates the system prompt for an assistant.
    
    Args:
        assistant_id: The ID of the assistant to update
        new_system_prompt: The new system prompt to set
        
    Returns:
        A dictionary with success status and message
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
        
        return {
            "success": True,
            "message": f"Assistant {assistant_id} has been updated with new instructions"
        }
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error modifying system prompt after {duration:.4f} seconds: {str(e)}", exc_info=True)
        
        return {
            "success": False,
            "message": f"Error modifying system prompt: {str(e)}"
        }

def delete_assistant(assistant_id: str) -> Dict[str, Any]:
    """
    Deletes an assistant by ID.
    
    Args:
        assistant_id: The ID of the assistant to delete
        
    Returns:
        A dictionary with success status and message
    """
    logger.info(f"Deleting assistant: {assistant_id}")
    start_time = time.time()
    
    try:
        deletion_response = openai_client.beta.assistants.delete(assistant_id)
        
        duration = time.time() - start_time
        logger.info(f"Assistant deleted in {duration:.4f} seconds")
        
        if deletion_response.deleted:
            return {
                "success": True,
                "message": f"Assistant {assistant_id} has been successfully deleted"
            }
        else:
            return {
                "success": False,
                "message": f"Failed to delete assistant {assistant_id}"
            }
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error deleting assistant after {duration:.4f} seconds: {str(e)}", exc_info=True)
        
        return {
            "success": False,
            "message": f"Error deleting assistant: {str(e)}"
        }

# ======= Thread Management Functions =======

def create_thread() -> Dict[str, Any]:
    """
    Create a new thread for conversation with an assistant.
    
    Returns:
        A dictionary with success status, thread ID, and message
    """
    logger.info("Creating new thread")
    start_time = time.time()
    
    try:
        new_thread = openai_client.beta.threads.create()
        
        duration = time.time() - start_time
        logger.info(f"Thread created in {duration:.4f} seconds. ID: {new_thread.id}")
        
        return {
            "success": True,
            "thread_id": new_thread.id,
            "message": f"Thread created successfully with ID: {new_thread.id}"
        }
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error creating thread after {duration:.4f} seconds: {str(e)}", exc_info=True)
        
        return {
            "success": False,
            "thread_id": None,
            "message": f"Error creating thread: {str(e)}"
        }

def send_message_to_thread(thread_id: str, message_content: str, role: str = "user") -> Dict[str, Any]:
    """
    Send a message to a thread.
    
    Args:
        thread_id: The ID of the thread to send the message to
        message_content: The message content
        role: The role of the message sender (user or system)
        
    Returns:
        A dictionary with success status and message
    """
    logger.info(f"Sending {role} message to thread: {thread_id}")
    start_time = time.time()
    
    try:
        # The OpenAI API expects the thread_id first, then the message object
        message = openai_client.beta.threads.messages.create(
            thread_id,
            role=role,
            content=message_content
        )
        
        duration = time.time() - start_time
        logger.info(f"{role.capitalize()} message sent in {duration:.4f} seconds")
        logger.debug(f"Message content: {message_content[:100]}...")
        
        return {
            "success": True,
            "message": f"{role.capitalize()} message sent successfully to thread {thread_id}",
            "message_id": message.id
        }
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error sending {role} message after {duration:.4f} seconds: {str(e)}", exc_info=True)
        
        return {
            "success": False,
            "message": f"Error sending {role} message: {str(e)}",
            "message_id": None
        }

def run_assistant_on_thread(assistant_id: str, thread_id: str, wait_for_completion: bool = True) -> Dict[str, Any]:
    """
    Run an assistant on a thread and get the response.
    
    Args:
        assistant_id: The ID of the assistant to run
        thread_id: The ID of the thread to run the assistant on
        wait_for_completion: Whether to wait for the run to complete
        
    Returns:
        A dictionary with success status, run details, and message
    """
    logger.info(f"Running assistant {assistant_id} on thread {thread_id}")
    start_time = time.time()
    
    try:
        # Create a run
        run = openai_client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )
        
        logger.info(f"Created run {run.id}")
        
        if not wait_for_completion:
            return {
                "success": True,
                "run_id": run.id,
                "status": run.status,
                "message": f"Run initiated with ID: {run.id}",
                "response_content": None
            }
        
        # Poll for the run to complete
        response_content = None
        while True:
            run_status = openai_client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            
            if run_status.status == "completed":
                logger.info(f"Run {run.id} completed")
                # Get the latest message
                messages = openai_client.beta.threads.messages.list(thread_id=thread_id)
                
                if messages.data:
                    latest_message = messages.data[0]
                    response_content = latest_message.content[0].text.value
                break
            elif run_status.status in ["failed", "cancelled", "expired"]:
                logger.error(f"Run {run.id} {run_status.status} with reason: {run_status.last_error}")
                return {
                    "success": False,
                    "run_id": run.id,
                    "status": run_status.status,
                    "message": f"Run {run_status.status}: {run_status.last_error}",
                    "response_content": None
                }
            
            logger.debug(f"Run {run.id} status: {run_status.status}, waiting...")
            time.sleep(1)  # Wait before polling again
        
        duration = time.time() - start_time
        logger.info(f"Assistant response received in {duration:.4f} seconds")
        
        return {
            "success": True,
            "run_id": run.id,
            "status": "completed",
            "message": f"Run completed successfully",
            "response_content": response_content
        }
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error running assistant after {duration:.4f} seconds: {str(e)}", exc_info=True)
        
        return {
            "success": False,
            "run_id": None,
            "status": "error",
            "message": f"Error running assistant: {str(e)}",
            "response_content": None
        }

def get_thread_messages(thread_id: str, limit: int = 20) -> Dict[str, Any]:
    """
    Get messages from a thread.
    
    Args:
        thread_id: The ID of the thread to get messages from
        limit: Maximum number of messages to retrieve
        
    Returns:
        A dictionary with success status, messages, and message
    """
    logger.info(f"Getting messages from thread: {thread_id}")
    start_time = time.time()
    
    try:
        messages = openai_client.beta.threads.messages.list(
            thread_id=thread_id,
            limit=limit
        )
        
        # Format messages for easier handling
        formatted_messages = []
        for msg in messages.data:
            content = msg.content[0].text.value if msg.content else "No content"
            formatted_messages.append({
                "id": msg.id,
                "role": msg.role,
                "content": content,
                "created_at": msg.created_at
            })
        
        duration = time.time() - start_time
        logger.info(f"Retrieved {len(formatted_messages)} messages in {duration:.4f} seconds")
        
        return {
            "success": True,
            "messages": formatted_messages,
            "message": f"Retrieved {len(formatted_messages)} messages from thread {thread_id}"
        }
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error getting messages after {duration:.4f} seconds: {str(e)}", exc_info=True)
        
        return {
            "success": False,
            "messages": [],
            "message": f"Error getting messages: {str(e)}"
        }

def check_run_status(thread_id: str, run_id: str) -> Dict[str, Any]:
    """
    Check the status of a run.
    
    Args:
        thread_id: The ID of the thread
        run_id: The ID of the run to check
        
    Returns:
        A dictionary with success status, run status, and message
    """
    logger.info(f"Checking status of run {run_id} on thread {thread_id}")
    start_time = time.time()
    
    try:
        run_status = openai_client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )
        
        duration = time.time() - start_time
        logger.info(f"Run status checked in {duration:.4f} seconds: {run_status.status}")
        
        # If the run is completed, get the latest message
        response_content = None
        if run_status.status == "completed":
            messages = openai_client.beta.threads.messages.list(thread_id=thread_id, limit=1)
            if messages.data:
                response_content = messages.data[0].content[0].text.value
        
        return {
            "success": True,
            "status": run_status.status,
            "message": f"Run status: {run_status.status}",
            "response_content": response_content
        }
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error checking run status after {duration:.4f} seconds: {str(e)}", exc_info=True)
        
        return {
            "success": False,
            "status": "error",
            "message": f"Error checking run status: {str(e)}",
            "response_content": None
        }

# Tool function versions for use in LangChain/LangGraph tools

def create_assistant_tool(campaign_id: str) -> str:
    """Tool version of create_assistant for LangGraph integration"""
    result = create_assistant(campaign_id)
    return result.get("message")

def create_generation_assistant_tool(campaign_id: str) -> str:
    """Tool version of create_generation_assistant for LangGraph integration"""
    result = create_generation_assistant(campaign_id)
    return result.get("message")

def retrieve_assistant_tool(assistant_id: str) -> str:
    """Tool version of retrieve_assistant for LangGraph integration"""
    result = retrieve_assistant(assistant_id)
    if result.get("success"):
        assistant = result.get("assistant", {})
        return (
            f"Assistant ID: {assistant.get('id')}\n"
            f"Name: {assistant.get('name')}\n"
            f"Model: {assistant.get('model')}\n"
            f"Instructions: {assistant.get('instructions')[:200]}..."
        )
    return result.get("message")

def modify_system_prompt_tool(assistant_id: str, new_system_prompt: str) -> str:
    """Tool version of modify_system_prompt for LangGraph integration"""
    result = modify_system_prompt(assistant_id, new_system_prompt)
    return result.get("message")

def delete_assistant_tool(assistant_id: str) -> str:
    """Tool version of delete_assistant for LangGraph integration"""
    result = delete_assistant(assistant_id)
    return result.get("message")

def create_thread_tool() -> str:
    """Tool version of create_thread for LangGraph integration"""
    result = create_thread()
    return result.get("message")

def send_message_to_thread_tool(thread_id: str, message_content: str, role: str = "user") -> str:
    """Tool version of send_message_to_thread for LangGraph integration"""
    result = send_message_to_thread(thread_id, message_content, role)
    return result.get("message")

def run_assistant_on_thread_tool(assistant_id: str, thread_id: str) -> str:
    """Tool version of run_assistant_on_thread for LangGraph integration"""
    result = run_assistant_on_thread(assistant_id, thread_id)
    if result.get("success") and result.get("response_content"):
        return f"Assistant response: {result.get('response_content')}"
    return result.get("message")

def get_thread_messages_tool(thread_id: str) -> str:
    """Tool version of get_thread_messages for LangGraph integration"""
    result = get_thread_messages(thread_id)
    if result.get("success"):
        messages = result.get("messages", [])
        formatted_text = "Thread messages:\n\n"
        for i, msg in enumerate(messages):
            content = msg.get("content", "No content")
            formatted_text += f"{i+1}. {msg.get('role')}: {content[:200]}...\n\n" if len(content) > 200 else f"{i+1}. {msg.get('role')}: {content}\n\n"
        return formatted_text
    return result.get("message")


# Test function to run if this file is executed directly
def test_assistant_manager():
    """Test the assistant manager functions"""
    print("\n--- Testing Assistant Creation ---")
    campaign_id = f"TestCampaign_{int(time.time())}"
    assistant_result = create_assistant(campaign_id)
    print(f"Creation result: {assistant_result}")
    
    if assistant_result.get("success"):
        assistant_id = assistant_result.get("assistant_id")
        
        print("\n--- Testing Thread Creation ---")
        thread_result = create_thread()
        print(f"Thread creation result: {thread_result}")
        
        if thread_result.get("success"):
            thread_id = thread_result.get("thread_id")
            
            print("\n--- Testing Message Sending ---")
            message = "I need help drafting a professional email to introduce our services"
            message_result = send_message_to_thread(thread_id, message)
            print(f"Message result: {message_result}")
            
            print("\n--- Testing Assistant Run ---")
            run_result = run_assistant_on_thread(assistant_id, thread_id)
            print(f"Run result: {run_result}")
            
            print("\n--- Testing Thread Messages ---")
            messages_result = get_thread_messages(thread_id)
            print(f"Messages: {messages_result}")
        
        print("\n--- Testing Assistant Deletion ---")
        delete_result = delete_assistant(assistant_id)
        print(f"Deletion result: {delete_result}")


if __name__ == "__main__":
    test_assistant_manager()
