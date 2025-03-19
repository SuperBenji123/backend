from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

from AssistantManager import create_generation_assistant, retrieve_assistant, modify_system_prompt, get_all_messages, get_most_recent_message, delete_assistant, create_thread, retrieve_thread, create_user_message, create_system_message, run_assistant_on_thread

import requests
from dotenv import load_dotenv
import os
import json
import logging
import re

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

# Initialize the model
model = ChatOpenAI(model="gpt-4o", api_key=api_key)

# Define tool functions
def web_search(query: str) -> str:
    """Search the web for information."""
    # This is a mock implementation. In a real system, you might use 
    # an actual search API like Google Custom Search or Bing
    if "weather" in query.lower():
        return "Current weather: Partly cloudy with temperatures around 72°F."
    elif "news" in query.lower():
        return "Latest news: New advances in AI technology announced today."
    elif "company" in query.lower() or "business" in query.lower():
        return (
            "Company information:\n"
            "1. **Apple**: Market cap $2.8T, 164,000 employees\n"
            "2. **Microsoft**: Market cap $2.7T, 181,000 employees\n"
            "3. **Google**: Market cap $1.7T, 181,269 employees"
        )
    else:
        return f"Search results for {query}: Found multiple relevant information sources."

def draft_email(subject: str, recipient: str, content: str) -> str:
    """Draft an email with the given subject, recipient, and content."""
    email_template = f"""
    To: {recipient}
    Subject: {subject}
    
    {content}
    
    Best regards,
    AI Assistant
    """
    logger.info("drafted email")
    return email_template

def analyze_sentiment(text: str) -> str:
    """Analyze the sentiment of the given text."""
    positive_words = ["good", "great", "excellent", "happy", "positive", "amazing"]
    negative_words = ["bad", "terrible", "sad", "negative", "awful", "poor"]
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return "Positive sentiment detected in the text."
    elif negative_count > positive_count:
        return "Negative sentiment detected in the text."
    else:
        return "Neutral sentiment detected in the text."
    
def send_response_to_frontend(response_data: dict):
    """Send the response data to the frontend via a POST request."""
    frontend_url = "http://localhost:3000/api/response"  # Adjust this URL as per your React app

    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(frontend_url, json=response_data, headers=headers)
        
        if response.status_code == 200:
            print("Response successfully sent to frontend.")
        else:
            print(f"Failed to send response. Status code: {response.status_code}, Response: {response.text}")
    
    except requests.exceptions.RequestException as e:
        print(f"Error sending response to frontend: {e}")


# Create specialized agents
research_agent = create_react_agent(
    model=model,
    tools=[web_search, analyze_sentiment],
    name="research_expert",
    prompt=(
        "You are a world-class researcher with access to web search and sentiment analysis tools. "
        "You can find information on the web and analyze the sentiment of text. "
        "When asked to research something, always use the web_search tool. "
        "When asked about emotions or feelings in text, use the analyze_sentiment tool."
        "When research has been completed capitalise every character in the response and return it."
    )
)

email_agent = create_react_agent(
    model=model,
    tools=[draft_email],
    name="email_expert",
    prompt=(
        "You are an email writing expert. You specialize in drafting professional emails. "
        "When asked to create an email, always use the draft_email tool with appropriate "
        "subject, recipient, and content parameters. Be professional and concise."
    )
)

# Create supervisor workflow
supervisor = create_supervisor(
    [research_agent, email_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a research expert and an email expert. "
        "For questions about information, web searches, or sentiment analysis, use research_expert. "
        "For tasks related to creating or drafting emails, use email_expert. "
        "Make sure to route each query to the appropriate expert."
    )
)
logger.info("Running supervisor agent")

email_training_agent = create_react_agent(
    model=model,
    tools=[web_search, analyze_sentiment],
    name="research_expert",
    prompt=(
        "You are a world-class researcher with access to web search and sentiment analysis tools. "
        "You can find information on the web and analyze the sentiment of text. "
        "When asked to research something, always use the web_search tool. "
        "When asked about emotions or feelings in text, use the analyze_sentiment tool."
    )
)

# Compile the workflow
workflow = supervisor.compile()

def process_query(query: str) -> dict:
    """
    Process a user query through the agent system and return a JSON response.
    
    Args:
        query: The user's question or request
        
    Returns:
        A dictionary containing the processed response and metadata
    """
    try:
        # Classify the query to determine intent
        intent = classify_intent(query)
        
        # Invoke the appropriate workflow
        result = workflow.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        })
        
        # Extract the response content from the result
        response_content = ""
        if isinstance(result, dict) and "messages" in result:
            for message in reversed(result["messages"]):
                if hasattr(message, 'content') and message.content:
                    response_content = message.content
                    break
        
        # Format the response as a dictionary
        response = {
            "success": True,
            "intent": intent,
            "response": response_content,
            "source": "agent"
        }
        logger.info("Process query")
        return response
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "intent": "unknown"
        }

def classify_intent(query: str) -> str:
    """Classify the intent of a user query."""
    query_lower = query.lower()
    
    # Search-related keywords
    search_keywords = ["search", "find", "look up", "information", "tell me about", 
                       "what is", "who is", "where is", "when", "why", "how"]
    
    # Email-related keywords
    email_keywords = ["email", "draft", "compose", "write", "message", "send",
                      "to:", "subject:", "dear", "sincerely"]
    
    # Check for search intent
    if any(keyword in query_lower for keyword in search_keywords):
        return "search"
    
    # Check for email intent
    if any(keyword in query_lower for keyword in email_keywords):
        return "email"
    
    # Default to general query
    return "general"

# Example usage
if __name__ == "__main__":
    # Test with a search query
    search_query = "What's the latest weather?"
    search_result = process_query(search_query)
    print(f"Search Query: {search_query}")
    print(f"Result: {json.dumps(search_result, indent=2)}")
    
    # Test with an email query
    email_query = "Draft an email to john@example.com about the project update"
    email_result = process_query(email_query)
    print(f"\nEmail Query: {email_query}")
    print(f"Result: {json.dumps(email_result, indent=2)}")
