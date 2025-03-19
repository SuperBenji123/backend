import openai

async def create_assistant(campaign_id, openai_client):
    """Creates an assistant using the Campaign ID of the current seat being onboarded."""
    new_assistant = await openai_client.beta.assistants.create(
        instructions=(
            "You are an email writing assistant that will work with a client to understand "
            "their style and how they would like their emails to look."
        ),
        name=f"{campaign_id} Brain",
        model="gpt-4o"
    )
    return new_assistant


async def create_generation_assistant(campaign_id, openai_client):
    """Creates an assistant with specific instructions for generating emails and LinkedIn messages."""
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

    try:
        new_assistant = await openai_client.beta.assistants.create(
            instructions=instructions,
            name=f"{campaign_id} Message Generator",
            model="gpt-4o"
        )
        print(new_assistant.id)
        return new_assistant
    except Exception as error:
        print("Error creating assistant:", error)
        return None


async def retrieve_assistant(assistant_id, openai_client):
    """Retrieves an assistant by its ID."""
    return await openai_client.beta.assistants.retrieve(assistant_id)


async def modify_system_prompt(assistant_id, new_system_prompt, openai_client):
    """Modifies the system prompt of an assistant."""
    updated_assistant = await openai_client.beta.assistants.update(
        assistant_id,
        {"instructions": new_system_prompt}
    )
    print(updated_assistant)
    return f"Assistant {assistant_id} has been updated with new instructions"


async def delete_assistant(assistant_id, openai_client):
    """Deletes an assistant by its ID."""
    return await openai_client.beta.assistants.delete(assistant_id)


async def create_thread(openai_client):
    """Creates a new thread."""
    new_thread = await openai_client.beta.threads.create()
    print(new_thread)
    return new_thread


async def retrieve_thread(thread_id, openai_client):
    """Retrieves a thread by its ID."""
    thread = await openai_client.beta.threads.retrieve(thread_id)
    print(thread)
    return thread


async def create_user_message(thread_id, message_content, openai_client):
    """Creates a user message in a thread."""
    thread_messages = await openai_client.beta.threads.messages.create(
        thread_id, {"role": "user", "content": message_content}
    )
    print("User Message sent to Assistant with thread:", thread_id)
    return thread_messages


async def create_system_message(thread_id, message_content, openai_client):
    """Creates a system message in a thread."""
    thread_messages = await openai_client.beta.threads.messages.create(
        thread_id, {"role": "system", "content": message_content}
    )
    print("System Message sent to Assistant with thread:", thread_id)
    return thread_messages


async def get_all_messages(thread_id, openai_client):
    """Gets all messages from a thread."""
    return await openai_client.beta.threads.messages.list(thread_id)


async def get_most_recent_message(thread_id, openai_client):
    """Gets the most recent message from a thread."""
    thread_messages = await openai_client.beta.threads.messages.list(thread_id)
    return thread_messages.data[0].content[0].text.value if thread_messages.data else None


async def run_assistant_on_thread(assistant_id, thread_id, openai_client):
    """Runs an assistant on a given thread."""
    return await openai_client.beta.threads.runs.create(
        thread_id, {"assistant_id": assistant_id}
    )
