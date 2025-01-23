from openai import OpenAI
import shelve
from dotenv import load_dotenv
import os
import time
import logging

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")
client = OpenAI(api_key=OPENAI_API_KEY)


def upload_file(path):
    # Upload a file with an "assistants" purpose
    file = client.files.create(
        file=open("../../data/airbnb-faq.pdf", "rb"), purpose="assistants"
    )


# def create_assistant(file):
#     """
#     You currently cannot set the temperature for Assistant via the API.
#     """
#     assistant = client.beta.assistants.create(
#         name="WhatsApp AirBnb Assistant",
#         instructions="You're a helpful WhatsApp assistant that can assist guests that are staying in our Paris AirBnb. Use your knowledge base to best respond to customer queries. If you don't know the answer, say simply that you cannot help with question and advice to contact the host directly. Be friendly and funny.",
#         tools=[{"type": "retrieval"}],
#         model="gpt-4-1106-preview",
#         file_ids=[file.id],
#     )
#     return assistant


# Use context manager to ensure the shelf file is closed properly
def check_if_thread_exists(wa_id):
    with shelve.open("threads_db") as threads_shelf:
        return threads_shelf.get(wa_id, None)


def store_thread(wa_id, thread_id):
    with shelve.open("threads_db", writeback=True) as threads_shelf:
        threads_shelf[wa_id] = thread_id


def run_assistant(thread, name):
    # Retrieve the Assistant
    assistant = client.beta.assistants.retrieve(OPENAI_ASSISTANT_ID)

    # Run the assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        # instructions=f"You are having a conversation with {name}",
    )

    # Wait for completion
    # https://platform.openai.com/docs/assistants/how-it-works/runs-and-run-steps#:~:text=under%20failed_at.-,Polling%20for%20updates,-In%20order%20to
    while run.status != "completed":
        # Be nice to the API
        time.sleep(0.5)
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

    # Retrieve the Messages
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    new_message = messages.data[0].content[0].text.value
    logging.info(f"Generated message: {new_message}")
    return new_message


def generate_response(message_content, wa_id, name, message_type="text", media_content=None):
    print("API Key present:", bool(os.getenv('OPENAI_API_KEY')))
    thread_id = check_if_thread_exists(wa_id)
    
    # Upload media file if present
    file_id = None
    supported_image_types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
    
    if media_content and message_type == "image":
        try:
            import magic
            mime_type = magic.from_buffer(media_content, mime=True)
            
            if mime_type in supported_image_types:
                temp_filename = f"temp_media_{wa_id}"
                with open(temp_filename, "wb") as f:
                    f.write(media_content)
                
                with open(temp_filename, "rb") as f:
                    response = client.files.create(file=f, purpose="assistants")
                    file_id = response.id
                    content = f"[Image uploaded successfully. File ID: {file_id}]\n{content}"
                
                os.remove(temp_filename)
            else:
                logging.error(f"Unsupported image type: {mime_type}")
                content = f"[Unsupported image format. Only JPEG, PNG, GIF, and WebP are supported]\n{content}"
        except Exception as e:
            logging.error(f"Failed to upload media to OpenAI: {e}")
    elif media_content and message_type in ["audio", "video"]:
        content = f"[{message_type.capitalize()} messages are not supported yet. Please send your message as text.]\n{content}"
    
    # Prepare content
    content = message_content

    # If a thread doesn't exist, create one and store it
    if thread_id is None:
        logging.info(f"Creating new thread for {name} with wa_id {wa_id}")
        thread = client.beta.threads.create()
        store_thread(wa_id, thread.id)
        thread_id = thread.id

    # Otherwise, retrieve the existing thread
    else:
        logging.info(f"Retrieving existing thread for {name} with wa_id {wa_id}")
        thread = client.beta.threads.retrieve(thread_id)

    # Add message to thread
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content,
    )

    # Run the assistant and get the new message
    new_message = run_assistant(thread, name)

    return new_message
