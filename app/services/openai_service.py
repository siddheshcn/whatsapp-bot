from openai import OpenAI
import shelve
from dotenv import load_dotenv
import os
import time
import logging
from flask import current_app

# Initialize client as None
client = None

def get_openai_client():
    global client
    if client is None:
        client = OpenAI(api_key=current_app.config["OPENAI_API_KEY"])
    return client

def upload_file(path):
    # Get client within context
    client = get_openai_client()
    file = client.files.create(
        file=open("../../data/airbnb-faq.pdf", "rb"), purpose="assistants"
    )

def check_if_thread_exists(wa_id):
    with shelve.open("threads_db") as threads_shelf:
        return threads_shelf.get(wa_id, None)

def store_thread(wa_id, thread_id):
    with shelve.open("threads_db", writeback=True) as threads_shelf:
        threads_shelf[wa_id] = thread_id

def run_assistant(thread, name):
    client = get_openai_client()
    # Retrieve the Assistant
    assistant = client.beta.assistants.retrieve(current_app.config["OPENAI_ASSISTANT_ID"])

    # Run the assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    while run.status != "completed":
        time.sleep(0.5)
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

    messages = client.beta.threads.messages.list(thread_id=thread.id)
    new_message = messages.data[0].content[0].text.value
    logging.info(f"Generated message: {new_message}")
    return new_message

from .langchain_service import generate_langchain_response

def generate_response(message_content, wa_id, name, message_type="text", media_content=None, use_langchain=True):
    print("API Key present:", bool(os.getenv('OPENAI_API_KEY')))

    if use_langchain:
        return generate_langchain_response(message_content)

    client = get_openai_client()
    thread_id = check_if_thread_exists(wa_id)

    # Upload media file if present
    file_id = None
    supported_image_types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']

    content = message_content
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
            content = f"[Failed to process image: {str(e)}]\n{content}"
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