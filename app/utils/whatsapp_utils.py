import logging
from flask import current_app, jsonify
import json
import requests

from app.services.openai_service import generate_response
from app.utils.progress_tracker import log_progress  #custom logging

import re


def log_http_response(response):
    logging.info(f"Status: {response.status_code}")
    logging.info(f"Content-type: {response.headers.get('content-type')}")
    logging.info(f"Body: {response.text}")


def get_text_message_input(recipient, text):
    return json.dumps({
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient,
        "type": "text",
        "text": {
            "preview_url": False,
            "body": text
        },
    })


def download_media(media_id):
    url = f"https://graph.facebook.com/v17.0/{media_id}"
    headers = {'Authorization': f'Bearer {current_app.config["ACCESS_TOKEN"]}'}

    # Get media URL
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        logging.error(f"Failed to get media URL: {response.text}")
        return None

    media_url = response.json().get('url')

    # Download media
    response = requests.get(media_url, headers=headers)
    if response.status_code != 200:
        logging.error(f"Failed to download media: {response.text}")
        return None

    return response.content


def send_message(data):
    headers = {
        "Content-type": "application/json",
        "Authorization": f"Bearer {current_app.config['ACCESS_TOKEN']}",
    }

    url = f"https://graph.facebook.com/{current_app.config['VERSION']}/{current_app.config['PHONE_NUMBER_ID']}/messages"

    try:
        response = requests.post(
            url, data=data, headers=headers,
            timeout=10)  # 10 seconds timeout as an example
        response.raise_for_status(
        )  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.Timeout:
        logging.error("Timeout occurred while sending message")
        return jsonify({
            "status": "error",
            "message": "Request timed out"
        }), 408
    except (requests.RequestException
            ) as e:  # This will catch any general request exception
        logging.error(f"Request failed due to: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to send message"
        }), 500
    else:
        # Process the response as normal
        log_http_response(response)
        log_progress(f"Curie Responded to the query successfully.")
        return response


def process_text_for_whatsapp(text):
    # Remove brackets
    pattern = r"\【.*?\】"
    # Substitute the pattern with an empty string
    text = re.sub(pattern, "", text).strip()

    # Pattern to find double asterisks including the word(s) in between
    pattern = r"\*\*(.*?)\*\*"

    # Replacement pattern with single asterisks
    replacement = r"*\1*"

    # Substitute occurrences of the pattern with the replacement
    whatsapp_style_text = re.sub(pattern, replacement, text)

    return whatsapp_style_text


def process_whatsapp_message(body):
    wa_id = body["entry"][0]["changes"][0]["value"]["contacts"][0]["wa_id"]
    name = body["entry"][0]["changes"][0]["value"]["contacts"][0]["profile"][
        "name"]

    message = body["entry"][0]["changes"][0]["value"]["messages"][0]
    message_type = message.get("type", "")

    # Extract message content and media URL based on type
    message_content = ""
    media_url = None

    if message_type == "text":
        message_content = message["text"]["body"]
        media_content = None
    elif message_type == "image":
        caption = message['image'].get('caption', 'No caption')
        media_id = message['image'].get('id')
        media_content = download_media(media_id) if media_id else None
        message_content = caption
    elif message_type == "document":
        filename = message['document'].get('filename', 'No filename')
        media_id = message['document'].get('id')
        media_content = download_media(media_id) if media_id else None
        message_content = filename
    elif message_type == "video":
        caption = message['video'].get('caption', 'No caption')
        media_id = message['video'].get('id')
        media_content = download_media(media_id) if media_id else None
        message_content = caption
    elif message_type == "audio":
        media_id = message['audio'].get('id')
        media_content = download_media(media_id) if media_id else None
        message_content = "Audio message"
    else:
        message_content = "Unsupported message type received"

    # OpenAI Integration
    log_progress(f"Processing message from {name}: {message_content}")
    response = generate_response(message_content, wa_id, name, message_type,
                                 media_content)
    response = process_text_for_whatsapp(response)

    data = get_text_message_input(wa_id,
                                  response)
    send_message(data)


def is_valid_whatsapp_message(body):
    """
    Check if the incoming webhook event has a valid WhatsApp message structure.
    Supports text, image, document, video, and audio messages.
    """
    try:
        return (
            body.get("object") and body.get("entry")
            and body["entry"][0].get("changes")
            and body["entry"][0]["changes"][0].get("value")
            and body["entry"][0]["changes"][0]["value"].get("messages")
            and body["entry"][0]["changes"][0]["value"]["messages"][0] and
            body["entry"][0]["changes"][0]["value"]["messages"][0].get("type")
            in ["text", "image", "document", "video", "audio"])
    except (KeyError, IndexError):
        return False
