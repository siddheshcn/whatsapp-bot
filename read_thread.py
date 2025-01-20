
from openai import OpenAI
import shelve
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with shelve.open('threads_db') as db:
    for key, value in db.items():
        print(f"WhatsApp ID: {key}")
        print(f"Thread ID: {value}")

        # Retrieve and display messages for this thread
        try:
            messages = client.beta.threads.messages.list(thread_id=value)
            print("\nMessages in this thread:")
            for msg in messages.data:
                print(f"Role: {msg.role}")
                print(f"Content: {msg.content[0].text.value}")
                print("-" * 40)
        except Exception as e:
            print(f"Could not retrieve messages: {e}")
        print("=" * 50)
