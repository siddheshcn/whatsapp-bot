
from app import create_app
from app.services.openai_service import generate_response
from app.services.eo_asst import EOAssistant
import os
from dotenv import load_dotenv

load_dotenv()
print("Loading configurations from .env file")

# Create Flask app context
app = create_app()
with app.app_context():
    # Initialize EOAssistant first
    assistant = EOAssistant()
    
    # Test the generate_response function
    response = generate_response(
        message_content="Hello, how are you?",
        wa_id="test_user_123",
        name="Test User",
        use_langchain=True  # Set to True to use langchain response
    )
    print("Response:", response)
