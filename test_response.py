
from app import create_app
from app.services.openai_service import generate_response
from app.services.eo_asst import EOAssistant
import os
from dotenv import load_dotenv

load_dotenv()

# Create Flask app context
app = create_app()
with app.app_context():
    # Initialize EOAssistant using deployment method
    assistant = EOAssistant.initialize_on_deployment()
    
    # Test the generate_response function
    response = generate_response(
        message_content="My patients come with Google results on their symptoms and it bothers me a lot.",
        wa_id="test_user_123",
        name="Test User",
        use_langchain=True
    )
    print("\nResponse:", response)
