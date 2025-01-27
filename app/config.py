
import sys
import os
import logging
from dotenv import load_dotenv

def load_configurations(app):
    # Load from .env file for local development
    if not os.getenv("REPLIT_DEPLOYMENT"):
        load_dotenv()
        print("Loading configurations from .env file")
    else:
        print("Loading configurations from Replit Secrets")
    
    # Configuration mapping with fallbacks
    config_vars = {
        "ACCESS_TOKEN": os.environ.get("ACCESS_TOKEN"),
        "APP_ID": os.environ.get("APP_ID"),
        "APP_SECRET": os.environ.get("APP_SECRET"),
        "RECIPIENT_WAID": os.environ.get("RECIPIENT_WAID"),
        "VERSION": os.environ.get("VERSION"),
        "PHONE_NUMBER_ID": os.environ.get("PHONE_NUMBER_ID"),
        "VERIFY_TOKEN": os.environ.get("VERIFY_TOKEN"),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "OPENAI_ASSISTANT_ID": os.environ.get("OPENAI_ASSISTANT_ID")
    }
    
    # Update app config
    for key, value in config_vars.items():
        app.config[key] = value
        
    # Validate required configurations
    missing_configs = [key for key, value in config_vars.items() if not value]
    if missing_configs:
        raise ValueError(f"Missing required configurations: {', '.join(missing_configs)}")

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
