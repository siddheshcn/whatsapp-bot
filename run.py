import logging
import os
from app import create_app
from app.services.eo_asst import EOAssistant

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = create_app()

if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 8080))
        logger.info(f"Starting Flask app on port {port}")
        EOAssistant.initialize_on_deployment() # Initialize vector store before starting the server
        app.run(host="0.0.0.0", port=port)
    except Exception as e:
        logger.error(f"Failed to start Flask app: {str(e)}")