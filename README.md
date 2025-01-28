
# WhatsApp AI Assistant Documentation

## Overview
A WhatsApp bot powered by LangChain and OpenAI that processes messages, handles multimedia content, and provides AI-assisted responses. The bot can understand YouTube links, process general queries, and maintain conversation context.

## Core Components

### 1. Main Services
- **LangChain Service** (`app/services/langchain_service.py`): 
  - Handles AI response generation using LangChain
  - Processes YouTube links and extracts transcripts
  - Manages conversation chains and intent detection

- **OpenAI Service** (`app/services/openai_service.py`):
  - Manages OpenAI API interactions
  - Handles thread management for conversations
  - Processes multimedia content

### 2. Webhook Handler
- **Views** (`app/views.py`):
  - Manages incoming WhatsApp webhook events
  - Handles message verification and security
  - Routes messages to appropriate processors

### 3. Utilities
- **WhatsApp Utils** (`app/utils/whatsapp_utils.py`):
  - Processes incoming messages
  - Handles media downloads
  - Formats responses for WhatsApp

- **Progress Tracker** (`app/utils/progress_tracker.py`):
  - Monitors application progress
  - Logs important events
  - Provides debugging information

### 4. Security
- **Security Decorators** (`app/decorators/security.py`):
  - Validates webhook signatures
  - Ensures secure communication
  - Protects endpoints

## Features
1. Message Processing
   - Text message handling
   - Image processing
   - Document handling
   - Audio/Video message support

2. AI Capabilities
   - Context-aware conversations
   - YouTube content analysis
   - Natural language understanding
   - Multi-purpose response generation

3. System Features
   - Progress tracking
   - Error handling
   - Secure webhook verification
   - Environment configuration

## Setup Requirements
1. Meta Developer Account
2. WhatsApp Business API access
3. OpenAI API key
4. Required Python packages (see requirements.txt)

## Configuration
The application uses environment variables for configuration:
- ACCESS_TOKEN
- APP_ID
- APP_SECRET
- VERIFY_TOKEN
- OPENAI_API_KEY
- OPENAI_ASSISTANT_ID

## Running the Application
1. Set up environment variables
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python run.py`

## Monitoring
- Access `/progress` endpoint for real-time progress monitoring
- Check console logs for detailed debugging information
- Monitor webhook events through Meta's developer dashboard

## Future Enhancements
1. Enhanced multimedia processing
2. Additional AI model integration
3. Advanced conversation management
4. Analytics and reporting features
