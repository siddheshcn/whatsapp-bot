
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import YoutubeLoader
import json
from app.utils.progress_tracker import log_progress #custom logging
load_dotenv()

def generate_langchain_response(prompt_text, template=None):

    # Initialize the LLM
    log_progress("Initiating LLM...")
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

    #Chains
    #detect if the inent is 'youtubelink' or 'generalquery'
    intent_detection_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that identifies the user's intent."),
            ("human", "Here's the user's message: {user_message}. If the message consists of a youtube URL, it means that the user wants to learn about the content of that video. Identify and highlight the intent to either 'youtubelink' if it contains a youtube URL or 'generalquery' if it does not contain a youtube URL. Your response should not contain anything other than these two options.")
        ])
        | llm
        | StrOutputParser()
    )


    
    #if the intent is 'youtubelink', then load the youtube video and extract the text
    message_parsing_chain = (
        ChatPromptTemplate.from_messages([
            ("system", """You are a JSON formatting assistant. You MUST return a valid JSON object and NOTHING else.
            No explanations, no additional text, just the JSON object."""),
            ("human", """Extract information from this message: {user_message}

            Rules:
            1. Return ONLY a JSON object in this EXACT format:
            {{"youtube_url": "URL", "condition": "CONDITION"}}
            2. For YouTube URL: Extract any YouTube URL from the message. If none exists, use null
            3. For condition: Extract any user request or query accompanied with this video. If none exists, use "summarize"
            4. Do not add any explanations or text before or after the JSON""")
        ])
        | llm
        | StrOutputParser()
    )
    

    yt_summarization_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that responds to user queries or requests regarding youtube videos."),
            ("human", "The user asks {yt_conditions}. Youtube Video: {yt_transcript}")
        ])
        | llm
        | StrOutputParser()
    )
    
    #function to load the youtube transcripts
    def load_yt_transcript(yt_url: str) -> str:
        """
        Loads the transcript for a YouTube video given its URL.
        """
        try:
            loader = YoutubeLoader.from_youtube_url(yt_url, add_video_info=False)
            result = loader.load()
            if not result:
                log_progress("No transcript found for the given YouTube URL.")
                return "No transcript found for this video."
            return result[0].page_content
        except Exception as e:
            log_progress(f"Error loading YouTube transcript: {e}")
            return f"Failed to load transcript: {str(e)}"

    
    
    #function to process user message (called on the original user prompt)
    def process_user_message(user_message: str) -> str:
        """
        Processes a user's message using LCEL chains and intent detection.
        """
        log_progress(f"Processing user message: {user_message}")
        try:
            # Step 1: Detect intent
            intent_result = intent_detection_chain.invoke({"user_message": user_message})
            intent = intent_result.strip().lower()
            log_progress(f"Intent detected: {intent}")

            if "youtubelink" in intent:
                log_progress("Extracting YouTube URL and conditions (if any)")
                # Step 2: Parse YouTube-related input
                try:
                    parsing_result = message_parsing_chain.invoke({"user_message": user_message})

                    # Clean the response by removing markdown code block syntax
                    parsing_result = parsing_result.replace("```json", "").replace("```", "").strip()

                    # Parse the JSON
                    parsing_result = json.loads(parsing_result)

                    yt_url = parsing_result.get("youtube_url")
                    if not yt_url:
                        log_progress("No YouTube URL found in the message.")
                        return "No YouTube URL found in the message."

                    yt_conditions = parsing_result.get("condition", "summarize")

                    # Step 3: Load YouTube transcript and summarize
                    log_progress(f"YouTube link: {yt_url}, User request: {yt_conditions}")
                    log_progress("Loading YouTube transcript and summarizing")
                    yt_transcript = load_yt_transcript(yt_url)
                    summary_result = yt_summarization_chain.invoke({
                        "yt_conditions": yt_conditions,
                        "yt_transcript": yt_transcript
                    })
                    return summary_result

                except json.JSONDecodeError as e:
                    print("Failed to parse JSON:", parsing_result)  # Debug print
                    return f"Error parsing YouTube information: {str(e)}"

            elif "generalquery" in intent:
                general_response = llm.invoke(user_message)
                response_content = general_response.content if hasattr(general_response, 'content') else str(general_response)
                log_progress("General query response: " + response_content)
                return response_content

            else:
                log_progress("No intent detected. Generating error response.")
                return "I'm sorry, I couldn't understand your request. Could you clarify?"

        except Exception as e:
            log_progress("Error processing user message: " + str(e))
            return f"An error occurred: {str(e)}"


    
    response = process_user_message(prompt_text)
    
    log_progress("Response generated: " + response)
    return response.strip()