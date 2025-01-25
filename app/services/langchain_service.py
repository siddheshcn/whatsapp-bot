
from langchain_OpenAI import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import YoutubeLoader
import json

load_dotenv()

def generate_langchain_response(prompt_text, template=None):

    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

    #Chains
    intent_detection_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that identifies the user's intent."),
            ("human", "Here's the user's message: {user_message}. Identify the intent to either 'youtubesummary' or 'generalquery'. Your response should not contain anything other than these two options.")
        ])
        | llm
        | StrOutputParser()
    )

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

    def load_yt_transcript(yt_url: str) -> str:
        """
        Loads the transcript for a YouTube video given its URL.
        """
        loader = YoutubeLoader.from_youtube_url(yt_url, add_video_info=False)
        result = loader.load()
        return result[0].page_content

    def process_user_message(user_message: str) -> str:
        """
        Processes a user's message using LCEL chains and intent detection.
        """
        try:
            # Step 1: Detect intent
            intent_result = intent_detection_chain.invoke({"user_message": user_message})
            intent = intent_result.strip().lower()

            if "youtubesummary" in intent:
                # Step 2: Parse YouTube-related input
                try:
                    parsing_result = message_parsing_chain.invoke({"user_message": user_message})

                    # Clean the response by removing markdown code block syntax
                    parsing_result = parsing_result.replace("```json", "").replace("```", "").strip()

                    # Parse the JSON
                    parsing_result = json.loads(parsing_result)

                    yt_url = parsing_result.get("youtube_url")
                    if not yt_url:
                        return "No YouTube URL found in the message."

                    yt_conditions = parsing_result.get("condition", "summarize")

                    # Step 3: Load YouTube transcript and summarize
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
                return general_response.content

            else:
                return "I'm sorry, I couldn't understand your request. Could you clarify?"

        except Exception as e:
            return f"An error occurred: {str(e)}"


    
    response = process_user_message(prompt_text)
    
    return response.strip()