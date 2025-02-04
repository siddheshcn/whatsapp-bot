from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_google_firestore import FirestoreChatMessageHistory
from google.cloud import firestore
from google.oauth2 import service_account
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma

def init_model():
    """Initialize the language model"""
    load_dotenv()
    return ChatOpenAI(model="gpt-4o-mini")

def load_chat_db():
    """Initialize and load the Firestore chat database"""
    PROJECT_ID = "langchain-pilot-ff684"
    SESSION_ID = "user_session_chain1"
    COLLECTION_NAME = "chat_history"

    print("Init Firestore chat message history....")
    credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

    if credentials_json:
        credentials_dict = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        client = firestore.Client(project=PROJECT_ID, credentials=credentials)
    else:
        client = firestore.Client(project=PROJECT_ID)

    chat_history = FirestoreChatMessageHistory(
        session_id=SESSION_ID,
        collection=COLLECTION_NAME,
        client=client,
    )
    print("Chat History session initialized")
    print("Current Chat History: ", chat_history.messages)
    return chat_history

def init_embeddings():
    """Initialize the embedding model"""
    return OpenAIEmbeddings(model="text-embedding-3-small")

def get_paths():
    """Get all necessary file paths"""
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    kb_folder = os.path.join(current_dir, "data")
    local_kb_path = os.path.join(kb_folder, "Chapter_5_Seven_Scenarios.md")
    persistent_directory = os.path.join(current_dir, "app", "utils", "db", "chroma_db")
    return current_dir, kb_folder, local_kb_path, persistent_directory

def load_kb_files(kb_folder):
    """Load knowledge base files"""
    kb_files = []
    for file in os.listdir(kb_folder):
        if file.endswith('.md'):
            file_path = os.path.join(kb_folder, file)
            kb_files.append(file_path)
    return kb_files

def process_documents(kb_files):
    """Process documents and split into chunks"""
    all_docs = []
    for file_path in kb_files:
        print(f"\nProcessing file: {file_path}")
        loader = TextLoader(file_path)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        all_docs.extend(docs)

    print("\n---Document Chunk Info---")
    print(f"Total number of doc chunks: {len(all_docs)}")
    print(f"\nSample chunk 0: {all_docs[0].page_content}")
    if len(all_docs) > 1:
        print(f"\nSample chunk 1: {all_docs[1].page_content}")

    return all_docs

def initialize_vector_store(persistent_directory, embeddings):
    """Initialize or load the vector store"""
    if not os.path.exists(persistent_directory):
        print("Persistent Directory does not exist. Creating new vector store...")
        kb_files = load_kb_files(get_paths()[1])

        if not kb_files:
            raise FileNotFoundError(f"No markdown files found in {get_paths()[1]} directory.")

        all_docs = process_documents(kb_files)
        print("\n\n---Creating Embeddings and Vector Store---")
        db = Chroma.from_documents(all_docs, embeddings, persist_directory=persistent_directory)
        print("\n\n---Finished Creating Vector Store---")
    else:
        print("Vector store already present.")
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    return db

def load_embedding_model():
    """Load the embedding model"""
    return OpenAIEmbeddings(model="text-embedding-3-small")

def get_relevant_chunks(query, embeddings, persistent_directory):
    """Retrieve relevant documents based on query"""
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.1},
    )
    relevant_knowledge = retriever.invoke(query)
    print("---Relevant Documents---")
    for i, doc in enumerate(relevant_knowledge, 1):
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source','Unknown')}")
    return relevant_knowledge

def init_chain():
    """Initialize the processing chain"""
    eodr_message = [
        ("system", """
You embody the knowledge and essence of chapters 5 and 7 of the book Extraordinary Doctor, allowing users to interact with the content as if they were engaging with the book itself. Your responses should be natural and informative, staying true to the text while maintaining a conversational and approachable tone.

Your Role & Behavior:
    •   Answer user queries based on the content of chapters 5 and 7 only.
    •   When relevant, refer to specific sections or chapter topics that contain useful insights from the author's experiences, key pointers, or notable discussions.
    •   Mention relevant chapter numbers and section titles subtly to guide the user while keeping the response natural and engaging. (e.g., "In one section, the author discusses...", rather than "According to the knowledge base...")
    •   Seamlessly integrate information without explicitly stating that it is retrieved from a stored database or chunked text.
    •   Maintain a balanced tone—do not overact or excessively personify the book.

Handling User Queries:
    •   If passage provided is relevant: Use it to provide clear, insightful, and helpful answers while referencing applicable sections or topics.
    •   If the query extends beyond available content: Gently inform the user that your insights are limited to the available chapters and suggest related themes from the text if possible.
    •   If multiple relevant sections exist: Prioritize clarity by summarizing key insights and guiding the user toward applicable sections without overwhelming them with unnecessary detail.

Your goal is to provide a smooth, engaging, and contextually rich experience, ensuring that the book's wisdom is accessible in a meaningful way.
     """),
        ("human", "Here's the text from the book: {knowledge}. User query is as follows:{user_problem}."),
    ]
    eodr_template = ChatPromptTemplate.from_messages(eodr_message)
    model = init_model()
    return eodr_template | model | StrOutputParser()

def gen_response(myproblem):
    """Generate response for user query"""
    embeddings = init_embeddings()
    _, _, _, persistent_directory = get_paths()
    relevant_knowledge = get_relevant_chunks(myproblem, embeddings, persistent_directory)
    chain = init_chain()
    return chain.invoke({
        "user_problem": myproblem,
        "knowledge": relevant_knowledge
    })

# Initialize necessary components
model = init_model()
embeddings = init_embeddings()
_, _, _, persistent_directory = get_paths()
db = initialize_vector_store(persistent_directory, embeddings)