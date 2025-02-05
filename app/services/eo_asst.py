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
from app.utils.progress_tracker import log_progress  #custom logging



class EOAssistant:

    def __init__(self):
        log_progress("Initializing EOAssistant...")
        self.model = self.init_model()
        self.embeddings = self.init_embeddings()
        self.current_dir, self.kb_folder, self.persistent_directory = self.get_paths()
        self.db = self.initialize_vector_store(self.persistent_directory, self.kb_folder, self.embeddings)
        self.chain = self.init_chain()
        log_progress("EOAssistant initialization completed")

    def init_model(self):
        """Initialize the language model"""
        log_progress("Initializing ChatOpenAI model...")
        load_dotenv()
        model = ChatOpenAI(model="gpt-4o-mini")
        log_progress("ChatOpenAI model initialized successfully")
        return model

    def init_embeddings(self):
        """Initialize the embedding model"""
        log_progress("Initializing OpenAI embeddings...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        log_progress("OpenAI embeddings initialized successfully")
        return embeddings

    def get_paths(self):
        """Get all necessary file paths"""
        current_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        kb_folder = os.path.join(current_dir, "data")
        persistent_directory = os.path.join(current_dir, "db", "chroma_db")
        try:
            os.makedirs(persistent_directory, exist_ok=True)
            log_progress(f"Created/verified directory: {persistent_directory}")
        except Exception as e:
            log_progress(f"Error creating directory: {str(e)}")
        return current_dir, kb_folder, persistent_directory

    def load_kb_files(self):
        """Load knowledge base files"""
        kb_files = []
        if os.path.exists(self.kb_folder):
            for file in os.listdir(self.kb_folder):
                if file.endswith('.md'):
                    file_path = os.path.join(self.kb_folder, file)
                    kb_files.append(file_path)
            log_progress(f"Found {len(kb_files)} markdown files in knowledge base")
        else:
            log_progress("Knowledge base folder not found")
        return kb_files

    def process_documents(self, kb_files):
        """Process documents and split into chunks"""
        all_docs = []
        for file_path in kb_files:
            print(f"\nProcessing file: {file_path}")
            loader = TextLoader(file_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=10000,
                                                  chunk_overlap=100)
            docs = text_splitter.split_documents(documents)
            all_docs.extend(docs)
            log_progress(f"Processed {len(docs)} documents")
        return all_docs

    @classmethod
    def initialize_on_deployment(cls):
        """Initialize vector store during deployment"""
        log_progress("Initializing vector store...")
        try:
            # Initialize basic components
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            _, kb_folder, persistent_directory = cls().get_paths()

            # Initialize vector store
            cls.initialize_vector_store(persistent_directory, kb_folder, embeddings)
            log_progress("Vector store initialization completed")
        except Exception as e:
            log_progress(f"Failed to initialize vector store: {str(e)}")
            raise

    @staticmethod
    def initialize_vector_store(persistent_directory, kb_folder, embeddings):
        """Initialize or load the vector store"""
        log_progress(f"Checking vector store in: {persistent_directory}")
        
        try:
            if os.path.exists(persistent_directory):
                log_progress("Loading existing vector store")
                return Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
            
            log_progress(f"Creating new vector store in: {persistent_directory}")
            
            # Load knowledge base files
            kb_files = []
            if os.path.exists(kb_folder):
                for file in os.listdir(kb_folder):
                    if file.endswith('.md'):
                        kb_files.append(os.path.join(kb_folder, file))
            
            if not kb_files:
                raise FileNotFoundError(f"No markdown files found in {kb_folder} directory.")
            
            # Process documents
            all_docs = []
            for file_path in kb_files:
                loader = TextLoader(file_path)
                docs = loader.load()
                text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
                split_docs = text_splitter.split_documents(docs)
                all_docs.extend(split_docs)
            
            # Create and return vector store
            return Chroma.from_documents(
                all_docs,
                embeddings,
                persist_directory=persistent_directory
            )
        else:
            print("Vector store already present.")
            db = Chroma(persist_directory=persistent_directory,
                        embedding_function=embeddings)
        return db

    def get_relevant_chunks(self, query):
        """Retrieve relevant documents based on query"""
        log_progress(f"Retrieving relevant chunks for query: {query[:50]}...")
        
        # Fetch all documents
        retriever = self.db.as_retriever(
            search_kwargs={
                "k": 1000  # Increased k value to ensure we fetch everything
            },
            search_type="similarity"  # Explicitly set search type
        )
        
        # Previous MMR implementation for reference:
        # retriever = self.db.as_retriever(
        #     search_type="mmr",
        #     search_kwargs={
        #         "k": 3,
        #         "fetch_k": 10,
        #         "lambda_mult": 0.7
        #     },
        # )
        print("\nAttempting to retrieve documents...")
        relevant_knowledge = retriever.invoke(query)
        print(f"\nRaw retrieval result type: {type(relevant_knowledge)}")
        print(f"Raw retrieval content: {relevant_knowledge}")
        
        if isinstance(relevant_knowledge, list):
            log_progress(f"Found {len(relevant_knowledge)} relevant documents")
            print(f"\nFound {len(relevant_knowledge)} relevant documents")
            print("\nRetrieved content snippets:")
        else:
            log_progress("Warning: Retrieved content is not a list")
            print("\nWarning: Retrieved content is not in expected format")
        for i, doc in enumerate(relevant_knowledge, 1):
            print(f"\nDocument {i}:")
            print(doc.page_content[:200] + "...")
        for i, doc in enumerate(relevant_knowledge, 1):
            if doc.metadata:
                log_progress(f"Document {i} source: {doc.metadata.get('source','Unknown')}")
        return relevant_knowledge

    def init_chain(self):
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
            ("human",
             "Here's the text from the book: {knowledge}. User query is as follows:{user_problem}."
             ),
        ]
        eodr_template = ChatPromptTemplate.from_messages(eodr_message)
        return eodr_template | self.model | StrOutputParser()

    def generate_response(self, query):
        """Generate response for user query"""
        relevant_knowledge = self.get_relevant_chunks(query)
        return self.chain.invoke({
            "user_problem": query,
            "knowledge": relevant_knowledge
        })


# Create a singleton instance
_assistant = None


def get_assistant():
    global _assistant
    if _assistant is None:
        _assistant = EOAssistant()
    return _assistant


def gen_response(query):
    """Generate response for user query"""
    log_progress(f"Generating response for query: {query[:50]}...")
    assistant = get_assistant()
    response = assistant.generate_response(query)
    log_progress("Response generated successfully")
    return response