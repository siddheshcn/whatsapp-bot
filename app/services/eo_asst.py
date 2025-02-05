"""
EOAssistant - Extraordinary Doctor Assistant
This module implements an AI assistant specialized in medical knowledge from the Extraordinary Doctor book.
It uses LangChain and ChromaDB for document retrieval and OpenAI for response generation.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from app.utils.progress_tracker import log_progress
import os

class EOAssistant:
    """
    A specialized knowledge assistant that processes queries using
    content from the Extraordinary Doctor book chapters.
    """

    def __init__(self):
        """Initialize the assistant with necessary components and configurations."""
        log_progress("Initializing EOAssistant...")
        self.model = self.init_model()
        self.embeddings = self.init_embeddings()
        self.current_dir, self.kb_folder, self.persistent_directory = self.get_paths()
        #self.db = self.initialize_vector_store(self.persistent_directory, self.kb_folder, self.embeddings)
        self.chain = self.init_chain()
        log_progress("EOAssistant initialization completed")

    def init_model(self):
        """Initialize the OpenAI chat model."""
        log_progress("Initializing ChatOpenAI model...")
        load_dotenv()
        model = ChatOpenAI(model="gpt-4o-mini")
        log_progress("ChatOpenAI model initialized successfully")
        return model

    def init_embeddings(self):
        """Initialize the OpenAI embeddings model."""
        log_progress("Initializing OpenAI embeddings...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        log_progress("OpenAI embeddings initialized successfully")
        return embeddings

    def get_paths(self):
        """
        Set up necessary file paths for the knowledge base and vector store.
        Returns:
            tuple: (current_directory, knowledge_base_folder, vector_store_directory)
        """
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        kb_folder = os.path.join(current_dir, "data")
        persistent_directory = os.path.join(current_dir, "db", "chroma_db")

        try:
            os.makedirs(persistent_directory, exist_ok=True)
            log_progress(f"Created/verified directory: {persistent_directory}")
        except Exception as e:
            log_progress(f"Error creating directory: {str(e)}")

        return current_dir, kb_folder, persistent_directory

    def load_kb_files(self):
        """
        Load knowledge base files from the data directory.
        Returns:
            list: Paths to all markdown files in the knowledge base
        """
        kb_files = []
        if os.path.exists(self.kb_folder):
            kb_files = [
                os.path.join(self.kb_folder, f)
                for f in os.listdir(self.kb_folder)
                if f.endswith('.md')
            ]
            log_progress(f"Found {len(kb_files)} markdown files in knowledge base")
        else:
            log_progress("Knowledge base folder not found")
        return kb_files

    def process_documents(self, kb_files):
        """
        Process and split documents into chunks for vectorization.
        Args:
            kb_files (list): List of file paths to process
        Returns:
            list: Processed document chunks
        """
        all_docs = []
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separator="\n\n"  # Split on paragraph breaks
        )
        for file_path in kb_files:
            loader = TextLoader(file_path)
            documents = loader.load()
            docs = text_splitter.split_documents(documents)
            all_docs.extend(docs)
            log_progress(f"Processed {len(docs)} chunks from {file_path}")

        return all_docs

    @classmethod
    def initialize_on_deployment(cls):
        """Initialize vector store during deployment."""
        log_progress("Checking vector store...")
        try:
            instance = cls()  # Create an instance first
            
            if check_chroma_db_validity(instance.persistent_directory):
                log_progress("Found valid existing vector store")
                instance.db = Chroma(
                    persist_directory=instance.persistent_directory,
                    embedding_function=instance.embeddings
                )
            else:
                log_progress("No valid vector store found, initializing new one...")
                instance.db = instance.initialize_vector_store(
                    instance.persistent_directory,
                    instance.kb_folder,
                    instance.embeddings
                )
                
            log_progress("Vector store initialization completed")
            print(f"Vector store initialized successfully in {instance.persistent_directory}")
            return instance
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            log_progress(f"Failed to initialize vector store: {str(e)}")
            raise

    def initialize_vector_store(self, persistent_directory, kb_folder, embeddings):
        """
        Initialize or load the vector store for document retrieval.
        Args:
            persistent_directory (str): Directory for storing vector database
            kb_folder (str): Knowledge base folder path
            embeddings: Embedding model instance
        Returns:
            Chroma: Vector store instance
        """
        print(f"\n=== Vector Store Initialization ===")
        print(f"Persistent directory: {persistent_directory}")
        print(f"KB folder: {kb_folder}")

        # Load and process knowledge base files
        print("\nScanning for markdown files...")
        kb_files = [
            os.path.join(kb_folder, f)
            for f in os.listdir(kb_folder)
            if f.endswith('.md')
        ]
        print(f"Found {len(kb_files)} markdown files: {kb_files}")

        if not kb_files:
            print(f"No markdown files found in {kb_folder}")
            raise FileNotFoundError(f"No markdown files found in {kb_folder}")

        # Process documents
        print("\nProcessing documents...")
        all_docs = []
        text_splitter = CharacterTextSplitter(
            chunk_size=30000,
            chunk_overlap=200,
            length_function=len,
            separator="\n\n"  # Split on paragraph breaks
        )

        for file_path in kb_files:
            print(f"\nProcessing file: {file_path}")
            loader = TextLoader(file_path)
            documents = loader.load()
            print(f"Loaded document length: {len(documents[0].page_content)} characters")
            docs = text_splitter.split_documents(documents)
            print(f"Split into {len(docs)} chunks")
            all_docs.extend(docs)

        print(f"\nTotal chunks across all documents: {len(all_docs)}")

        # Initialize vector store
        print("\nInitializing vector store...")
        # Remove existing directory to force fresh creation
        if os.path.exists(persistent_directory):
            import shutil
            shutil.rmtree(persistent_directory)
            print("Removed existing vector store")

        try:
            os.makedirs(persistent_directory, exist_ok=True)
            log_progress(f"Created/verified directory: {persistent_directory}")
        except Exception as e:
            log_progress(f"Error creating directory: {str(e)}")

        print("Creating new vector store...")
        vector_store = Chroma.from_documents(
            documents=all_docs,
            embedding=embeddings,
            persist_directory=persistent_directory
        )
        print("Vector store created and populated with documents")

        # Verify vector store contents
        collection = vector_store._collection
        print(f"\nVector store statistics:")
        print(f"Collection name: {collection.name}")
        print(f"Number of documents: {collection.count()}")

        print(f"\nVector store initialized successfully in {persistent_directory}")
        return vector_store

    def get_relevant_chunks(self, query):
        """
        Retrieve relevant document chunks based on the query.
        Args:
            query (str): User's question or problem
        Returns:
            list: Relevant document chunks
        """
        log_progress(f"Retrieving relevant chunks for query: {query[:50]}...")

        retriever = self.db.as_retriever(
            search_kwargs={"k": 1000},
            search_type="similarity"
        )

        relevant_knowledge = retriever.invoke(query)

        if isinstance(relevant_knowledge, list):
            log_progress(f"Found {len(relevant_knowledge)} relevant documents")
            for i, doc in enumerate(relevant_knowledge, 1):
                if doc.metadata:
                    log_progress(f"Document {i} source: {doc.metadata.get('source','Unknown')}")

        return relevant_knowledge

    def init_chain(self):
        """
        Initialize the LangChain processing chain with system prompts.
        Returns:
            Chain: Configured processing chain
        """
        system_prompt = """
You embody the knowledge and essence of chapters 5 and 7 of the book Extraordinary Doctor, allowing users to interact with the content as if they were engaging with the book itself. Your responses should be natural and informative, staying true to the text while maintaining a conversational and approachable tone.

Your Role & Behavior:
    •   Answer user queries based on the content of chapters 5 and 7 only.
    •   When relevant, refer to specific sections or chapter topics that contain useful insights.
    •   Mention relevant chapter numbers and section titles subtly while keeping responses natural.
    •   Seamlessly integrate information without explicitly stating database retrieval.
    •   Maintain a balanced tone—do not overact or excessively personify the book.
"""
        messages = [
            ("system", system_prompt),
            ("human", "Here's the text from the book: {knowledge}. User query is as follows:{user_problem}."),
        ]

        template = ChatPromptTemplate.from_messages(messages)
        return template | self.model | StrOutputParser()

    def generate_response(self, query):
        """
        Generate a response for the user's query.
        Args:
            query (str): User's question or problem
        Returns:
            str: Generated response
        """
        relevant_knowledge = self.get_relevant_chunks(query)
        return self.chain.invoke({
            "user_problem": query,
            "knowledge": relevant_knowledge
        })

# Singleton instance
_assistant = None

def get_assistant():
    """Get or create the singleton EOAssistant instance."""
    global _assistant
    if _assistant is None:
        _assistant = EOAssistant()
    return _assistant

def gen_response(query):
    """
    Generate a response using the singleton assistant.
    Args:
        query (str): User's question or problem
    Returns:
        str: Generated response
    """
    log_progress(f"Generating response for query: {query[:50]}...")
    assistant = get_assistant()
    response = assistant.generate_response(query)
    log_progress("Response generated successfully")
    return response

def check_chroma_db_validity(db_path):
    print(f"\nChecking ChromaDB validity in: {db_path}")
    
    # Check if sqlite file exists
    sqlite_path = os.path.join(db_path, 'chroma.sqlite3')
    if not os.path.exists(sqlite_path):
        print(f"SQLite file not found at: {sqlite_path}")
        return False

    # Find the UUID directory (there should be exactly one)
    subdirs = [d for d in os.listdir(db_path) 
              if os.path.isdir(os.path.join(db_path, d))]
    print(f"Found directories: {subdirs}")
    
    if len(subdirs) != 1:
        print(f"Expected exactly one UUID directory, found {len(subdirs)}")
        return False

    uuid_dir = subdirs[0]
    required_files = ['data_level0.bin', 'header.bin', 'length.bin', 'link_lists.bin']
    print(f"Checking required files in UUID directory: {uuid_dir}")
    
    # Check each required file
    for f in required_files:
        file_path = os.path.join(db_path, uuid_dir, f)
        exists = os.path.exists(file_path)
        print(f"- {f}: {'Found' if exists else 'Missing'}")
        if not exists:
            return False
            
    print("All ChromaDB files verified successfully")
    return True