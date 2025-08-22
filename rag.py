import os
import bs4
from typing import Literal, List
from typing_extensions import Annotated, TypedDict
from dotenv import load_dotenv

# LangChain imports
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# LangGraph imports
from langgraph.graph import START, StateGraph

class RAGPipeline:
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.graph = None
        
    def setup_environment(self):
        """Load environment variables from .env file"""
        load_dotenv()  # Load variables from .env file
        
        # Check if required API keys are loaded
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in environment variables. Please add it to your .env file.")
        
        # LangSmith is optional
        if not os.environ.get("LANGSMITH_TRACING"):
            os.environ["LANGSMITH_TRACING"] = "true"
        
        if not os.environ.get("LANGSMITH_API_KEY"):
            raise ValueError("LANGSMITH_API_KEY not found in environment variables. Please add it to your .env file.")

        print("Environment variables loaded from .env file")
    
    def initialize_components(self):
        """Initialize the LLM, embeddings, and vector store"""
        # Initialize chat model
        self.llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
        
        # Use HuggingFace embeddings (free and local)
        from langchain_community.embeddings import HuggingFaceEmbeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Initialize ChromaDB vector store
        self.vector_store = Chroma(
            collection_name="rag_documents",
            embedding_function=self.embeddings,
            persist_directory="./chroma_db",  # Local storage directory
        )
        
        print("Components initialized successfully!")
    
    def load_and_process_documents(self, docs):
        """
        Process uploaded documents (PDF, TXT, DOCX) into chunks
        and add them to the Chroma vector store.
        """
        if not docs:
            raise ValueError("No documents provided for processing.")

        print(f"Loaded {len(docs)} document(s)")
        print(f"Total characters in first doc: {len(docs[0].page_content)}")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        all_splits = text_splitter.split_documents(docs)
        print(f"Split into {len(all_splits)} chunks")

        # Optional: add section metadata
        total_documents = len(all_splits)
        if total_documents > 0:
            third = total_documents // 3
            for i, document in enumerate(all_splits):
                if i < third:
                    document.metadata["section"] = "beginning"
                elif i < 2 * third:
                    document.metadata["section"] = "middle"
                else:
                    document.metadata["section"] = "end"

        # Add chunks to vector store
        document_ids = self.vector_store.add_documents(all_splits)
        print(f"Indexed {len(document_ids)} document chunks")

        return all_splits
    
    def create_basic_rag_graph(self):
        """Create a basic RAG graph with retrieve and generate steps"""
        
        # Define state
        class State(TypedDict):
            question: str
            context: List[Document]
            answer: str
        
        # Load prompt
        from langchain_core.prompts import PromptTemplate

        template = """You are an assistant for question-answering tasks.
    Use the following documents to answer the question.
    If you don't know the answer, just say that you don't know.

        {context}

        Question: {question}

        Helpful Answer:"""
        prompt = PromptTemplate.from_template(template)
        
        # Define nodes
        def retrieve(state: State):
            retrieved_docs = self.vector_store.similarity_search(state["question"])
            return {"context": retrieved_docs}
        
        def generate(state: State):
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = prompt.invoke({"question": state["question"], "context": docs_content})
            response = self.llm.invoke(messages)
            return {"answer": response.content}
        
        # Build graph
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        self.graph = graph_builder.compile()
        
        print("Basic RAG graph created!")
        return State
    
    
    
    def query(self, question: str, stream_mode=None):
        """Query the RAG system"""
        if self.graph is None:
            raise ValueError("Graph not created. Please create a graph first.")
        
        if stream_mode == "updates":
            print("Streaming updates:")
            for step in self.graph.stream({"question": question}, stream_mode="updates"):
                print(f"{step}\n" + "-" * 50)
        elif stream_mode == "messages":
            print("Streaming tokens:")
            for message, metadata in self.graph.stream({"question": question}, stream_mode="messages"):
                print(message.content, end="|")
            print("\n")
        else:
            result = self.graph.invoke({"question": question})
            return result


# Example usage and demonstration
def main():
    # Create RAG pipeline
    rag = RAGPipeline()
    
    print("=== Setting up RAG Pipeline ===")
    
    # Setup environment
    rag.setup_environment()
    
    # Initialize components
    rag.initialize_components()
    
    # Load and process documents
    print("\n=== Loading and Processing Documents ===")
    rag.load_and_process_documents()
    
    # Create different types of RAG graphs
    print("\n=== Creating RAG Graphs ===")
    
    # Option 1: Basic RAG
    print("\n1. Basic RAG:")
    state_class = rag.create_basic_rag_graph()
    
    # Test basic RAG
    print("\n=== Testing Basic RAG ===")
    question = "What is Task Decomposition?"
    result = rag.query(question)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    
    

if __name__ == "__main__":
    main()