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

# LangGraph imports
from langgraph.graph import START, StateGraph
from langchain_core.tools import tool


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
        
        # Initialize ChromaDB vector store (in-memory, not persisted)
        self.vector_store = Chroma(
            collection_name="rag_documents",
            embedding_function=self.embeddings
        )
        
        print("Components initialized successfully!")
        
    
    def clean_text(self, text):
        """
        Clean extracted text from PDFs to improve chunking and retrieval quality.
        Handles common OCR and PDF extraction artifacts.
        """
        import re
        
        if not text or not text.strip():
            return ""
        
        # Step 1: Fix excessive whitespace patterns common in PDF extraction
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double newline
        text = re.sub(r'\n\s+', ' ', text)            # Newline + spaces to single space
        text = re.sub(r'\s+', ' ', text)              # Multiple spaces to single space
        
        # Step 2: Fix broken words (common in PDF extraction)
        # Handle cases like "compu\nter" -> "computer"
        text = re.sub(r'([a-z])\n([a-z])', r'\1\2', text)
        
        # Step 3: Preserve paragraph structure while cleaning
        # Convert single newlines between sentences to spaces
        text = re.sub(r'(?<=[.!?])\s*\n\s*(?=[A-Z])', ' ', text)
        
        # Step 4: Clean up common PDF artifacts
        text = re.sub(r'[^\w\s.,;:!?()\-\'"]+', ' ', text)  # Remove special chars except punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)        # Fix spaces before punctuation
        
        # Step 5: Ensure proper sentence spacing
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        # Step 6: Final cleanup
        text = text.strip()
        
        return text

    def structure_aware_chunking(self, text):
        """
        Academic document-aware chunking that respects document structure.
        """
        import re
        
        # Identify potential section headers and important breaks
        sections = []
        current_section = ""
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                current_section += "\n"
                continue
                
            # Check if line is likely a section header
            is_header = (
                len(line) < 100 and  # Headers are usually short
                (line.isupper() or  # ALL CAPS
                (line[0].isupper() and sum(1 for c in line if c.isupper()) / len(line) > 0.3) or  # Title Case
                re.match(r'^[0-9]+\.', line) or  # Numbered sections
                re.match(r'^[A-Z][a-z]+ [A-Z]', line))  # Title Case patterns
            )
            
            if is_header and len(current_section) > 200:  # Start new section if current is substantial
                sections.append(current_section.strip())
                current_section = line + "\n"
            else:
                current_section += line + "\n"
        
        # Add the last section
        if current_section.strip():
            sections.append(current_section.strip())
        
        return sections if sections else [text]  # Fallback to original text if no structure detected

    def create_smart_chunks(self, sections, base_chunk_size=1000, overlap_size=200):
        """
        Create overlapping chunks with smart boundary detection.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        # Smart separators in priority order
        separators = [
            "\n\n",    # Paragraph breaks (highest priority)
            "\n",      # Line breaks
            ". ",      # Sentence endings
            "! ",      # Exclamations  
            "? ",      # Questions
            "; ",      # Semicolons
            ", ",      # Commas
            " ",       # Spaces
            ""         # Characters (fallback)
        ]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=base_chunk_size,
            chunk_overlap=overlap_size,
            add_start_index=True,
            separators=separators,
            length_function=len,
            is_separator_regex=False
        )
        
        all_chunks = []
        
        for i, section in enumerate(sections):
            if len(section) <= base_chunk_size:
                # Small sections: keep as single chunk
                all_chunks.append({
                    'content': section,
                    'section_id': i,
                    'chunk_type': 'complete_section'
                })
            else:
                # Large sections: apply recursive splitting
                section_chunks = text_splitter.split_text(section)
                for j, chunk in enumerate(section_chunks):
                    all_chunks.append({
                        'content': chunk,
                        'section_id': i,
                        'chunk_id': j,
                        'chunk_type': 'split_section'
                    })
        
        return all_chunks

    def load_and_process_documents(self, docs):
        """
        Enhanced document processing with advanced text preprocessing and smart chunking.
        Processes uploaded documents (PDF, TXT, DOCX) into high-quality chunks
        and adds them to the Chroma vector store.
        """
        if not docs:
            raise ValueError("No documents provided for processing.")

        print(f"📄 Loaded {len(docs)} document(s)")
        
        # Step 1: Text Preprocessing
        print("🧹 Cleaning and preprocessing documents...")
        cleaned_docs = []
        total_original_chars = 0
        total_cleaned_chars = 0
        
        for doc in docs:
            original_content = doc.page_content
            total_original_chars += len(original_content)
            
            # Clean the text content
            cleaned_content = self.clean_text(original_content)
            total_cleaned_chars += len(cleaned_content)
            
            # Create new document with cleaned content
            from langchain_core.documents import Document
            cleaned_doc = Document(
                page_content=cleaned_content,
                metadata={**doc.metadata, 'preprocessing': 'enhanced_cleaning'}
            )
            cleaned_docs.append(cleaned_doc)
        
        print(f"📊 Text cleaning stats:")
        print(f"   • Original: {total_original_chars:,} characters")
        print(f"   • Cleaned: {total_cleaned_chars:,} characters") 
        print(f"   • Reduction: {((total_original_chars - total_cleaned_chars) / total_original_chars * 100):.1f}%")
        
        # Step 2: Structure-Aware Processing
        print("🏗️ Applying structure-aware chunking...")
        all_processed_chunks = []
        
        for doc_idx, doc in enumerate(cleaned_docs):
            # Detect document structure
            sections = self.structure_aware_chunking(doc.page_content)
            print(f"   • Document {doc_idx + 1}: Found {len(sections)} sections")
            
            # Create smart chunks
            chunk_data = self.create_smart_chunks(sections)
            
            # Convert to Document objects with enhanced metadata
            for chunk_info in chunk_data:
                chunk_doc = Document(
                    page_content=chunk_info['content'],
                    metadata={
                        **doc.metadata,
                        'doc_index': doc_idx,
                        'section_id': chunk_info.get('section_id'),
                        'chunk_id': chunk_info.get('chunk_id'),
                        'chunk_type': chunk_info['chunk_type'],
                        'processing_version': 'enhanced_v1.0'
                    }
                )
                all_processed_chunks.append(chunk_doc)
        
        print(f"📝 Created {len(all_processed_chunks)} intelligent chunks")
        
        # Step 3: Enhanced Metadata Assignment
        print("🏷️ Adding positional metadata...")
        total_chunks = len(all_processed_chunks)
        
        for i, chunk in enumerate(all_processed_chunks):
            # Add positional metadata
            if i < total_chunks // 3:
                chunk.metadata["section"] = "beginning"
            elif i < 2 * total_chunks // 3:
                chunk.metadata["section"] = "middle"  
            else:
                chunk.metadata["section"] = "end"
            
            # Add chunk quality metrics
            chunk.metadata["chunk_length"] = len(chunk.page_content)
            chunk.metadata["word_count"] = len(chunk.page_content.split())
            
            # Add content type hints
            content_lower = chunk.page_content.lower()
            if any(keyword in content_lower for keyword in ['figure', 'table', 'chart', 'graph']):
                chunk.metadata["content_type"] = "visual_reference"
            elif any(keyword in content_lower for keyword in ['conclusion', 'summary', 'abstract']):
                chunk.metadata["content_type"] = "summary"
            elif any(keyword in content_lower for keyword in ['introduction', 'background']):
                chunk.metadata["content_type"] = "introductory"
            else:
                chunk.metadata["content_type"] = "main_content"
        
        # Step 4: Add to Vector Store
        print("🗃️ Indexing chunks in vector store...")
        try:
            document_ids = self.vector_store.add_documents(all_processed_chunks)
            print(f"✅ Successfully indexed {len(document_ids)} enhanced document chunks")
            
            # Print quality statistics
            avg_chunk_length = sum(len(chunk.page_content) for chunk in all_processed_chunks) / len(all_processed_chunks)
            print(f"📈 Quality metrics:")
            print(f"   • Average chunk length: {avg_chunk_length:.0f} characters")
            print(f"   • Chunks with visual references: {sum(1 for chunk in all_processed_chunks if chunk.metadata.get('content_type') == 'visual_reference')}")
            print(f"   • Summary chunks: {sum(1 for chunk in all_processed_chunks if chunk.metadata.get('content_type') == 'summary')}")
            
        except Exception as e:
            print(f"❌ Error indexing documents: {str(e)}")
            raise
        
        print("Final Chunks: ", all_processed_chunks)
        return all_processed_chunks
    
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
        @tool(response_format="content_and_artifact")
        def retrieve(state: State):
            retrieved_docs = self.vector_store.similarity_search(state["question"], k=2)
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