"""
Wikipedia Vector Database Module
==================================
This module handles loading documents from Wikipedia, chunking them into smaller pieces,
creating embeddings, and storing them in a ChromaDB vector database for semantic search.
"""

import os
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "mxbai-embed-large:335m"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DB_LOCATION = "./wikipedia_chroma_db"

def load_from_wikipedia(query, num_docs=5):
    """
    Load top documents from Wikipedia for a given query.
    
    Args:
        query (str): Search query for Wikipedia
        num_docs (int): Number of top documents to retrieve (default: 5)
        
    Returns:
        list: List of Document objects from Wikipedia
    """
    loader = WikipediaLoader(query=query, load_max_docs=num_docs)
    return loader.load()


def chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Split documents into smaller chunks for better retrieval and embedding.
    
    Args:
        documents (list): List of Document objects
        chunk_size (int): Maximum characters per chunk
        chunk_overlap (int): Characters to overlap between chunks
        
    Returns:
        list: List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)


def create_vector_db_from_wikipedia(query, num_docs=5, persist_directory=DB_LOCATION, force_rebuild=False):
    """
    Create or load a vector database from Wikipedia documents.
    
    Args:
        query (str): Search query for Wikipedia
        num_docs (int): Number of top documents to retrieve
        persist_directory (str): Where to save the vector database
        force_rebuild (bool): If True, rebuild entire DB from scratch
        
    Returns:
        Chroma: Vector database instance
    """
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    if os.path.exists(persist_directory) and not force_rebuild:
        print(f"Loading existing vector DB from {persist_directory}")
        vectordb = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        existing_count = vectordb._collection.count()
        print(f"✓ Loaded existing DB with {existing_count} chunks")
        return vectordb
    
    print(f"Step 1: Searching Wikipedia for '{query}'...")
    documents = load_from_wikipedia(query, num_docs=num_docs)
    print(f"✓ Loaded {len(documents)} documents from Wikipedia")
    
    print(f"\nStep 2: Chunking documents (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunked_docs = chunk_documents(documents)
    print(f"✓ Created {len(chunked_docs)} chunks from {len(documents)} documents")
    
    print(f"\nStep 3: Initializing embedding model ({EMBEDDING_MODEL})...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    print("✓ Embedding model initialized")
    
    print(f"\nStep 4: Creating vector database and generating embeddings...")
    vectordb = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"✓ Vector DB created and saved to {persist_directory}")
    print(f"  Total chunks embedded: {len(chunked_docs)}")
    
    return vectordb

def add_documents_to_vector_db(query, num_docs=5, persist_directory=DB_LOCATION):
        """
        Add new documents to existing vector database or create one if it doesn't exist.
        
        Args:
            query (str): Search query for Wikipedia
            num_docs (int): Number of top documents to retrieve
            persist_directory (str): Where to save/load the vector database
            
        Returns:
            Chroma: Updated vector database instance
        """
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
        if not os.path.exists(persist_directory):
            print(f"Vector DB not found at {persist_directory}. Creating new DB...")
            return create_vector_db_from_wikipedia(query, num_docs, persist_directory)
        
        print(f"Loading existing vector DB from {persist_directory}")
        vectordb = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        existing_count = vectordb._collection.count()
        print(f"✓ Loaded existing DB with {existing_count} chunks")
        
        print(f"\nFetching new documents for query: '{query}'...")
        new_documents = load_from_wikipedia(query, num_docs=num_docs)
        print(f"✓ Loaded {len(new_documents)} new documents")
        
        print(f"\nChunking new documents...")
        new_chunked_docs = chunk_documents(new_documents)
        print(f"✓ Created {len(new_chunked_docs)} new chunks")
        
        print(f"\nAdding chunks to vector database...")
        vectordb.add_documents(new_chunked_docs)
        print(f"✓ Successfully added {len(new_chunked_docs)} chunks to DB")
        print(f"  Total chunks in DB: {vectordb._collection.count()}")
        
        return vectordb