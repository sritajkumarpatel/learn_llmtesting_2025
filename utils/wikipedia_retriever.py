# Import necessary modules for LangChain RAG (Retrieval-Augmented Generation)
from langchain_community.retrievers import WikipediaRetriever  # Retrieves Wikipedia articles


def retrieve_context_from_wiki(topic: str) -> str:
    """
    Retrieve Wikipedia article content for a given topic.
    
    Fetches relevant Wikipedia articles and returns content up to the middle
    article if multiple results exist, otherwise returns the first article.
    
    Args:
        topic: The topic to search for on Wikipedia
        
    Returns:
        str: Formatted Wikipedia article content, or empty string if no results found
    """
    # Initialize Wikipedia retriever
    retriever = WikipediaRetriever()
    
    # Retrieve top article for the given topic
    docs = retriever.invoke(topic, k=1)

    # Format retrieved documents with headers
    context_list = []
    num_articles = 0
    for i, doc in enumerate(docs, 1):
        context_list.append(f"=== Article {i} ===\n{doc.page_content}\n\n")
        num_articles += 1

    # Return middle section if multiple articles, first article if one, or empty string
    if num_articles > 1:
        middle_index = (num_articles // 2) + 1
        return "".join(context_list[:middle_index])
    elif num_articles == 1:
        return context_list[0]
    else:
        return ""
