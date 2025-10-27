# RUNS WITH HUGGING FACE - Free models, no API keys required

"""
Text Summarization Utility (Hugging Face Transformers)
======================================================

What is Text Summarization?
- Condenses long text into shorter, coherent summaries while preserving key information
- Uses sequence-to-sequence transformer models trained for abstractive summarization
- Generates human-like summaries that capture main ideas and essential details

How It Works:
- Takes: text input string and optional max_length parameter (default 150)
- Loads BART-large-CNN model fine-tuned for summarization tasks
- Processes text through encoder-decoder architecture to generate summary
- Returns: concise summary text within specified length constraints

Use Cases:
- Content summarization for articles and documents
- Meeting notes and transcript condensation
- Research paper abstract generation
- News article summarization
- Educational content simplification
- Social media post summarization

Reference: Hugging Face Transformers Summarization
https://huggingface.co/docs/transformers/tasks/summarization
"""

from transformers import pipeline

def summarize_text(text: str, max_length: int = 150) -> str:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    long_text = """
    Artificial intelligence is transforming industries worldwide by enabling machines to learn from data 
    and make intelligent decisions. Machine learning models can now analyze vast amounts of data and make 
    predictions with unprecedented accuracy. Companies are investing heavily in AI research and development 
    to gain competitive advantages. From healthcare to finance, AI applications are revolutionizing how 
    businesses operate and serve customers. AI-powered tools are enhancing productivity, automating routine tasks, 
    and providing deeper insights through data analysis. As AI technology continues to evolve, its impact on
    society will only grow, raising important ethical and governance considerations.   
    """
    summary = summarize_text(long_text, max_length=100)
    print("Summary:")
    print(summary)

    very_long_text = """
    Harry potter is a series of seven fantasy novels written by British author J.K. Rowling. The novels follow the
    life of a young wizard, Harry Potter, and his friends Hermione Granger and Ron Weasley, all of whom are students at
    the Hogwarts School of Witchcraft and Wizardry. The main story arc concerns Harry's struggle against the dark wizard Lord Voldemort, who aims to become immortal, overthrow the wizard governing body known as the Ministry of Magic, 
    and conquer the wizarding world. Throughout the series, Harry discovers his own magical heritage, faces numerous
    challenges, and learns about friendship, bravery, and sacrifice. The series explores themes of good versus evil,
    the power of love, and the importance of choices. The books have been immensely popular worldwide, leading to a successful film
    franchise, merchandise, and a dedicated fan base. The Harry Potter series has had a significant cultural impact, inspiring a generation of readers and contributing to the popularity of fantasy literature.
    There are also several spin-off works, including the "Fantastic Beasts" film series, which expands the wizarding world and explores its history. The series has been praised for its imaginative storytelling, complex characters, and moral lessons, making it a beloved classic in children's and young adult literature.
    Harry Potter movies are also made based on the books which are also very popular.
    The most recent movie in the series is "Fantastic Beasts: The Secrets of Dumbledore," which delves into the backstory of Albus Dumbledore and his connection to Gellert Grindelwald. The Harry Potter universe continues to expand with new stories, theme parks, and a dedicated fan community that celebrates the magic and wonder of the series.
    Overall the highest grossing movie series of all time from Harry Potter series is $9.2 billion and the title of the highest grossing movie from the series is Harry Potter and the Deathly Hallows – Part 2 which grossed $1.342 billion.   
    """
    summary = summarize_text(very_long_text, max_length=100)
    print("Summary:")
    print(summary)