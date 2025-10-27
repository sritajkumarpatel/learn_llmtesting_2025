# RUNS WITH HUGGING FACE - Free models, no API keys required

"""
Named Entity Recognition (NER) Utility (Hugging Face Transformers)
=================================================================

What is Named Entity Recognition?
- Identifies and classifies named entities in text into predefined categories
- Extracts entities like persons, organizations, locations, dates, etc.
- Uses token-level classification to assign labels to individual tokens

How It Works:
- Takes: text input string
- Loads BERT-large model fine-tuned on CoNLL-03 dataset for NER
- Processes text through transformer to predict entity labels for each token
- Groups consecutive tokens belonging to same entity (with grouped_entities=True)
- Returns: list of detected entities with labels, scores, and positions

Entity Types (CoNLL-03 Standard):
- PER: Person names
- ORG: Organizations
- LOC: Locations/Geopolitical entities
- MISC: Miscellaneous entities

Use Cases:
- Information extraction from documents
- Content categorization and tagging
- Question answering systems
- Knowledge graph construction
- Text analytics and business intelligence

Reference: Hugging Face Transformers Token Classification
https://huggingface.co/docs/transformers/tasks/token_classification
"""

from transformers import pipeline

classifier = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)

def classify_text(text: str) -> dict:
    return classifier(text)

if __name__ == "__main__":
    text = "Raman is a SW engineer working at Google in California. He stays at San Francisco, Street 123 and commutes to work by Uber."
    result = classify_text(text)
    print("NER Result:")
    print(result)
