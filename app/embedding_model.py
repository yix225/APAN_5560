import spacy


class EmbeddingModel:
    def __init__(self):
        # Load spaCy model containing pretrained word vectors
        self.nlp = spacy.load("en_core_web_md")

    def get_embedding(self, word: str):
        # Process the input word
        doc = self.nlp(word)

        # Get its vector and convert NumPy array to Python list
        return doc.vector.tolist()