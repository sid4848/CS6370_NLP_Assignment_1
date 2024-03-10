import numpy as np
from nltk import bigrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

class SpellCheck:
    def __init__(self, text_data):
        self.vocabulary_vectors = {}
        self.load_and_process_data(text_data)
    
    def load_and_process_data(self, text_data):
        # Process the provided list of lists of sentences to form the corpus
        corpus = ' '.join([' '.join(sentence) for paragraph in text_data for sentence in paragraph])
        
        # Tokenize and clean the corpus
        tokens = word_tokenize(corpus.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in stopwords.words('english')]
        
        # Construct the vocabulary and represent tokens as vectors
        unique_tokens = set(tokens)
        self.vocabulary_vectors = {token: self.token_to_vector(token) for token in unique_tokens}
        
    def token_to_vector(self, token):
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        bigram_space = [a + b for a in alphabet for b in alphabet]
        bigram_index = {bigram: idx for idx, bigram in enumerate(bigram_space)}
        
        token_bigrams = [''.join(bg) for bg in bigrams(token)]
        vector = np.zeros(len(bigram_space))
        for bg in token_bigrams:
            if bg in bigram_index:
                vector[bigram_index[bg]] += 1
        return vector

    def find_corrections(self, typo):
        typo_vector = self.token_to_vector(typo)
        similarities = []
        for token, vector in self.vocabulary_vectors.items():
            sim = cosine_similarity([typo_vector], [vector])[0][0]
            similarities.append((token, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [token for token, sim in similarities[:5]]
