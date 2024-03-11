import numpy as np
import json
from nltk import bigrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from util import *

class SpellCheck:
    def __init__(self, json_file_path):
        self.vocabulary_vectors = self.load_and_process_data(json_file_path)

    def load_and_process_data(self, json_file_path):
        # Load JSON data
        with open(json_file_path, 'r') as file:
            cranfield_data = json.load(file)

        # Concatenate all bodies to form the corpus
        corpus = ' '.join([doc['body'] for doc in cranfield_data])

        # Tokenize and clean the corpus
        tokens = word_tokenize(corpus.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in stopwords.words('english')]

        # Construct the vocabulary and represent tokens as vectors
        unique_tokens = set(tokens)
        vocabulary_vectors = {token: self.token_to_vector(token) for token in unique_tokens}
        return vocabulary_vectors


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
        return [(token, sim) for token, sim in similarities[:5]]


spell_check = SpellCheck('./cranfield/cran_docs.json')
corrections_for_boundery = spell_check.find_corrections('boundery')
corrections_for_transiant = spell_check.find_corrections('transiant')
corrections_for_aerplain = spell_check.find_corrections('aerplain')

print('Corrections for "boundery":', corrections_for_boundery)
print('Corrections for "transiant":', corrections_for_transiant)
print('Corrections for "aerplain":', corrections_for_aerplain)

# Function to find the closest candidate for each typo
def find_closest_candidate(typo, candidates):
    min_distance = float('inf')
    closest_candidate = None
    for candidate in candidates:
        distance = edit_distance(typo, candidate)
        if distance < min_distance:
            min_distance = distance
            closest_candidate = candidate
    return closest_candidate, min_distance

candidates = {
    'boundery': [correction for correction, score in corrections_for_boundery],
    'transiant': [correction for correction, score in corrections_for_transiant],
    'aerplain': [correction for correction, score in corrections_for_aerplain]
}

# Finding the closest candidate for each typo
closest_candidates = {typo: find_closest_candidate(typo, candidates[typo]) for typo in candidates}

for typo, closest in closest_candidates.items():
    print(f'The closest candidate for "{typo}" is "{closest[0]}" with disatance "{closest[1]}".')