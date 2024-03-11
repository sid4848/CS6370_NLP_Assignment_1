from util import *

# Add your import statements here
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

class InflectionReduction:
    def reduce(self, text):
        """
        Stemming/Lemmatization
        
        Parameters
        ----------
        arg1 : list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of
            stemmed/lemmatized tokens representing a sentence
        """
        # Initialize the Porter Stemmer
        porter = PorterStemmer()
        # Perform stemming on each token in each sentence
        return [[porter.stem(token) for token in sentence] for sentence in text]


