from util import *
nltk.download('stopwords')

# Add your import statements here
from nltk.corpus import stopwords

class StopwordRemoval:

    def __init__(self):
        # Download the 'stopwords' resource, if not already done so
        # nltk.download('stopwords')
        
        # Initialize the set of English stopwords upon creation of the class instance
        self.stop_words = set(stopwords.words("english"))

    def fromList(self, text):
        """
        Remove stopwords from a list of tokenized sentences.

        Parameters
        ----------
        text : list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence.

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence with stopwords removed.
        """
        
        stopwords_removed_text = [
            [token for token in sentence if token.lower() not in self.stop_words]
            for sentence in text
        ]

        return stopwords_removed_text

	