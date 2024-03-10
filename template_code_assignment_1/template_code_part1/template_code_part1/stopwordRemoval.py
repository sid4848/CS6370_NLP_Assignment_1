from util import *

# Add your import statements here
from nltk.corpus import stopwords

class StopwordRemoval:

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
        
		# nltk.download('stopwords')
  
        stopword_removed_text = []

        # Get the list of stopwords from NLTK
        stop_words = set(stopwords.words("english"))

        for sentence in text:
            # Remove stopwords from each sentence
            filtered_tokens = [token for token in sentence if token.lower() not in stop_words]

            # Add the filtered tokens to the result
            stopword_removed_text.append(filtered_tokens)

        return stopword_removed_text





	