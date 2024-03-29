from util import *

# Add your import statements here
from nltk.tokenize import word_tokenize
from nltk.tokenize import TreebankWordTokenizer


class Tokenization:
    def naive(self, text):
        """
        Tokenization using a Naive Approach.

        Parameters
        ----------
        arg1 : list
            A list of strings where each string is a single sentence.

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens.
        """
        # The comprehension method provides a compact way to iterate over sentences
        return [sentence.split() for sentence in text]

    def pennTreeBank(self, text):
        """
        Tokenization using the Penn Tree Bank Tokenizer.

        Parameters
        ----------
        arg1 : list
            A list of strings where each string is a single sentence.

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens.
        """
        # Initializing the tokenizer outside the loop is more efficient
        tokenizer = TreebankWordTokenizer()
        # List comprehension for a more compact and Pythonic approach
        return [tokenizer.tokenize(sentence) for sentence in text]