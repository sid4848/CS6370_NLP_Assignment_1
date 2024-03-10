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
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""
		# Download the 'punkt' resource
		nltk.download('punkt')
  
		reducedText = []
        # Initialize the Porter Stemmer
		porter_stemmer = PorterStemmer()

		for sentence in text:
			
			# Perform stemming on each token
			stemmed_tokens = [porter_stemmer.stem(token) for token in sentence]
			# Add the stemmed tokens to the result
			reducedText.append(stemmed_tokens)

		return reducedText


