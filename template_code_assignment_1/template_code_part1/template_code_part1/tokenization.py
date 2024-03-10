from util import *

# Add your import statements here
from nltk.tokenize import word_tokenize
from nltk.tokenize import TreebankWordTokenizer


class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		#Fill in code here

		# Initialize the Penn Treebank Tokenizer
		for sentence in text:
            # Split tokens based on whitespace
			tokens = sentence.split()
			tokenizedText.append(tokens)
		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		#Fill in code here

		# Initialize the Penn Treebank Tokenizer
		treebank_tokenizer = TreebankWordTokenizer()

		for sentence in text:
			# Use the Penn Treebank Tokenizer for word tokenization
			tokens = treebank_tokenizer.tokenize(sentence)
			tokenizedText.append(tokens)

		return tokenizedText