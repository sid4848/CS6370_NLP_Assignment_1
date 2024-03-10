from util import *

# Add your import statements here
import re
from nltk.tokenize import sent_tokenize



class SentenceSegmentation():

	def naive(self, text):
		
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		#Fill in code here
		segmentedText = self._top_down_segment(text)
		return sentences

		return segmentedText


		def _top_down_segment(self, text):
			"""
			Recursive function for top-down sentence segmentation

			Parameters
			----------
			text : str
				A string (a bunch of sentences)

			Returns
			-------
			list
				A list of strings where each string is a single sentence
			"""

			# Base case: if the text is empty, return an empty list
			if not text:
				return []

			# Look for the first occurrence of a sentence-ending punctuation (.?!)
			match = re.search(r'(?<=[.?!])\s', text)
			# print(match)
			if match:
				# If found, split the text at the position after the punctuation
				index = match.end()
				sentence = text[:index]
				rest_of_text = text[index:]

				# Recursively process the remaining text
				return [sentence] + self._top_down_segment(rest_of_text)
			else:
				# If no sentence-ending punctuation is found, consider the entire text as one sentence
				return [text]


	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		
		# Use NLTK's pre-trained Punkt Tokenizer
		#Fill in code here
		segmentedText = sent_tokenize(text)
		return segmentedText