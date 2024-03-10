from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from spellCheck import SpellCheck
from util import *


import argparse
import json
from sys import version_info


# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print ("Unknown python version - input function not safe")


class SearchEngine:

	def __init__(self, args):
		self.args = args
		self.tokenizer = Tokenization()
		self.sentenceSegmenter = SentenceSegmentation()
		self.inflectionReducer = InflectionReduction()
		self.stopwordRemover = StopwordRemoval()
		self.spellChecker = None
  
	def segmentSentences(self, text):
		"""
		Return the required sentence segmenter
		"""
		if self.args.segmenter == "naive":
			return self.sentenceSegmenter.naive(text)
		elif self.args.segmenter == "punkt":
			return self.sentenceSegmenter.punkt(text)


	def tokenize(self, text):
		"""
		Return the required tokenizer
		"""
		if self.args.tokenizer == "naive":
			return self.tokenizer.naive(text)
		elif self.args.tokenizer == "ptb":
			return self.tokenizer.pennTreeBank(text)


	def reduceInflection(self, text):
		"""
		Return the required stemmer/lemmatizer
		"""
		return self.inflectionReducer.reduce(text)

	def removeStopwords(self, text):
		"""
		Return the required stopword remover
		"""
		return self.stopwordRemover.fromList(text)

	def find_closest_correction(self, word, query):
		"""
		Given a word, find the closest correction using the SpellCheck class.
		"""
		# self.spellChecker = SpellCheck(query)
		# Find top 5 candidate corrections for the word
		candidates = self.spellChecker.find_corrections(word)
        
		# Select the candidate with the minimum edit distance to the original word
		closest_correction = min(candidates, key=lambda candidate: edit_distance(word, candidate))
		
		return closest_correction

	def preprocessQueries(self, queries):
		"""
		Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
		"""

		# Segment queries
		segmentedQueries = []
		for query in queries:
			segmentedQuery = self.segmentSentences(query)
			segmentedQueries.append(segmentedQuery)
		json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'))
		# Tokenize queries
		tokenizedQueries = []
		for query in segmentedQueries:
			tokenizedQuery = self.tokenize(query)
			tokenizedQueries.append(tokenizedQuery)
		json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'))
		# Stem/Lemmatize queries
		reducedQueries = []
		for query in tokenizedQueries:
			reducedQuery = self.reduceInflection(query)
			reducedQueries.append(reducedQuery)
		json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'))
		# Remove stopwords from queries
		stopwordRemovedQueries = []
		for query in reducedQueries:
			stopwordRemovedQuery = self.removeStopwords(query)
			stopwordRemovedQueries.append(stopwordRemovedQuery)
		json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'))

		if self.args.custom:
			# Spell check and find closest corrections in a more efficient manner
			correctedQueries = []
			correctedTopFiveWordsInQueries = []
			for query in stopwordRemovedQueries:
				correctedQuery = [
					[self.find_closest_correction(word, query) for word in sentence] for sentence in query
				]
				correctedTopFiveWordsInQuery = [
					[self.spellChecker.find_corrections(word) for word in sentence] for sentence in query
				]
				correctedQueries.append(correctedQuery)
				correctedTopFiveWordsInQueries.append(correctedTopFiveWordsInQuery)
			
			# Optionally, dump the corrections to a file
			json.dump(correctedQueries, open(self.args.out_folder + "corrected_queries.txt", 'w'))
			json.dump(correctedTopFiveWordsInQueries, open(self.args.out_folder + "corrected_top_five_words_in_queries.txt", 'w'))
			preprocessedQueries = correctedQueries
		else:
			preprocessedQueries = stopwordRemovedQueries
		
		return preprocessedQueries

	def preprocessDocs(self, docs):
		"""
		Preprocess the documents
		"""
  
		# Segment docs
		segmentedDocs = []
		for doc in docs:
			segmentedDoc = self.segmentSentences(doc)
			segmentedDocs.append(segmentedDoc)
		json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))
		# Tokenize docs
		tokenizedDocs = []
		for doc in segmentedDocs:
			tokenizedDoc = self.tokenize(doc)
			tokenizedDocs.append(tokenizedDoc)
		json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))
		# Stem/Lemmatize docs
		reducedDocs = []
		for doc in tokenizedDocs:
			reducedDoc = self.reduceInflection(doc)
			reducedDocs.append(reducedDoc)
		json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))
		# Remove stopwords from docs
		stopwordRemovedDocs = []
		for doc in reducedDocs:
			stopwordRemovedDoc = self.removeStopwords(doc)
			stopwordRemovedDocs.append(stopwordRemovedDoc)
		json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'))

		if(args.custom):
			# Intializing spell check
			self.spellChecker = SpellCheck(stopwordRemovedDocs)
   
		preprocessedDocs = stopwordRemovedDocs
		return preprocessedDocs



	def evaluateDataset(self):
		"""
		Evaluate document-query relevances for all document-query pairs
		"""
		nltk.download('wordnet')
		nltk.download('punkt')
		nltk.download('stopwords')

		# Read queries
		queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]
		queries = [item["query"] for item in queries_json]
		

		# Read documents
		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
		docs = [item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)
		# Process queries 
		processedQueries = self.preprocessQueries(queries)
		
		
  
		# Remaning code will be added later


	def handleCustomQuery(self):
		"""
		Take a custom query as input and return relevances with all documents
		"""
		nltk.download('wordnet')
		nltk.download('punkt')
		nltk.download('stopwords')

		#Get query
		print("Enter query below")
		query = input()
		

		# Read documents
		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:10]
		docs = [item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)
        # Process documents
		processedQuery = self.preprocessQueries([query])[0]
		
		# Remaning code will be added later


if __name__ == "__main__":

	# Create an argument parser
	parser = argparse.ArgumentParser(description='main.py')

	# Tunable parameters as external arguments
	parser.add_argument('-dataset', default = "cranfield/", 
						help = "Path to the dataset folder")
	parser.add_argument('-out_folder', default = "output/", 
						help = "Path to output folder")
	parser.add_argument('-segmenter', default = "punkt",
	                    help = "Sentence Segmenter Type [naive|punkt]")
	parser.add_argument('-tokenizer',  default = "ptb",
	                    help = "Tokenizer Type [naive|ptb]")
	parser.add_argument('-custom', action = "store_true", 
						help = "Take custom query as input")
	
	# Parse the input arguments
	args = parser.parse_args()

	# Create an instance of the Search Engine
	searchEngine = SearchEngine(args)

	# Either handle query from user or evaluate on the complete dataset 
	if args.custom:
		searchEngine.handleCustomQuery()
	else:
		searchEngine.evaluateDataset()
