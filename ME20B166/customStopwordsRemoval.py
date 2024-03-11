import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
nltk.download('punkt')
nltk.download('stopwords')

# Load the Cranfield dataset
with open('./cranfield/cran_docs.json', 'r') as file:
    cranfield_data = json.load(file)
    documents = [item["body"] for item in cranfield_data]

# Tokenize and aggregate word frequencies
word_counts = Counter()
for doc in documents:
    words = word_tokenize(doc.lower())  # Tokenize and lowercase
    words = [word for word in words if word.isalpha()]  # Keep only alphabetic tokens
    word_counts.update(words)

# Identify the top N high-frequency words as potential stopwords
N = 500  # You might adjust this based on your analysis
top_n_words = [word for word, count in word_counts.most_common(N)]

# Compare with NLTK's stopwords
nltk_stopwords = set(stopwords.words('english'))
custom_stopwords = set(top_n_words)
common_stopwords = nltk_stopwords.intersection(custom_stopwords)
unique_to_cranfield = custom_stopwords.difference(nltk_stopwords)
unique_to_nltk = nltk_stopwords.difference(custom_stopwords)

print(f"Common stopwords: {sorted(common_stopwords)}")
print(f"Length of Common stopwords: {len(common_stopwords)}")
print(f"Unique to Cranfield: {sorted(unique_to_cranfield)}")
print(f"Unique to NLTK: {sorted(unique_to_nltk)}")
