import nltk
nltk.download('wordnet')

from nltk.corpus import wordnet

# PART 7: 1
def get_synsets(word):

    synsets = wordnet.synsets(word)
    return synsets

words = ['progress', 'advance']

for word in words:
    synsets = get_synsets(word)
    print(f"Synsets for '{word}':")

    for synset in synsets:
        print(f"  - {synset.name()} ")

    print("\n")


# PART 7: 2
for word in words:
    synsets = get_synsets(word)
    print(f"Definitions for '{word}':")

    for synset in synsets:
        print(f"  - {synset.name()} - {synset.definition()}")

    print("\n")
    

# PART 7: 3
# Load synsets for both words
advance_synsets = wordnet.synsets('advance')
progress_synsets = wordnet.synsets('progress')

# Function to calculate path-based similarity between synsets of two words
def calculate_similarity(synsets1, synsets2):
    max_similarity = 0
    for syn1 in synsets1:
        for syn2 in synsets2:
            similarity = syn1.path_similarity(syn2)
            if similarity is not None and similarity > max_similarity:
                max_similarity = similarity
    return max_similarity

# Calculate and display the maximum path-based similarity between 'advance' and 'progress'
max_similarity = calculate_similarity(advance_synsets, progress_synsets)
print("Max Simalirity between advance and progess: ", max_similarity)