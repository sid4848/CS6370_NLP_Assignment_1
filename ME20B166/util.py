# Add your import statements here
import nltk
from nltk.corpus import wordnet
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')


# Add any utility functions here
## Edit distance for spell correction
def edit_distance(str1, str2):
    """
    Compute the edit distance between two input strings.
    
    Args:
    str1 (str): The first input string.
    str2 (str): The second input string.
    
    Returns:
    int: The edit distance between the two strings.
    """
    m, n = len(str1), len(str2)
    
    # Initialize a matrix to store the distances
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base case initialization
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    # Compute the distances
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 3
            dp[i][j] = min(dp[i - 1][j] + 1,        # Deletion
                           dp[i][j - 1] + 1,        # Insertion
                           dp[i - 1][j - 1] + cost) # Substitution
            
    return dp[m][n]



## Synset of corresponding words
def get_synsets(word):
    
    synsets = wordnet.synsets(word)
    return synsets

## Function to calculate the path-based similarity between synsets of two words
def calculate_similarity(synsets1, synsets2):
    max_similarity = 0
    for syn1 in synsets1:
        for syn2 in synsets2:
            similarity = syn1.path_similarity(syn2)
            if similarity is not None and similarity > max_similarity:
                max_similarity = similarity
    return max_similarity
