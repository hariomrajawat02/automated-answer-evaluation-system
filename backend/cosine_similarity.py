from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_cosine_similarity(vector1, vector2):
    """
    Calculate cosine similarity between two vectors
    """

    v1 = np.array(vector1).reshape(1, -1)
    v2 = np.array(vector2).reshape(1, -1)

    similarity = cosine_similarity(v1, v2)[0][0]

    return float(similarity)