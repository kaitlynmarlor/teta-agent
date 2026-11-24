"""Step 5: Embeddings — turning text into meaning vectors."""

from sentence_transformers import SentenceTransformer

# Load the embedding model (first run downloads ~1.3GB)
print("Loading embedding model (this takes a minute the first time)...")
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
print("Model loaded!\n")

# Embed three sentences — two about thymomas, one unrelated
text_a = "Thymoma B1 is a lymphocyte-rich tumor resembling normal thymic cortex."
text_b = "Type B1 contains abundant immature T cells with scattered epithelial cells."
text_c = "The weather in Chicago is cold in January."

embedding_a = model.encode(text_a)
embedding_b = model.encode(text_b)
embedding_c = model.encode(text_c)

print(f"Each embedding is a list of {len(embedding_a)} numbers.")
print(f"First 5 numbers of text A: {embedding_a[:5].round(4)}\n")

# Measure similarity between them
from numpy import dot
from numpy.linalg import norm

def similarity(v1, v2):
    return dot(v1, v2) / (norm(v1) * norm(v2))

sim_ab = similarity(embedding_a, embedding_b)
sim_ac = similarity(embedding_a, embedding_c)

print(f'A: "{text_a}"')
print(f'B: "{text_b}"')
print(f'C: "{text_c}"')
print()
print(f"Similarity A ↔ B: {sim_ab:.4f}  (same topic, different words)")
print(f"Similarity A ↔ C: {sim_ac:.4f}  (completely unrelated)")
print()
print("A and B describe the same thing — B1 thymoma — using")
print("different terminology, yet the model knows they're related.")
print("This is why vector search finds answers that keyword search misses.")