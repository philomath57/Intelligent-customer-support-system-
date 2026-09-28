import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd



class perform_semantic_retrieval:
    def __init__(self):
        pass

    def retrieval_function(self, query, model, metadata_df, index, top_k=10):

        query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)

        scores, indices = index.search(query_embedding, top_k)

        results = metadata_df.iloc[indices[0]].copy()

        results['similarity'] = scores[0]

        results['rank'] = range(1, top_k+1)

        return results

