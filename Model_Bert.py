import numpy as np
from embedding4bert import Embedding4BERT


def Model_Bio_ClinicalBERT(Data, max_len=128):

    dim = 768
    pooled_vectors = []

    emb4bert = Embedding4BERT(
        model_name="emilyalsentzer/Bio_ClinicalBERT"
    )

    for text in Data:
        text = str(text)
        tokens, embeddings = emb4bert.extract_word_embeddings(text)
        if embeddings.shape[0] < max_len:
            pad = np.zeros((max_len - embeddings.shape[0], dim))
            padded = np.vstack((embeddings, pad))
        else:
            padded = embeddings[:max_len]

        pooled = np.mean(padded, axis=0)
        pooled_vectors.append(pooled)

    return np.array(pooled_vectors, dtype=np.float32)
