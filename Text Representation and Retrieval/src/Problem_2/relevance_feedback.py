import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_centroid(D):

    summation = np.zeros((1, D.shape[1]))

    for i in range(D.shape[0]):
        summation += D[i, :]

    return summation / D.shape[0]

def relevance_feedback(vec_docs, vec_queries, sim, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """

    alpha = 1.0
    beta = 0.9
    gamma = .5

    for epoch in range(2):
        vec_queries_new = np.zeros((vec_queries.shape))

        for q in range(vec_queries.shape[0]):
            old_query = vec_queries[q, :]

            r = sim[:, q]
            sorted_ = np.argsort(r)
            
            first_n = sorted_[:n]
            D_irrel = vec_docs[first_n, :]

            last_n = sorted_[-n:]
            D_rel = vec_docs[last_n, :]

            centroid_rel = get_centroid(D_rel)
            centroid_irrel = get_centroid(D_irrel)

            new_query = (alpha/n) * old_query + (beta/n) * centroid_rel - (gamma/n) * centroid_irrel
            new_query = new_query.clip(min=0)
            vec_queries_new[q, :] = new_query

        rf_sim = cosine_similarity(vec_docs, vec_queries_new)
        vec_queries = vec_queries_new
        sim = rf_sim
    
    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """

    alpha = 0.1
    beta = 0.9
    gamma = 1.4
    closest = 5

    vec_docs = vec_docs / np.sum(vec_docs, axis=1)

    thesaurus = np.dot(np.transpose(vec_docs), vec_docs)
    
    for epoch in range(2):
        vec_queries_new = np.zeros((vec_queries.shape))

        for q in range(vec_queries.shape[0]):
            old_query = vec_queries[q, :].reshape(1, -1)

            highest = np.argmax(old_query)
            highest_value = np.max(old_query)

            closest_words = np.argsort(thesaurus[highest, :])[:, -closest:]
            closest_words = np.array(closest_words)[0]
            
            for idx in range(closest):
                old_query[:, closest_words[idx]] = highest_value

            old_query = old_query.reshape(1, -1)
            
            r = sim[:, q]
            sorted_ = np.argsort(r)
            
            first_n = sorted_[:n]
            D_irrel = vec_docs[first_n, :]

            last_n = sorted_[-n:]
            D_rel = vec_docs[last_n, :]

            centroid_rel = get_centroid(D_rel)
            centroid_irrel = get_centroid(D_irrel)

            new_query = (alpha/n) * old_query + (beta/n) * centroid_rel - (gamma/n) * centroid_irrel
            new_query = new_query.clip(min=0)
            vec_queries_new[q, :] = new_query

        rf_sim = cosine_similarity(vec_docs, vec_queries_new)
        vec_queries = vec_queries_new
        sim = rf_sim
    
    return rf_sim

