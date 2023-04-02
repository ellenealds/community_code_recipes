from typing import List
from pandas import DataFrame
from cohere import CohereAPI
from annoy import AnnoyIndex
import numpy as np

def search(query: str, n_results: int, df: DataFrame, search_index: AnnoyIndex, co: CohereAPI) -> DataFrame:
    """Search for similar documents using a search index built from embeddings.

    Args:
        query (str): The query text to search for.
        n_results (int): The number of search results to return.
        df (DataFrame): A pandas dataframe containing the documents to search.
        search_index (AnnoyIndex): The Annoy search index built from embeddings.
        co (CohereAPI): A CohereAPI object for generating embeddings.

    Returns:
        DataFrame: A pandas dataframe containing the search results, sorted by similarity score.
    """
    # Get the query's embedding
    query_embed = co.embed(texts=[query],
                           model="large",
                           truncate="LEFT").embeddings

    # Get the nearest neighbors and similarity score for the query
    # and the embeddings, append it to the dataframe
    nearest_neighbors = search_index.get_nns_by_vector(query_embed[0], n_results, include_distances=True)

    # filter the dataframe to include the nearest neighbors using the index
    df = df[df.index.isin(nearest_neighbors[0])]
    df['similarity'] = nearest_neighbors[1]
    df['nearest_neighbors'] = nearest_neighbors[0]
    df = df.sort_values(by='similarity', ascending=False)
    return df
