from annoy import AnnoyIndex
import numpy as np

def build_search_index(embeddings: np.ndarray, trees: int = 10, index_filename: str = 'search_index.ann') -> None:
    """Build a search index for a set of embeddings using the Annoy library.

    Args:
        embeddings (np.ndarray): An array of shape (num_embeddings, embedding_dim) containing the embeddings to be indexed.
        trees (int, optional): The number of trees to use in the search index. Defaults to 10.
        index_filename (str, optional): The name of the file to save the search index to. Defaults to 'search_index.ann'.
    """
    # create the search index
    search_index = AnnoyIndex(embeddings.shape[1], 'angular')

    # add all the embeddings to the search index
    for i in range(len(embeddings)):
        search_index.add_item(i, embeddings[i])

    # build the search index with the specified number of trees
    search_index.build(trees)

    # save the search index to disk
    search_index.save(index_filename)
