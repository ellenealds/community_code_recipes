from cohere import CohereAPI
import numpy as np

def generate_embeddings(df: pd.DataFrame, api_key: str) -> np.ndarray:
    """Generate embeddings for a dataframe using the Cohere API.

    Args:
        df (pd.DataFrame): A pandas dataframe containing text to be embedded.
        api_key (str): A string containing your Cohere API key.

    Returns:
        np.ndarray: A numpy array containing the embeddings of the text in the dataframe.
    """
    co = CohereAPI(api_key)
    # Get the embeddings
    embeds = co.embed(texts=list(df['text_chunk']),
                      model="large",
                      truncate="RIGHT").embeddings
    # Check the dimensions of the embeddings
    embeds = np.array(embeds)
    return embeds
