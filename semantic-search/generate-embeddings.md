## Embedding Generation Function

This code block contains a function that can be used to generate embeddings for a pandas dataframe using the Cohere API.

### `generate_embeddings`

This function takes a pandas dataframe and a Cohere API key as input and returns embeddings of the text in the dataframe.

#### Parameters

* `df` (`pd.DataFrame`): A pandas dataframe containing text to be embedded.
* `api_key` (`str`): A string containing your Cohere API key.

#### Returns

* `np.ndarray`: A numpy array containing the embeddings of the text in the dataframe.

### Example Usage

```python
import cohere
co = cohere.Client('API_KEY')
def embed(cluster):
    response = co.embed(
      model='large',
      texts=cluster
    )
    return response.embeddings
```
