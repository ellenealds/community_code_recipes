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
import pandas as pd
import numpy as np
from cohere import CohereAPI

# create a CohereAPI object
co = CohereAPI('<API_KEY>')

# create a dataframe with some text
df = pd.DataFrame({'id': [1, 2],
                   'text': ['This is some text that we want to embed.',
                            'This is another piece of text that we want to embed.']})

# generate embeddings using the generate_embeddings function
embeddings = generate_embeddings(df, '<API_KEY>')

# print the shape of the embeddings
print(embeddings.shape)
