## Semantic Search Function

This code block contains a function that can be used to search for similar documents using a search index built from embeddings.

### `search`

This function takes a query string, a number of search results to return, a pandas dataframe containing the documents to search, an Annoy search index built from embeddings, and a CohereAPI object as input and returns a pandas dataframe containing the search results, sorted by similarity score.

#### Parameters

* `query` (`str`): The query text to search for.
* `n_results` (`int`): The number of search results to return.
* `df` (`DataFrame`): A pandas dataframe containing the documents to search.
* `search_index` (`AnnoyIndex`): The Annoy search index built from embeddings.
* `co` (`CohereAPI`): A CohereAPI object for generating embeddings.

#### Returns

* `DataFrame`: A pandas dataframe containing the search results, sorted by similarity score.

### Example Usage

```python
import pandas as pd
import numpy as np
from cohere import CohereAPI
from annoy import AnnoyIndex

# create a CohereAPI object
co = CohereAPI('<API_KEY>')

# create a dataframe with some text
df = pd.DataFrame({'id': [1, 2],
                   'text': ['This is some text that we want to search.',
                            'This is another piece of text that we want to search.']})

# generate embeddings using the co.embed function
embeddings = np.array(co.embed(texts=list(df['text']),
                                model="large",
                                truncate="RIGHT").embeddings)

# build an Annoy search index using the embeddings
search_index = AnnoyIndex(embeddings.shape[1], 'angular')
for i in range(len(embeddings)):
    search_index.add_item(i, embeddings[i])
search_index.build(10) # 10 trees

# search for similar documents using the search function
results = search('search text', 1, df, search_index, co)

# print the search results
print(results)
```
