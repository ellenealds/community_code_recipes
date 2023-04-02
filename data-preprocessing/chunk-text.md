## Chunk Text Function

This function takes a pandas dataframe containing text and chunks it into smaller pieces. 

### Parameters

* `df` (`pd.DataFrame`): A pandas dataframe with a 'text' column containing the text to be chunked.
* `width` (`int`, optional): The width of each chunk. Defaults to 1500.
* `overlap` (`int`, optional): The amount of overlap between adjacent chunks. Defaults to 500.

### Returns

* `pd.DataFrame`: A new pandas dataframe with the text chunked into pieces of the specified width, with the specified overlap.

### Example Usage

```python
import pandas as pd

# create a dataframe with some text
df = pd.DataFrame({'id': [1, 2],
                   'text': ['This is some text that we want to chunk.',
                            'This is another piece of text.']})

# chunk the text using the chunk_text function
new_df = chunk_text(df, width=10, overlap=5)

# print the result
print(new_df)
```
