## Chunk Text Function

This function takes a pandas dataframe containing text and chunks it into smaller pieces. 

### `chunk_text`

This function takes a pandas dataframe containing text to be chunked, a width specifying the size of each chunk, and an overlap specifying the amount of overlap between adjacent chunks.

### Parameters

* `df` (`pd.DataFrame`): A pandas dataframe with a 'text' column containing the text to be chunked.
* `width` (`int`, optional): The width of each chunk. Defaults to 1500.
* `overlap` (`int`, optional): The amount of overlap between adjacent chunks. Defaults to 500.

### Returns

* `pd.DataFrame`: A new pandas dataframe with the text chunked into pieces of the specified width, with the specified overlap. The returned dataframe has columns 'id', 'text_chunk', and 'start_index'.

### Example Usage

```python
import pandas as pd

# create a dataframe with some text
df = pd.DataFrame({'id': [1, 2],
                   'text': ['This is some text that we want to chunk into smaller pieces.',
                            'This is another piece of text that we want to chunk.']})

# define the width and overlap for chunking the text
width = 20
overlap = 10

# define the chunking function
def chunk_text(df, width=1500, overlap=500):
    # create an empty dataframe to store the chunked text
    new_df = pd.DataFrame(columns=['id', 'text_chunk'])

    # iterate over each row in the original dataframe
    for index, row in df.iterrows():
        # split text into chunks of size 'width', with overlap of 'overlap'
        chunks = []
        for i in range(0, len(row['text']), width - overlap):
            chunk = row['text'][i:i+width]
            chunks.append(chunk)

        # iterate over each chunk and add it to the new dataframe
        for i, chunk in enumerate(chunks):
            # calculate the start index based on the chunk index and overlap
            start_index = i * (width - overlap)
            
            # create a new row with the chunked text and the original row's ID
            new_row = {
              'id': row['id'], 
              'text_chunk': chunk, 
              'start_index': start_index}

            new_df = new_df.append(new_row, ignore_index=True)

    return new_df

# run the function on the dataframe
new_df = chunk_text(df, width=width, overlap=overlap)

# print the new dataframe with the text chunked into smaller pieces
print(new_df)

```
