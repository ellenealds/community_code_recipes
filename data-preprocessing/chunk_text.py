import pandas as pd

def chunk_text(df: pd.DataFrame, width: int = 1500, overlap: int = 500) -> pd.DataFrame:
    """Chunk text into smaller pieces.

    Args:
        df (pd.DataFrame): A pandas dataframe with a 'text' column containing the text to be chunked.
        width (int, optional): The width of each chunk. Defaults to 1500.
        overlap (int, optional): The amount of overlap between adjacent chunks. Defaults to 500.

    Returns:
        pd.DataFrame: A new pandas dataframe with the text chunked into pieces of the specified width, with the specified overlap.
    """
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
