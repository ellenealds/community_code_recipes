---
created: 2023-04-02T19:03:25 (UTC +01:00)
tags: []
source: https://txt.cohere.ai/turbocharge-semantic-search-with-ai/
author: Elle Neal, Guest Author
---

# Turbocharge Semantic Search With AI in 5 Easy Steps

> ## Excerpt
> Transform your documentation search experience using Streamlit and powerful language models.


TL;DR:


In this post, we walk you through building Cofinder, an AI-powered semantic search app using Cohere's API and Streamlit. The 5-step process includes pre-processing data sources, creating a search index, building the Streamlit front end, adding an AI search function, and adding AI-generated answer functions. Join the NLP dev community and enhance your document search experience with this exciti

---
### Transform your documentation search experience using Streamlit and powerful language models.

### TL;DR:

  
In this post, we walk you through building Cofinder, an AI-powered semantic search app using Cohere's API and Streamlit. The 5-step process includes pre-processing data sources, creating a search index, building the Streamlit front end, adding an AI search function, and adding AI-generated answer functions. Join the NLP dev community and enhance your document search experience with this exciting project!

___

In this article, I will introduce [Cofinder](https://ellenealds-coheresemanticsearchtool-main-8phb6k.streamlit.app/?ref=txt.cohere.ai), a semantic search application built using Cohere’s technology. I’ll [provide you with a repository](https://github.com/ellenealds/streamlit_template_cohere_semantic_search?ref=txt.cohere.ai) and explanations of the code snippets to help you follow along and build your own semantic search application.

We will go through the five stages of building your application:

1.  ****Pre-processing your data sources****
2.  ****Creating a search index****
3.  ****Building your Streamlit front end****
4.  ****Adding an AI search function****
5.  ****Adding AI-generated answer functions.****

![](https://miro.medium.com/v2/resize:fit:770/1*l6wNUN-m2IQy2STFCYiLvg.png)

In recent years, natural language processing (NLP) has rapidly advanced, thanks to the latest generation of large language models. However, not all developers are able to use these models due to the high computing barriers and the lack of technical expertise required to do so. This is where Cohere comes in.

Cohere provides access for all developers, without the need for ML expertise, all that’s needed to access LLMs via cohere is a simple API call to large language models that can be used to generate or analyze text to do things like write copy, moderate content, classify data and extract information, all at a massive scale. Cohere offers a generous free developer tier, meaning you can build and test your ideas quickly and with no cost.

> __Join Cohere Discord channel [here](https://discord.gg/co-mmunity?ref=txt.cohere.ai) to hang with the supportive and innovative NLP dev community and get up to speed with latest Cohere API updates__

## What is Cofinder?

[This was a demo project](https://lablab.ai/event/semantic-search-hackathon/cofinder/cofinder?ref=txt.cohere.ai) I built for my first hackathon with [Lablab.ai](https://lablab.ai/event?ref=txt.cohere.ai), I would recommend hackathons to anyone that wants to learn and push themselves in the space of AI.

> __Cofinder is designed to help the Cohere Community find relevant content in one place based on their personal goals. With Cofinder, users can ask natural language questions, and the tool will provide the most relevant content, aim to answer their questions, and provide context.__

Its inspiration comes from Cohere’s mission to give technology language and to put large language models into more hands.

Cofinder is the result of my desire to make it easier for the Cohere community to access the wealth of knowledge and resources available on the platform. [While Cohere provides an incredible variety of resources, such as product explanations, tutorials, open repositories, and a Discord channel](https://lablab.ai/tech/cohere?ref=txt.cohere.ai), I recognized that finding specific information can still be time-consuming and inefficient. The dataset behind Cofinder is text extracted from these sources. By creating a semantic search tool that brings together information from multiple sources, Cofinder aims to streamline the process and save users valuable time. The goal is to enhance the Cohere community experience by making it easy for developers, entrepreneurs, corporations, and data scientists to find what they need in one place.

![](https://miro.medium.com/v2/resize:fit:561/1*6ZhcwY_-z-XNVlbjwTqOfQ.png)

Let’s build our own semantic search application to enhance search!

[![](https://txt.cohere.ai/content/images/2023/03/image-4.png)](https://ellenealds-coheresemanticsearchtool-main-8phb6k.streamlit.app/?ref=txt.cohere.ai)

Cofinder | [Streamlit](https://ellenealds-coheresemanticsearchtool-main-8phb6k.streamlit.app/?ref=txt.cohere.ai) demo

But before we dive into the technical details, let’s take a closer look at the Cohere products we used in building Cofinder. We will use Cofinder to learn more about each product.

### Cohere’s Embedding and Generate Endpoints: The Key to Cofinder’s Semantic Search

Cofinder uses two of Cohere’s products, [co.embed](https://docs.cohere.ai/reference/embed?ref=txt.cohere.ai) and [co.generate](https://docs.cohere.ai/reference/generate?ref=txt.cohere.ai), to power its semantic search functionality. `co.embed` is an API endpoint that gives easy access to a robust embedding model that generates numeric representations of text, [which can be used for various natural language processing (NLP) tasks](https://cohere.ai/embed?ref=txt.cohere.ai) such as clustering and classification. In Cofinder, `co.embed` is used to generate vector embeddings for the articles that users can search.

`co.generate` is an API endpoint that uses a text representation model that generates text based on the given prompt. In Cofinder, `co.generate` is used to formulate an [answer to the user’s question using the context from the search and the question itself.](https://cohere.ai/generate?ref=txt.cohere.ai) By leveraging these powerful NLP models, Cofinder is able to provide accurate and relevant search results to users, making it easier and more efficient to find information on Cohere

Let’s check out [Cofinder](https://ellenealds-coheresemanticsearchtool-main-8phb6k.streamlit.app/?ref=txt.cohere.ai) to help find relevant Cohere resources.

****Question:**** What are embeddings?

![](https://miro.medium.com/v2/resize:fit:770/1*H2mHffEELdxUDHHS4mv-8Q.png)

Cofinder | What are embeddings?

****Results:**** As Cofinder quickly provides the user with content from across multiple platforms. Each relevant document contains:

-   The category (video, blog, user and product documentation)
-   The content title
-   An answer generated using the search query and content as context
-   Link to the content

We now have a variety of content to watch and read to help understand what embeddings are and formulate a good solid understanding.

![](https://miro.medium.com/v2/resize:fit:625/1*A1clZz2DTujWhwCgjOUbgQ.png)

Cofinder | Search response

Once you have a better understanding of what embeddings are, you may now want to explore what you can do with embeddings, let’s ask Cofinder again.

****Question:**** What can embeddings be used for?

This time, our content is more focused on possible use cases, and each one gives us the opportunity to explore multiple areas in more depth.

![](https://miro.medium.com/v2/resize:fit:631/1*HqRpO4zedN-xHUJJYxzPMQ.png)

Cofinder | Search results

### Let’s Start Building!

## 5 Steps to Build your own Semantic Search Application

We can breakdown the process into five steps of development:

1.  ****Data Sources:**** pre-processing the article text into chunks.
2.  ****Embeddings & Search Index:**** use `co.embed` to obtain a vector representation of our data. Store your embeddings in a vector database which we will later use to search for relevant content.
3.  ****Front End:**** Streamlit for our users to interact with our search engine.
4.  ****Search:**** we use `co.embed` to get the vector representation of the user query, using nearest neighbors to return relevant content.
5.  ****Answer:**** use `co.generate` to answer the query given the context from the search results and the question.

## Github Repository

Here is a GitHub repository template to fork and build your own application: [ellenealds/streamlit\_template\_cohere\_semantic\_search (github.com)](https://github.com/ellenealds/streamlit_template_cohere_semantic_search?ref=txt.cohere.ai)

### Repository Contents

-   ****cohere\_text\_preprocessing.csv:**** contains text prior to pre-processing
-   ****preprocessing.ipynb:**** this contains our code for generating embeddings and preparing our search index using the cohere\_text\_preprocessing.csv file
-   ****main.py:**** this contains our code for the Streamlit app

These files will be produced when you run ****preprocessing.ipynb****

-   ****cohere\_text\_final.csv:**** text after pre-processing
-   ****search\_index.ann:**** a search index containing our embeddings

You will need your Cohere API key to continue with the development.

> ****New to Cohere?****
> 
> [Get Started](https://dashboard.cohere.ai/welcome/register?ref=txt.cohere.ai) now and get unprecedented access to world-class Generation and  
> Representation models with billions of parameters.
> 
> ****Important****
> 
> Create your secrect.toml in the repository and enter your cohere API Key as API\_KEY = ‘…’

## Creating our Search Index

Our first three stages are to ****pre-process the text**** by splitting it into small chunks, we will then ****generate embeddings**** using `co.embed` and finally ****create a search index**** using Annoy.

__This section of code can be found in preprocessing.ipynb__

![](https://miro.medium.com/v2/resize:fit:770/1*LmQVJOhNwWOIcvxLFw7RRA.png)

[Semantic Search (cohere.ai)](https://docs.cohere.ai/docs/semantic-search?ref=txt.cohere.ai)

### 1\. Data Sources — Pre-processing Text

The repository contains a CSV file called `cohere_text_preprocessing.csv` with a row for each URL, the title and the text taken from the webpage. Each item contains a category and type which is returned in the search results.

![](https://miro.medium.com/v2/resize:fit:770/1*st7DViVAFI62NB7un91Q_Q.png)

CSV File | cohere\_text\_preprocessing

__Note: Some of the information may be outdated as this text was extracted in October 2022 and is for demonstration purposes only.__

After importing the `CSV` file, we need to pre-process the text into chunks, the chunks will be used in a later stage as context where we want to generate an answer. As we will use `co.generate` for this task, we need to make sure our text chunks are no larger than 1500 words.

To maintain context, we will split the text chunks into 1500 and overlap the chunks using 500 tokens from the previous chunk relevant to the article.

```
import textwrap

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
new_df = chunk_text(df)
```

### 2a. Get embeddings

We use the `co.embed` to obtain a vector representation of our `data.import` cohere

```
import cohere
import numpy as np

co = cohere.Client('<API_KEY>')
# Get the embeddings
embeds = co.embed(texts=list(df['text_chunk']),
                  model="large",
                  truncate="RIGHT").embeddings
# Check the dimensions of the embeddings
embeds = np.array(embeds)
embeds.shape
```

Great! We now have our pre-processed text and embeddings ready for the next stage where we create a search index to store the data.

### 2b. Search Index

We now create a search index using Annoy to store the embeddings.

```
from annoy import AnnoyIndex

# Create the search index, pass the size of embedding
search_index = AnnoyIndex(embeds.shape[1], 'angular')

# Add all the vectors to the search index
for i in range(len(embeds)):
    search_index.add_item(i, embeds[i])

search_index.build(10) # 10 trees
search_index.save('search_index.ann')
```

The final three stages are to ****create our Streamlit application,**** we load the relevant libraries, data and search index from the previous stages. We then add functions to ****generate embeddings to search**** the Annoy index for the user’s query, and ****generate an answer**** from the context****.**** This is all tied together in the Streamlit app with widgets for user input and markdown to display the results.

__This section of code can be found in [main.py](https://github.com/ellenealds/streamlit_template_cohere_semantic_search/blob/main/main.py?ref=txt.cohere.ai)_._

### 3\. Front End — Streamlit

In the `main.py` file__,__ we build our code for the Streamlit application.

Here we import the libraries we need, the API key, initiate our cohere client and load the search index and `CSV` file.

```
import streamlit as st
import cohere  
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from concurrent.futures import ThreadPoolExecutor
import toml

# Load the secret.toml file
with open('secret.toml') as f:
    secrets = toml.load(f)

# Access the API key value
api_key = secrets['API_KEY']
co = cohere.Client('api_key')

# Load the search index
search_index = AnnoyIndex(f=4096, metric='angular')
search_index.load('search_index.ann')

# load the csv file called cohere_final.csv
df = pd.read_csv('cohere_text_final.csv')

# title
st.title("Cofinder")
st.subheader("A semantic search tool built for the Cohere community")
```

### 4\. Search Function

The `search` function takes in a query, the number of search results to return `n_results`, a dataframe `df`, an index `search_index`, and an embedding model `co.embed()`. Here's a step-by-step breakdown of what the function does:

1.  The function uses the `co.embed` function to get the embedding of the query using the specified model (`"large"`).
2.  The function uses the `search_index.get_nns_by_vector` function to get the `n_results` nearest neighbors to the query embedding in the specified index `search_index`. The function returns the indices of the nearest neighbors as well as the similarity scores.
3.  The function filters the original dataframe (`df`) to include only the rows that correspond to the nearest neighbors returned in step 2.
4.  The function adds two new columns to the filtered dataframe: `similarity`, which contains the similarity scores between the query embedding and the document embeddings; and `nearest_neighbors`, which contains the indices of the nearest neighbors.
5.  Finally, the function sorts the filtered dataframe by similarity in descending order and returns it.

The results of the search will be used as context to answer the user’s question.

```
def search(query, n_results, df, search_index, co):
    # Get the query's embedding
    query_embed = co.embed(texts=[query],
                    model="large",
                    truncate="LEFT").embeddings
    
    # Get the nearest neighbors and similarity score for the query 
    # and the embeddings, append it to the dataframe
    nearest_neighbors = search_index.get_nns_by_vector(
        query_embed[0], 
        n_results, 
        include_distances=True)
    # filter the dataframe to include the nearest neighbors using the index
    df = df[df.index.isin(nearest_neighbors[0])]
    df['similarity'] = nearest_neighbors[1]
    df['nearest_neighbors'] = nearest_neighbors[0]
    df = df.sort_values(by='similarity', ascending=False)
    return df
```

When we run this function, we get a JSON output containing the article title and the context relevant to the query

****Question:**** How do I use Cohere to build a chatbot?

```
[
0:[
0:" Use Cohere's Classify endpoint for intent recognition and text classification."
1:"Paragraph:Use Cohere’s Classify for intent recognition and text classification. Starting from the same Transformer architecture as Google’s Search and Translate, Cohere’s Classify endpoint can read, understand, and label the intent behind unstructured customer queries with exceptional speed and accuracy. Developers can further finetune Cohere’s large language models with their own dataset per their business or industry. Set the road rules, define your core customer intent categories, and train Cohere’s Classify model by providing it with example customer queries for each category. Start the engine, Classify then combines this custom understanding of your intent categories with a deep understanding of language and context provided by our large language models to create a finetuned model, ready to handle your customer data. Get driving, every time a user enters a prompt into your chatbot, Cohere will classify the request by intent, allowing your bot to provide the most relevant answer back. Answer the question using this paragraph. Question: How do I use Cohere to build a chatbot? Answer:"
]
1:[
0:" Cohere uses billions of examples to train their models, which allows them to understand and recognize the intent behind a specific phrase. This intent recognition is used to power sophisticated chatbots that move beyond basic keyword recognition."
1:"Paragraph:Billions of examples, Cohere’s models are trained on billions of sentences, allowing them to understand and recognize the intent behind a specific phrase. Our models excel at intent recognition tasks and can power sophisticated chatbots that move beyond basic keyword recognition. Answer the question using this paragraph. Question: How do I use Cohere to build a chatbot? Answer:"
]
2:[
0:" Cohere is a chatbot platform that is powered by basic keyword recognition. This means that the bot's ability to 'understand' queries is dependent upon customers using very specific phrases."
1:"Paragraph:Keywords aren’t enough. Many chatbot platforms are powered by basic keyword recognition. This means that the bot’s ability to “understand” queries is dependent upon customers using very specific phrases. Developers could augment this by painstakingly predicting the myriad ways that a customer could phrase a specific query, including slang, misspellings, and differences in context. However, that task would be time consuming, if not downright impossible Answer the question using this paragraph. Question: How do I use Cohere to build a chatbot? Answer:"
]
3:[
0:" "Cohere's API is created to help you build natural language understanding and generation into your production with a few lines of code. Our Quickstart Tutorials will show you how to implement our API from zero-to-one in under 5 minutes.""
1:"Paragraph:["Cohere's API is created to help you build natural language understanding and generation into your production with a few lines of code. Our Quickstart Tutorials will show you how to implement our API from zero-to-one in under 5 minutes. ", 'Chatbots are designed to understand and respond to human language. They need to be able to understand the text they hear and understand the context of the conversation. They also need to be able to respond to people’s questions and comments in a meaningful way. To accomplish this, chatbots must be able to recognize specific intents that people express in conversation.Here is an example of classifying the intent of customer inquiries on an eCommerce website into three categories: Shipping and handling policy, Start return or exchange, or Track order.', 'Updated about 1 month ago '] Answer the question using this paragraph. Question: How do I use Cohere to build a chatbot? Answer:"
]
4:[
0:" Our product is a Web based Application which improves the efficiency of chat based support systems by automating repetitive parts of the workflow. This is done by utilising Cohere's API in order to provide smart shortcuts for the Chat Support Agents."
1:"Paragraph:Our product is a Web based Application which improves the efficiency of chat based support systems by automating repetitive parts of the workflow. This is done by utilising Cohere’s API in order to provide smart shortcuts for the Chat Support Agents. We aim to maximise Customer and Customer Support Agent satisfaction by making the lookup of product and service related answers instantaneous, thereby allowing the Customer Support Agent to put more effort into the interaction with the customer rather than the mundane task of researching answers. Answer the question using this paragraph. Question: How do I use Cohere to build a chatbot? Answer:"
]
]
```

### 5\. Answer Function

We now want to take the user question and generate an answer to the question given the search results from the previous function.

Overall, the `display` function below is broken down into two tasks,

1.  The `gen_answer` and `gen_better_answer` functions are used to generate answers for the query and each of the search results. The `gen_answer` function generates an initial answer based on a prompt that includes the paragraph and the question, while `gen_better_answer` generates a better answer by incorporating the initial answers and the question.
2.  The results are displayed in a user-friendly format using the `st` module from the Streamlit library. The search query is displayed as a subheader, followed by the better answer generated by `gen_better_answer`. The relevant documents are then displayed, one by one. Each document is displayed with its type, category, title, and link, followed by the initial answer generated by `gen_answer`. The text of the document is then collapsed and can be expanded by clicking on "Read more".

```
def display(query, results):
    # 1. Run co.generate functions to generate answers

    # for each row in the dataframe, generate an answer concurrently
    with ThreadPoolExecutor(max_workers=1) as executor:
        results['answer'] = list(executor.map(gen_answer, 
                                              [query]*len(results), 
                                              results['text']))
    answers = results['answer'].tolist()
    # run the function to generate a better answer
    answ = gen_better_answer(query, answers)
    
    # 2. Display the resuls in a user-friendly format
    
    st.subheader(query)
    st.write(answ)
    # add a spacer
    st.write('')
    st.write('')
    st.subheader("Relevant documents")
    # display the results
    for i, row in results.iterrows():
        # display the 'Category' outlined
        st.markdown(f'**{row["Type"]}**')
        st.markdown(f'**{row["Category"]}**')
        st.markdown(f'{row["title"]}')
        # display the url as a hyperlink
        # add a button to open the url in a new tab
        st.markdown(f'[{row["link"]}]({row["link"]})')
        st.write(row['answer'])
        # collapse the text
        with st.expander('Read more'):
            st.write(row['text'])
        st.write('')
```

Let’s explore the `gen_answer` and `gen_better_answer` functions to see what is happening.

![](https://miro.medium.com/v2/resize:fit:770/1*3sUd3W4bgacWUd2AiLwIJA.png)

Cofinder | search context and generate the best answer

Here we create two co.generate() prompts, these functions use Cohere’s pre-trained models to generate text that answers a given question from the context provided from the search results.

-   `gen_answer(q, para)`: This function returns an answer to the question given the context returned from our search function. In the display function above, we ran this function by iterating over each of the search results from the JSON output to gather context from multiple sources.
-   `gen_better_answer(ques, ans)`: This function takes the question and all of the responses from the`gen_answer` function, and it uses the combined answers to formulate a more rounded answer taking into account all the available resources.

The `max_tokens` and `temperature` parameters can be tuned to control the length and randomness of the generated text.

```
# define a function to generate an answer
def gen_answer(q, para): 
    response = co.generate( 
        model='command-xlarge-20221108', 
        prompt=f'''Paragraph:{para}\n\n
                Answer the question using this paragraph.\n\n
                Question: {q}\nAnswer:''', 
        max_tokens=100, 
        temperature=0.4)
    return response.generations[0].text

def gen_better_answer(ques, ans): 
    response = co.generate( 
        model='command-xlarge-20221108', 
        prompt=f'''Answers:{ans}\n\n
                Question: {ques}\n\n
                Generate a new answer that uses the best answers 
                and makes reference to the question.''', 
        max_tokens=100, 
        temperature=0.4)
    return response.generations[0].text
```

Finally, we add a Streamlit search input along with some question examples for the user, and a button that runs our functions.

```
# add the if statements to run the search function when the user clicks the buttons

query = st.text_input('Ask a question about Cohere')
# write some examples to help the user

st.markdown('''Try some of these examples: 
- What is the Cohere API?
- What are embeddings?
- What is the Cohere playground?
- How can I build a chatbot?''')

if st.button('Search'):
    results = search(query, 3, df, search_index, co)
    display(query, results)
```

Well done! You are now ready to [publish your Streamlit application to the cloud!](https://streamlit.io/cloud?ref=txt.cohere.ai)

## Calling All Developers

The vision for Cofinder was “Built for the community, by the community”, can you help make Cofinder a better search tool for our community? Please contact me on [LinkedIn](https://www.linkedin.com/in/elle-neal-78994617/?ref=txt.cohere.ai) or Discord Elle Neal#0726 to discuss how we can make this happen.

As a starting point, here are some potential features I captured at the end of the hackathon, I am excited to see how we can take this application to the next level!

![](https://miro.medium.com/v2/resize:fit:770/1*H4SODILfgEpduqBHAvzADw.png)

Cofinder | Feature Pipeline

## Next Steps

So there you have it, a step-by-step guide to building an AI-powered semantic search application using Cohere’s API. But this is just the beginning! Cofinder was built for the community, by the community, and we are always looking for ways to make it better. As a community member, your input is crucial in shaping the future of Cofinder. Please reach out to me on LinkedIn or Discord to discuss potential features and how we can work together to take this application to the next level.

Let’s continue to innovate and make the Cohere community experience even better.

___

_This article was originally published on [Medium](https://medium.com/@elle.neal_71064/5-steps-to-build-an-ai-powered-semantic-search-application-using-coheres-api-f5a60cb797be?ref=txt.cohere.ai)._
