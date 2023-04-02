---
created: 2023-04-02T18:59:04 (UTC +01:00)
tags: []
source: https://txt.cohere.ai/generative-content-keyword-research/
author: Meor Amer
---

# Fueling Generative Content with Keyword Research

> ## Excerpt
> In this Python walkthrough, we’ll build a content idea generator that is backed by keyword research.

---
## Table of Contents

-   [Introduction](https://txt.cohere.ai/generative-content-keyword-research/#introduction)
-   [Step 1: Get a list of high-performing keywords](https://txt.cohere.ai/generative-content-keyword-research/#step-1-get-a-list-of-high-performing-keywords)
-   [Step 2: Group the keywords into topics](https://txt.cohere.ai/generative-content-keyword-research/#step-2-group-the-keywords-into-topics)
    -   [Embed the keywords with co.embed](https://txt.cohere.ai/generative-content-keyword-research/#embed-the-keywords-with-coembed)
    -   [Cluster the keywords into topics with scikit-learn](https://txt.cohere.ai/generative-content-keyword-research/#cluster-the-keywords-into-topics-with-scikit-learn)
    -   [Generate topic names with Topically](https://txt.cohere.ai/generative-content-keyword-research/#generate-topic-names-with-topically)
-   [Step 3: Generate blog post ideas for each topic](https://txt.cohere.ai/generative-content-keyword-research/#step-3-generate-blog-post-ideas-for-each-topic)
    -   [Take the top keywords from each topic](https://txt.cohere.ai/generative-content-keyword-research/#take-the-top-keywords-from-each-topic)
    -   [Create a prompt with these keywords](https://txt.cohere.ai/generative-content-keyword-research/#create-a-prompt-with-these-keywords)
    -   [Generate content ideas](https://txt.cohere.ai/generative-content-keyword-research/#generate-content-ideas)

## Introduction

While generative models have made significant strides in generating creative content ideas, most examples fail to ground these ideas in real-world search demand and trends. To fuel generative models with ideas that cut through the noise, keyword research is essential. By analyzing high-performing keywords, and trends in search queries, content creators can develop ideas tailored to what searchers want now.

Leveraging keyword research helps generate content that solves problems, addresses recent news topics, and capitalizes on current interests—ultimately producing a steady stream of impactful content. And with generative AI, we can use these keyword insights to produce content at scale.

In this blog post, we’ll build a simple Python application to generate content ideas informed by keyword research. We’ll use two Cohere endpoints, [Embed](https://cohere.ai/embed?ref=txt.cohere.ai) and [Generate](https://cohere.ai/generate?ref=txt.cohere.ai), together with other libraries such as [Topically](https://github.com/cohere-ai/sandbox-topically?ref=txt.cohere.ai) (a Cohere sandbox project) and [scikit-learn](https://scikit-learn.org/stable/?ref=txt.cohere.ai).

We’ll show snippets of the code in this article, but you can find the complete [Google Colab notebook here](https://colab.research.google.com/github/cohere-ai/notebooks/blob/main/notebooks/Fueling_Generative_Content_with_Keyword_Research.ipynb?ref=txt.cohere.ai).

![In a nutshell, three steps are involved: getting the keywords, grouping them into topics, and generating ideas from these topics.](https://lh5.googleusercontent.com/6Jcc2jq4fbSNXF0YjmIXqVFJom6hxhD89obqUieuqO146lwvImd3_A8sTcolOLj6Cmtbp4Rw7hWSLJDhrtFben1op2Kr-vjKGe7vfSXhCWBeQzLgvMB0WF4ctufWFXfrssp9YCMoydd2WjuhY6O1rxs)

_In a nutshell, three steps are involved: getting the keywords, grouping them into topics, and generating ideas from these topics._

## Step 1: Get a list of High-performing Keywords

First, we need to get a supply of high-traffic keywords for a given topic. We can get this via keyword research tools, of which are many available. We’ll use [Google Keyword Planner](https://ads.google.com/home/tools/keyword-planner/?ref=txt.cohere.ai), which is free to use.

These keyword research tools provide various information and statistics in their reports, but we’ll only need two types of information: the keywords and the search volume for these keywords.

Let’s say we are interested in generating content ideas around the topic “Remote Teams.” For this, we can use the “Discover new keywords” feature in Google Keyword Planner to get a list of high-performing keywords related to the term “Remote Teams.”

For convenience, the keyword list that we are using in this article is [available here](https://raw.githubusercontent.com/cohere-ai/notebooks/main/notebooks/data/remote_teams.csv?ref=txt.cohere.ai).

![Getting a list of high-performing keywords related to a particular search term in Google Keyword Planner](https://lh5.googleusercontent.com/P2xFX3uAwVxoW3gYCl75GQI0QGDo2z7U_zP4WjZxlZwRtkbcw2-vfmz15lJdBKrPpYP7trSDAr2o0C0uLt2mTu6DEZJvtzVhXvba5d7jw_Syztg2ynrpcQxxJqcbYESLIYLkuYjF1hX3eDqeATT6fbs)

_Getting a list of high-performing keywords related to a particular search term in Google Keyword Planner_

## Step 2: Group the Keywords into Topics

We now have a list of keywords, but this list is still raw. For example, “managing remote teams” is the top-ranking keyword in this list. But at the same time, there are many similar keywords further down in the list, such as “how to effectively manage remote teams.”

In any keyword research output, there are bound to be a lot of similar keywords such as these, which means that we need to be able to compare and distill them into broader themes.

We can do that by clustering them into topics. For this, we’ll leverage the following: Cohere’s Embed endpoint, scikit-learn, and Topically. Let’s see how we can do that.

### Embed the Keywords with co.embed

The first step is to turn each keyword into a text embedding. A text embedding is a list of numbers (also called a vector) that provides a numerical representation of the contextual meaning of the text. This enables use cases that involve comparing passages of text, such as what we are doing here with clustering. Some other use cases made possible by this are search, recommendation, and classification.

The Cohere [Embed endpoint](https://docs.cohere.ai/reference/embed?ref=txt.cohere.ai) turns a text input into a text embedding. Its usage is straightforward – we call the \`co.embed()\` method by passing a list of text inputs, and get back the corresponding embeddings.

```
import cohere
co = cohere.Client(api_key)
def embed_text(text):
 output = co.embed(
               model='large',
               texts=text)
 return output.embeddings


df = pd.read_csv('remote_teams.csv')
embeds = np.array(embed_text(df['keyword'].tolist()))
```

### Cluster the Keywords into Topics with scikit-learn

We then use these embeddings to cluster the keywords. A common term used for this exercise is “topic modeling.” Here, we can leverage scikit-learn’s [KMeans module](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html?ref=txt.cohere.ai), a machine learning algorithm for clustering.

The input to the module is the embeddings. We also need to define the number of clusters to be created. In this example, we choose four clusters, but there is no right or wrong number to use.

The implementation is as follows.

```
from sklearn.cluster import KMeans
NUM_TOPICS = 4
kmeans = KMeans(n_clusters=NUM_TOPICS, random_state=21, n_init="auto").fit(embeds)
df['topic'] = list(kmeans.labels_)
```

The output we get is the assigned topic for each keyword, ranging from 0 to 3 (as we had defined four clusters).

Here are some example topic assignments (see the right-hand column). For example, the first and last items are similar, and so they are grouped in the same topic.

| Index | Keyword | Volume | Topic |
| --- | --- | --- | --- |
| 0 | managing remote teams | 1000 | 2 |
| 1 | remote teams | 390 | 2 |
| 2 | collaboration tools for remote teams | 320 | 0 |
| 3 | online games for remote teams | 320 | 1 |
| 4 | how to manage remote teams | 260 | 2 |
| … | … | … | … |

And here’s an example list of keywords that belong to that topic (the theme is quite clearly about managing remote teams):

```
'keywords': 'managing remote teams, remote teams, how to manage remote teams, leading remote teams, managing remote teams best practices, remote teams best practices, ... [truncated for brevity]
```

### Generate Topic Names with Topically

We now have each keyword assigned to a topic, but we don’t have a topic name other than the 0-4 integers, which are not very informative. It would be nice to get a representative name for each topic, given these keywords.

For this, we can use the [Topically](https://github.com/cohere-ai/sandbox-topically?ref=txt.cohere.ai) package, released as part of the Cohere Sandbox project. The package takes in a list of text within a cluster and uses Cohere’s [Generate](https://docs.cohere.ai/reference/generate?ref=txt.cohere.ai) endpoint to generate a topic name for that cluster.

The usage is as follows.

```
from topically import Topically
app = Topically(cohere_api_key)
df['topic_name'], _ = app.name_topics((df['keyword'], df['topic']))
```

And here is the same table updated with the topic names (see the right-hand column). The topic 0 has been aptly named “How to lead remote teams.”

| Index | Keyword | Volume | Topic | Topic\_name |
| --- | --- | --- | --- | --- |
| 0 | managing remote teams | 1000 | 2 | Managing remote teams |
| 1 | remote teams | 390 | 2 | Managing remote teams |
| 2 | collaboration tools for remote teams | 320 | 0 | Collaboration tools for remote teams |
| 3 | online games for remote teams | 320 | 1 | Virtual games for remote teams |
| 4 | how to manage remote teams | 260 | 2 | Managing remote teams |
| … | … | … | … | … |

And here are all the four generated topic names:

-   Collaboration tools for remote teams
-   Virtual games for remote teams
-   Managing remote teams
-   remote team building activities

## Step 3: Generate Blog Post Ideas for Each Topic

Now that we have the keywords nicely grouped into topics, we can proceed to generate the content ideas.

### Take the Top Keywords from Each Topic

Depending on how many keywords you imported, the list can get very long. So, it’s probably a good idea to generate content ideas not based on all the keywords, but only on the best-performing ones.

So, here we can implement a filter to take just the top N keywords from each topic, sorted by the search volume. In our case, we use 10.

```
TOP_N = 10
top_keywords = (df.groupby('topic')
                       .apply(lambda x: x.nlargest(TOP_N, 'volume'))
                       .reset_index(drop=True))
```

Here are the top keywords for each topic in full.

![The top keywords for each topic in full.](https://txt.cohere.ai/content/images/2023/03/top-keywords.png)

The top keywords for each topic in full.

### Create a Prompt with These Keywords

Next, we use the [Generate endpoint](https://docs.cohere.ai/reference/generate?ref=txt.cohere.ai) to produce the content ideas. To do that, we first need to set up the prompt to the text generation model.

The prompt looks as follows. For each topic, we feed the keywords into the prompt and append an instruction. The instruction consists of, first, telling the model to generate three blog post ideas (an arbitrary choice) and, second, showing the format how the output should be generated (the blog title and its corresponding abstract).

```
{keywords}


The above is a list of high-traffic keywords obtained from a keyword research tool. 
Suggest three blog post ideas that are highly relevant to these keywords. 
For each idea, write a one paragraph abstract about the topic. 
Use this format:
Blog title: <text>
Abstract: <text>
```

### Generate Content Ideas

Next, we create a function to generate blog post ideas. It takes in a string of keywords, calls the Generate endpoint, and returns the generated text. There are a few settings we define here, which are the `model`, `max_tokens`, and `temperature`. If you’d like to learn more about them, visit the [API reference page](https://docs.cohere.ai/reference/generate?ref=txt.cohere.ai).

```
def generate_blog_ideas(keywords):
  prompt = f"{keywords}\n\nThe above is a list of …[truncated for brevity]
  response = co.generate(
    model='command-xlarge-nightly',
    prompt = prompt,
    max_tokens=300,
    temperature=0.9)
  return response.generations[0].text
```

And that’s it! We’ve got a list of content ideas informed by what readers actually want to read.

Let’s look at a sample topic and get a feel of what the output looks like. Below are the blog post ideas generated for the topic: “Managing remote teams.”

![Blog post ideas generated for the topic: “Managing remote teams.”](https://txt.cohere.ai/content/images/2023/03/topic-managing-remote-teams.png)

Blog post ideas generated for the topic: “Managing remote teams.”

And here’s another example for the topic: “Collaboration tools for remote teams.”

![Blog post ideas generated for the topic: “Collaboration tools for remote teams.”](https://lh6.googleusercontent.com/BK13a7hL6DFZpGa8wWWVLb71J1CEZxUN9wd6W9dKQcpQgLBgZyCC5CTk6Zfv9H50Uy8CY15Ds-305JM7MtmYGPXan-BSVip_7gff5WWykKwinK7y1lCCZ-9Z66qmATArzhz6BXVS79Dve2JMs6y5NmM)

Blog post ideas generated for the topic: “Collaboration tools for remote teams.”

## Final Thoughts

In this article, we looked at how to generate content ideas grounded on actual user demand and recency; all this is informed by keyword research results. We can even go on to generate a complete blog post given these abstracts, but that deserves an article of its own, so let’s leave that for another time!

This was possible via a combination of two Cohere endpoints: Embed and Generate. And quite likely, it will not be the last time you’ll see this combo in the wild. What one provides perfectly complements the other – text understanding and text generation, respectively – making this combination extremely useful in real-world applications.

Get started by creating a [free account on Cohere.](https://dashboard.cohere.ai/welcome/register?ref=txt.cohere.ai)
