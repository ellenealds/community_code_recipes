from concurrent.futures import ThreadPoolExecutor
from cohere import CohereAPI

def generate_answer(co: CohereAPI, question: str, paragraph: str) -> str:
    """Generate an answer to a question using a given paragraph using the Cohere API.

    Args:
        co (CohereAPI): A CohereAPI object for generating answers.
        question (str): The question to be answered.
        paragraph (str): The paragraph to be used as context.

    Returns:
        str: The generated answer.
    """
    response = co.generate(model='command-xlarge-20221108',
                            prompt=f'''Paragraph:{paragraph}\n\n
                                       Answer the question using this paragraph.\n\n
                                       Question: {question}\nAnswer:''',
                            max_tokens=100,
                            temperature=0.4)
    return response.generations[0].text


def generate_better_answer(co: CohereAPI, question: str, answers: List[str]) -> str:
    """Generate a better answer using a list of candidate answers and a given question using the Cohere API.

    Args:
        co (CohereAPI): A CohereAPI object for generating answers.
        question (str): The question to be answered.
        answers (List[str]): A list of candidate answers.

    Returns:
        str: The generated answer.
    """
    response = co.generate(model='command-xlarge-20221108',
                            prompt=f'''Answers:{answers}\n\n
                                       Question: {question}\n\n
                                       Generate a new answer that uses the best answers 
                                       and makes reference to the question.''',
                            max_tokens=100,
                            temperature=0.4)
    return response.generations[0].text


def display_results(query: str, results: pd.DataFrame, co: CohereAPI) -> str:
    """Generate answers for a given query and display the results.

    Args:
        query (str): The query to be used to generate answers.
        results (pd.DataFrame): A pandas dataframe containing the text to be used as context for generating answers.
        co (CohereAPI): A CohereAPI object for generating answers.

    Returns:
        str: The generated answer.
    """
    # 1. Run generate_answer function to generate answers
    # for each row in the dataframe, generate an answer concurrently
    with ThreadPoolExecutor(max_workers=1) as executor:
        results['answer'] = list(executor.map(generate_answer,
                                              [co]*len(results),
                                              [query]*len(results),
                                              results['text']))

    # 2. Run generate_better_answer function to generate a better answer
    answers = results['answer'].tolist()
    return generate_better_answer(co, query, answers)
