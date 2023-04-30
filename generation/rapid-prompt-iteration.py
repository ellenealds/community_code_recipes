import streamlit as st
from langchain.llms import Cohere
import pandas as pd

# Adjust this global instantiation of the Cohere api to set parameters across calls

llm = Cohere()

if "output" not in st.session_state:
    st.session_state["output"] = ""

if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame(columns = ["input_prompt", "completion"])

st.title("Rapid Prompt Iteration with Streamlit, Cohere, and Langchain")


prompt = st.text_input(label = "Write your prompt here!", value = "")

def return_completion(prompt):
    st.session_state.output = llm(prompt)

gen_result = st.button("Generate Completion")

if prompt != "" and gen_result:
    result = llm(prompt)
    result_dict = {
        "input_prompt": prompt,
        "completion": result
    }
    st.session_state["history"] = pd.concat([st.session_state["history"], pd.DataFrame([result_dict])],
    ignore_index=True)

st.write(st.session_state.history)


