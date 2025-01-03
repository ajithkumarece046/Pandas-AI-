from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
from pandasai import SmartDatalake
from pandasai.llm.azure_openai import AzureOpenAI
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
load_dotenv()

# Load Azure OpenAI configuration from environment variables
API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')  
ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT') 
DEPLOYMENT_NAME = os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME') 
API_VERSION =os.environ.get('API_VERSION') 

# Initialize AzureOpenAI
llm = AzureOpenAI(api_key=API_KEY, api_url=ENDPOINT, deployment_name=DEPLOYMENT_NAME, api_version=API_VERSION)

# Streamlit UI
st.title("Prompt Driven Analysis with PandasAI (Azure OpenAI)")

uploaded_file = st.file_uploader("Upload a file you want to analyse (CSV only)", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write(df.head(3))

    prompt = st.text_area("Enter your prompt:")
    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response.."):
                data=SmartDatalake(df, config={"llm" :llm})
                response = data.chat(prompt)
                st.write(response)
        else:
            st.write("Please enter a prompt!")
