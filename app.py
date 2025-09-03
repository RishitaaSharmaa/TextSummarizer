# This app uses stuff chain document technique for summarization

import streamlit as st
import validators
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader,YoutubeLoader

st.set_page_config(page_title="YT Video and Website Summarizer")
st.title("Youtube Video or Website Summarizer")

with st.sidebar:
    groq_api_key=st.text_input("Groq api key",value="",type="password")

url=st.text_input("URL",label_visibility="collapsed")

llm=ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct",api_key=groq_api_key)

prompt_template='''
Provide a summary of the following content in 300 words:
Content:{text}

'''
prompt=PromptTemplate(input_variables=["text"],template=prompt_template)

if st.button("Summarize the Content"):
    if not groq_api_key.strip() or not url.strip():
        st.error("Please provide the information")

    elif not validators.url(url):
        st.error("Please enter a valid url!")

    else:
        try:
            with st.spinner("Waiting..."):
                if "youtube.com" in url:
                    loader=YoutubeLoader.from_youtube_url(url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[url],ssl_verify=False,headers={"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    docs=loader.load()
                    chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                    output_summary=chain.run(docs)
                    st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception:{e}")

