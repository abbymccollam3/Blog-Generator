import streamlit as st # python library for creating web applications
from langchain.prompts import PromptTemplate # manages prompts/workflows for LLMs
from langchain.llms import ctransformers # helps interact with LLama2

## function to get response from LLama2 model

def getLLamaResponse(input_text, no_words, blog_style):

    # LLama2 model that we downloaded
    llm=ctransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type='llama',
                      config={'max_new_tokens': 256,
                              'temperature': 0.01})
    
    # Prompt Template
    
    template= """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
    """

    prompt = PromptTemplate(input_variables=["style","text","n_words"],
                            template=template)
    
    # Generate response from LLama2 model
    response=llm(prompt.format(style=blog_style, text=input_text, n_words=no_words))

    print(response)
    return response

# configures appearance and config for Streamlit app
st.set_page_config(page_title="Generate Blogs",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

# creating 2 more columns

col1, col2 = st.columns([5, 5])

# Number of words for blog post
with col1:
    no_words=st.text_input("Number of Words")
# for who you are writing the blog
with col2:
    blog_style=st.selectbox("Writing the blog for", 
                            ('Researchers', 'Data Scientists', 'Common People'), index=0)
    
# click and takes info
submit = st.button("Generate")

# Final response
if submit:
    st.write(getLLamaResponse(input_text, no_words, blog_style))