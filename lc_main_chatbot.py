from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from lc_utils import *
from lc_vector_search import *
import os

st.subheader("Niche AI Bot")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

openai.api_key = os.getenv('OPENAI_API_KEY')
if 'None' == os.getenv('OPENAI_API_KEY'):
    try:
        openai.api_key = st.secrets['OPENAI_API_KEY']
    except:
        print("error reading secrets")

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai.api_key)

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

with textcontainer:

    if 'something' not in st.session_state:
        st.session_state.something = ''

    def submit():
        st.session_state.something = st.session_state.input
        st.session_state.input = ''

    st.text_input("Query: ", key="input", on_change=submit)
    query = st.session_state.something
    if query:
        with st.spinner("processing..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            refined_query = query#query_refiner(conversation_string, query)
            #st.subheader("Refined Query:")
            #st.write(refined_query)
            context = find_match(refined_query)
            print(context)  
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i), avatar_style="identicon",seed="Aneka")
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user', avatar_style="bottts-neutral",seed="Oliver")
