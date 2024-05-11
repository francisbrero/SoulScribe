import streamlit as st
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import sentence_transformers
from datetime import datetime
import os
from utils import add_to_file, detect_person_mention

# Write history (list) to a file and overwrite the file with the history
def write_history_to_file(history):
    with open(f"data/history.txt", "w") as f:
        for message in history:
            f.write(message["role"] + ": " + message["content"] + "\n")

# Point to the local server
client = OpenAI(base_url="http://192.168.4.72:1234/v1", api_key="lm-studio")

embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chromadb", embedding_function=embedding_function)

# Add a sidebar to the app
st.sidebar.title("Pick a Model")
# Create a select box to choose the model for the chatbot
model = st.sidebar.selectbox("Select a Model", ["TheBloke/Llama-2-13B-chat-GGUF", "bartowski/Llama3-ChatQA-1.5-70B-GGUF"])

history = [
    {"role": "system", "content": """You are a mental health chatbot. 
     You are here to help people with their mental health. You ask questions to help the user understand their feelings and emotions. 
     You can also provide resources and advice. You are not a therapist, but you can help people find the right resources. 
     Use the following instructions to guide your conversation:
     - Keep your questions short and simple. Ask open-ended questions to encourage the user to share more about their feelings.
     - Do not spend too much time on one topic. Move on to another topic if you've already asked 4 questions on the same topic. This helps keep the conversation engaging.
     - Do not start by rephrasing of the user's input. Instead, ask questions or provide information that can help the user explore their feelings.
     - Use as few words as possible. Keep your responses concise and to the point.
     - Get the user to change topics quickly and ask them if they want to stick to the current topic.
     - If the user mentions a person you know about using get_person_info, gather more information about their relationship and use that information to be more specific in your questions.
     - Use your past conversations to better guide the conversation."""},
    {"role": "user", "content": "Hello. I'm Francis and I'm ready to start our session."},
    {"role": "assistant", "content": "Great! Let's start. How are you feeling today?"},
]


#### Sidebar
# Add a side bar to the app and display the names of the people the bot knows about
st.sidebar.title("People I Know About")
# get the list of people the bot knows about from the people folder in the data folder
people = [person.split(".")[0] for person in os.listdir("data/people")]
# Display the names of the people the bot knows about in the side bar
for person in people:
    st.sidebar.write(person)

# Add a text input to the side bar to enter the name of the interesting person the bot should know about
person_name = st.sidebar.text_input("Enter the name of the person you want to know more about:")
# Create a button to add the person to the list of people the bot knows about
if st.sidebar.button("Add Person"):
    # Add the person to the list of people the bot knows about
    people.append(person_name)
    # Create a file for the person in the people folder in the data folder
    with open(f"data/people/{person_name}.txt", "w") as f:
        f.write("")

# Add a button to reset the session state
if st.sidebar.button("Reset Session"):
    st.session_state.messages = []

#### Main Chatbot
st.title("Mental Health Chatbot")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How are you feeling?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    add_to_file("Francis: " + prompt)
    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # detect if the user mentions a person the chatbot knows about
    person_info = detect_person_mention(prompt)
    if person_info:
        history.append({"role": "assistant", "content": "here's more context about this person: " + person_info})

    with st.chat_message("assistant"):
        # create a stream using langchain chat completions
        stream = client.chat.completions.create(
            # model="TheBloke/Llama-2-13B-chat-GGUF", # this field is currently unused
            model="bartowski/Llama3-ChatQA-1.5-70B-GGUF", # this field is currently unused
            messages=history,
            temperature=0.0,
            stream=True,
            max_tokens=-1,
            tools=["get_person_info", "store_person_info"],
            stop=["<|im_end|>", "<|Im_end|>"],
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
    history.append({"role": "assistant", "content": response})
    write_history_to_file(history)