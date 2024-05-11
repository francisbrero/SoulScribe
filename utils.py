import random
from datetime import datetime
import chromadb
from langchain_openai import ChatOpenAI
import os

# Function to Initialize the Chroma Vector Database
def initialize_chroma_db():
    # Initialize the Chroma Vector Database
    chroma_client = chromadb.Client()
    # change the chromadb embedding model
    EMBEDDING_MODEL = "text-embedding-3-large"
    collection = chroma_client.get_or_create_collection(db_name="mental_health", collection_name="chat_history", embedding_model=EMBEDDING_MODEL)

    return collection

# Add text to today's file in the data folder
def add_to_file(text):
    with open(f"data/{datetime.now().strftime('%Y-%m-%d')}.txt", "a") as f:
        f.write(text + "\n")

# Write history (list) to a file and overwrite the file with the history
def write_history_to_file(history):
    with open(f"data/history.txt", "w") as f:
        for message in history:
            f.write(message)


# Store information about a person mentioned in the chat in the relevant file under the data folder
def store_person_info(person_info):
    # Store the person's information in the person's file
    with open(f"data/people/{person_info['name']}.txt", "a") as f:
        # the format should be date, time, and the information
        f.write(f"{person_info['info']}\n")


# Create a function for the LLM to retrieve information about a person
def get_person_info(person_name):
    # Retrieve the person's information from the person's file
    with open(f"data/people/{person_name}.txt", "r") as f:
        person_info = f.read()
    return person_info


# Detect from the user's input if they mention a person the chatbot knows about. If they do, store the information about the person in the relevant file and return the information about the person
def detect_person_mention(user_input):
    # Get the names of the people the chatbot knows about
    people = [person.split(".")[0] for person in os.listdir("data/people")]
    # Check if the user mentions a person the chatbot knows about
    for person in people:
        if person.lower() in user_input.lower():
            # Store the information about the person in the relevant file
            store_person_info({"name": person, "info": user_input})
            # Return the information about the person
            return get_person_info(person)
    return None
