from autogen import UserProxyAgent, config_list_from_json
from autogen.agentchat.contrib.capabilities.teachability import Teachability
from autogen.agentchat.contrib.text_analyzer_agent import TextAnalyzerAgent
from autogen import ConversableAgent
from custom_teachable_agent import CustomTeachableAgent
from dotenv import load_dotenv
from utils import get_random_question

load_dotenv()

# Load LLM inference endpoints from an env variable or a file
# See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample
# filter_dict = {"model": ["gpt-4"]} 
filter_dict = {"model": ["TheBloke/Llama-2-13B-chat-GGUF"]} # let's use our local model 
# filter_dict = {"model": ["bartowski/Llama3-ChatQA-1.5-70B-GGUF"]} # let's use our local model 
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST", filter_dict=filter_dict)
llm_config={
        "config_list": config_list, 
        "timeout": 120,
        # "stream": True, # Enable, disable streaming (defaults to False)
        }

teachable_agent_instructions = """You are Karl, a mental health chatbot.
     You are here to help people with their mental health. You ask questions to help the user understand their feelings and emotions. 
     You can also provide resources and advice. You are not a therapist, but you can help people find the right resources. 
     Your role involves the following key interactiosn with the user:
     1. Information Gathering: Engage in conversations to collect data about people (relationship, personality traits, memories), work, hobbies, and any specific life story.
     2. Data Analysis and Storage: Use your analysis capabilities (via TextAnalyzerAgent) to determine what information from the conversation is relevant to store. This includes personal details, relationship information, and question-answer pairs.
     3. Data Retrieval: When a user makes inquiries, retrieve relevant information from your database. This involves using past conversation details or context entries to provide helpful responses.
     4. User Interaction: Maintain a conversational tone, guiding users to provide information or ask questions relevant to mental health coaching activities. Be proactive in clarifying details and asking follow-up questions for comprehensive data collection.
     Always consider the context of the user's queries and comments for effective context management. Your interactions should be tailored to build and maintain strong, informative user relationships. You will work closely with TextAnalyzerAgent and your MemoStore to ensure accurate data processing and storage."""

# Start by instantiating any agent that inherits from ConversableAgent, which we use directly here for simplicity.
# teachable_agent = ConversableAgent(
teachable_agent = CustomTeachableAgent(
    name="Karl",  # The name can be anything.
    llm_config=llm_config,
    system_message=teachable_agent_instructions,
)

# Now we customize the part where the agent determines if the agent should store the information
text_analyzer_system_message = """You are an expert in text analysis.
The user will give you TEXT to analyze.
The user will give you analysis INSTRUCTIONS copied twice, at both the beginning and the end.
You will follow these INSTRUCTIONS in analyzing the TEXT, then give the results of your expert analysis in the format requested.
Most importantly, you will care about the user's relationships with people, and use that information to be more specific in your questions.
If TEXT relates to a person you don't know about, ask the user for more information about the relationship.
If TEXT doesn't relate to the users' relationships, respond with no.
If TEXT is asking you to move on to another toic, respond with no."""



teachable_agent.analyzer = TextAnalyzerAgent(
    system_message=text_analyzer_system_message,
    llm_config=llm_config,
)

# Instantiate a Teachability object. Its parameters are all optional.
teachability = Teachability(
    reset_db=False,  # Use True to force-reset the memo DB, and False to use an existing DB.
    path_to_db_dir="./data/memory/teachability_db",
    verbosity=1,  # 0 (default) for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
    recall_threshold=3,  # The maximum distance for retrieved memos, where 0.0 is exact match. Default 1.5. Larger values allow more (but less relevant) memos to be recalled.
)

# Now add teachability to the agent.
teachability.add_to_agent(teachable_agent)

# For this test, create a user proxy agent as usual.
user = UserProxyAgent("user", human_input_mode="ALWAYS")

# Get a random question from the seed_questions.txt file
question = get_random_question()

# This function will return once the user types 'exit'.
teachable_agent.initiate_chat(user, message="Hi, I'm Karl, your mental health coach! " + question)