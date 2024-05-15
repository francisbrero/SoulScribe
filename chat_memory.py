from autogen import UserProxyAgent, config_list_from_json
from autogen.agentchat.contrib.capabilities.teachability import Teachability
from autogen import ConversableAgent  # As an example
from dotenv import load_dotenv
from utils import get_random_question

load_dotenv()

# Load LLM inference endpoints from an env variable or a file
# See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample
# filter_dict = {"model": ["gpt-4"]} 
filter_dict = {"model": ["TheBloke/Llama-2-13B-chat-GGUF"]} # let's use our local model
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST", filter_dict=filter_dict)
llm_config={
        "config_list": config_list, 
        "timeout": 120,
        # "stream": True, # Enable, disable streaming (defaults to False)
        }

instructions = """You are Karl, a mental health chatbot.
     You are here to help people with their mental health. You ask questions to help the user understand their feelings and emotions. 
     You can also provide resources and advice. You are not a therapist, but you can help people find the right resources. 
     Use the following instructions to guide your conversation:
     - Keep your questions short and simple. Ask open-ended questions to encourage the user to share more about their feelings.
     - Do not spend too much time on one topic. Move on to another topic if you've already asked 4 questions on the same topic. This helps keep the conversation engaging.
     - Do not start by rephrasing of the user's input. Instead, ask questions or provide information that can help the user explore their feelings.
     - Use as few words as possible. Keep your responses concise and to the point.
     - Get the user to change topics quickly and ask them if they want to stick to the current topic.
     - If the user mentions a person you know about using get_person_info, gather more information about their relationship and use that information to be more specific in your questions.
     - Use your past conversations to better guide the conversation."""

# Start by instantiating any agent that inherits from ConversableAgent, which we use directly here for simplicity.
teachable_agent = ConversableAgent(
    name="Karl",  # The name can be anything.
    llm_config=llm_config,
    # system_message=instructions,
)

# Instantiate a Teachability object. Its parameters are all optional.
teachability = Teachability(
    reset_db=False,  # Use True to force-reset the memo DB, and False to use an existing DB.
    path_to_db_dir="./data/memory/teachability_db",
    verbosity=3,  # 0 (default) for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
    recall_threshold=0.5,  # The maximum distance for retrieved memos, where 0.0 is exact match. Default 1.5. Larger values allow more (but less relevant) memos to be recalled.
)

# Now add teachability to the agent.
teachability.add_to_agent(teachable_agent)

# For this test, create a user proxy agent as usual.
user = UserProxyAgent("user", human_input_mode="ALWAYS")

# Get a random question from the seed_questions.txt file
question = get_random_question()

# This function will return once the user types 'exit'.
teachable_agent.initiate_chat(user, message="Hi, I'm Karl, your mental health coach! " + question)