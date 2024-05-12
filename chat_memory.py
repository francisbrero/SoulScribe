from autogen import UserProxyAgent, config_list_from_json
from autogen.agentchat.contrib.capabilities.teachability import Teachability
from autogen import ConversableAgent  # As an example
from dotenv import load_dotenv

load_dotenv()

# Load LLM inference endpoints from an env variable or a file
# See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample
# filter_dict = {"model": ["gpt-4"]} 
filter_dict = {"model": ["TheBloke/Llama-2-13B-chat-GGUF"]} # let's use our local model
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST", filter_dict=filter_dict)
llm_config={"config_list": config_list, "timeout": 120}


# Start by instantiating any agent that inherits from ConversableAgent, which we use directly here for simplicity.
teachable_agent = ConversableAgent(
    name="SoulScribe",  # The name can be anything.
    llm_config=llm_config
)

# Instantiate a Teachability object. Its parameters are all optional.
teachability = Teachability(
    reset_db=False,  # Use True to force-reset the memo DB, and False to use an existing DB.
    path_to_db_dir="./data/memory/teachability_db" 
)

# Now add teachability to the agent.
teachability.add_to_agent(teachable_agent)

# For this test, create a user proxy agent as usual.
user = UserProxyAgent("user", human_input_mode="ALWAYS")

# This function will return once the user types 'exit'.
teachable_agent.initiate_chat(user, message="Hi, I'm a teachable user assistant! What's on your mind?")