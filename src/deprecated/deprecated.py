import openai
import os
# from memgpt.autogen.memgpt_agent import create_autogen_memgpt_agent

#os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]
#config_list = [
#    {
#        "model": "gpt-4-1106-preview",
#        "api_key": "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",
#    },
#]
print(openai.api_key)

#llm_config = {"config_list": config_list, "seed": 42}

# The user agent
#user_proxy = autogen.UserProxyAgent(
#    name="User_proxy",
#    system_message="A human admin.",
#    code_execution_config={"last_n_messages": 2, "work_dir": "groupchat"},
#    human_input_mode="TERMINATE",  # needed?
#)

# The agent playing the role of the project manager (PM)
#scientist = autogen.AssistantAgent(
#    name="ScientistGPT",
#    system_message="You are a super human AI scientist specialized in curation of ome xml files. Specifically you recieve unstructured meta data of an image and your supposed to return the respective ome xml file.",
#    llm_config=llm_config,
#)



"""
# This MemGPT agent will have all the benefits of MemGPT, ie persistent memory, etc.
MemGPT_SeqScientist = create_autogen_memgpt_agent(
    "MemGPT_SeqScientist",
    persona_description="I am a world class scientist, specialized in synthetic biology, specifically plasmid design one BASE SEQUENCE level. You work well with MemGPT_GeneScientist as he provides the general structure for your sequences.",
    user_description=f"You are MemGPT_scientist a world class scientist specialized in plasmid design."
                     f"You answer on point, concise and correctly."
                     f"You are participating in a group chat with the user ({user_proxy.name}) and the project manager"
                     f"of the department, ({pm.name}). Your colleague, MemGPT_GeneScientist, is also participating in "
                     f"the chat, and is specialized in plasmid design on GENE level.",
    # extra options
    # interface_kwargs={"debug": True},
)
"""


# Initialize the group chat between the user and two LLM agents (PM and coder)
groupchat = autogen.GroupChat(agents=[scientist, user_proxy], messages=[], max_round=15)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

path = "/ground_truth/raw_Metadata_Image8.txt"
with open(path, "r") as f:
    data = f.read()
    print(data)
# Begin the group chat with a message from the user
user_proxy.initiate_chat(
    manager,
    message=f"Please generate the ome xml for the following metadata: \n {data}",

)