import os

from typing import TypedDict,List,Dict,Union

from langchain_core.messages import HumanMessage, AIMessage

from langgraph.graph import StateGraph,START,END

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()


class Agent(TypedDict):
    messages: List[Union[HumanMessage,AIMessage]]   

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=1.0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


def process(state: Agent)-> Agent:
    """this node will solve the request input"""


    response = model.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response.content[0]["text"]))
    print(f"\n AI : {response.content[0]["text"]} \n")
    return state


graph = StateGraph(Agent)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)

agent = graph.compile()


convo_history = []

user_input = input("enter: ")

while user_input != "exit":
    convo_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages":convo_history})
    # print(result["messages"][-1].content )
    convo_history = result["messages"]
    user_input = input("enter: ")



with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    
    for message in convo_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")

