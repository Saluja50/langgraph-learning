from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages  #reducer fnc
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from langchain_core.tools import tool


load_dotenv()

#annotated - provides additional context without affecting the type

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]



@tool
def add(a:int , b:int):
    """ this is a addition function that adds 2 number"""

    return a + b

@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    return a * b

tools = [add, subtract, multiply]

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=1.0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
).bind_tools(tools)


def model_call(state:AgentState)->AgentState:
    system_prompt = SystemMessage(content="You are a helpful assistant.")

    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}




def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "loop"



graph = StateGraph(AgentState)

graph.add_node("model_call",model_call)


tool_node = ToolNode(tools)

graph.add_node("tools",tool_node)

graph.set_entry_point("model_call")

graph.add_conditional_edges(
    "model_call",
    should_continue,
    {
        "loop": "tools",
        "end": END,
    }

)


graph.add_edge("tools","model_call")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40 + 10 and multiply the result by 3 .  and also tell me a fun fact about pussy")]}
print_stream(app.stream(inputs, stream_mode="values"))