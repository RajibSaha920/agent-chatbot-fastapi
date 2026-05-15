from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_core.messages.ai import AIMessage


from dotenv import load_dotenv
#import streamlit as st
import os


load_dotenv()

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

#model = ChatGroq(model="openai/gpt-oss-120b")
#"act as an AI chatbot who is smart and friendly."

def get_response_from_ai_agent(llm_id,query,allow_search,system_prompy,provider):

    if provider=="Groq":
        llm = ChatGroq(model=llm_id)
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=llm_id)

    tools = [TavilySearch(max_results=2)] if allow_search else []
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompy
    )
    query =  query
    state = {"messages":query}
    response = agent.invoke(state)
    #messages= response.get("messages")
    messages = response.get("messages", [])
    ai_messages = [message.content  for message in messages if isinstance(message, AIMessage)]

    return ai_messages[-1] if ai_messages else "No response generated"
