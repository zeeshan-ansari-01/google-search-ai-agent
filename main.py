print("Google Search AI Agent Started!")

from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import Tool

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

search = GoogleSerperAPIWrapper()

search_tool = Tool(
    name="google_search",
    func=search.run,
    description="Search Google for current and latest information."
)

agent = create_agent(
    model=model,
    tools=[search_tool],
    system_prompt="You are an AI agent that can search Google to answer user questions."
)

while True:
    query = input("User: ")

    if query.lower() == "quit":
        print("Good Bye")
        break

    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"recursion_limit": 25}
    )

    print("AI:", response["messages"][-1].content)