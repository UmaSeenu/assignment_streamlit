from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

agent = create_agent(
    model="openai:gpt-4"
    
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is langchain?"}]}
)
print(result)
