from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

@tool
def friend_suggestion(question:str):
    """
        use this tool for emotion support, encouragment,asual conversation, friendadvice
        Answer the question as a friend
    """
    return "Answer the user's question from the perspective of a supportive friend."

@tool
def life_coach(question:str):
    """
        use this tool for lifestyle, guidance,motivation, goals, career planning
        Answer the question as a lifecoach
    """
    return "Answer the user's question from the perspective of a life coach."

agent = create_agent(
    model="openai:gpt-4",tools=[friend_suggestion,life_coach]
    
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "how to make friends?"}]}
)
print(result["messages"][-1].content)
