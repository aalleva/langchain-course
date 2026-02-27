from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from schemas import AgentResponse

load_dotenv()

tools = [TavilySearch()]
model = ChatOpenAI(temperature=0, model="gpt-4")

agent = create_agent(
    model=model,
    tools=tools,
    response_format=AgentResponse,
    system_prompt=(
        "You are a helpful search assistant. Use the available tools to find information. "
        "When answering, provide a clear answer and list the source URLs you used."
    ),
)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Search for three job postings for an ai engineer "
                    "using langchain in the bay area on linkedin and list their details.",
                }
            ]
        }
    )
    if "structured_response" in result:
        print(result["structured_response"])
    else:
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
