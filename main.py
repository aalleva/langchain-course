from dotenv import load_dotenv

from agent import create_search_agent

load_dotenv()


def main() -> None:
    print("Hello from langchain-course!")
    agent = create_search_agent()
    result = agent.invoke({
        "messages": [{"role": "user", "content": "What is LangChain?"}],
    })
    last_message = result["messages"][-1]
    print(last_message.content)


if __name__ == "__main__":
    main()
