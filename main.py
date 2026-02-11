from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

# Original: react_prompt = hub.pull("hwchase17/react")
# Defined inline because the EU LangSmith endpoint cannot fetch public prompts from the US hub.
react_prompt = PromptTemplate.from_template(
    """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

tools = [TavilySearch()]
llm = ChatOpenAI(temperature=0, model="gpt-4.1-nano")
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def main():
    print("Hello from langchain-course!")
    result = agent_executor.invoke(
        {"input": "Search for three job postings for an ai engineer using langchain in the bay area on linkedin and list their details."}
    )
    print(result)

if __name__ == "__main__":
    main()
