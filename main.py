from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()

# Original: react_prompt = hub.pull("hwchase17/react")
# Defined inline because the EU LangSmith endpoint cannot fetch public prompts from the US hub.

react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
).partial(format_instructions="")

tools = [TavilySearch()]
llm = ChatOpenAI(temperature=0, model="gpt-4")
structured_llm = llm.with_structured_output(AgentResponse)
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt_with_format_instructions)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x: x["output"])
chain = agent_executor | extract_output | structured_llm

def main():
    print("Hello from langchain-course!")
    result = chain.invoke(
        {
            "input": "Search for three job postings for an ai engineer using langchain in the bay area on linkedin and list their details."
        }
    )
    print(result)

if __name__ == "__main__":
    main()
