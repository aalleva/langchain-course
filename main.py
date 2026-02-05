from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(temperature=0, model="gpt-4.1-nano")

def main():
    print("Hello from langchain-course!")
    information = """
    Elon Reeve Musk FRS (born June 28, 1971) is a South African-Canadian-American businessman and entrepreneur. He moved to Canada and later became a U.S. citizen. He was the Commissioner of the Department of Government Efficiency for a few months in 2025 during the second Donald Trump administration.[1] He became rich through several technology projects, including an online finance company which merged with PayPal in the year 2000.[2] Musk has been the wealthiest person in the world since 2021, according to both the Bloomberg Billionaires Index and Forbes's real-time billionaires list; with an estimated net worth of US$500 billion, as recently as October 2025.[3][4] In 2021, he was Time Person of the Year.[5]

    Elon is the current CEO & Chief Product Architect of Tesla, Inc., a company that makes electric vehicles. He is also the CEO & CTO of SpaceX, an aerospace company. In 2022, he became the owner of the social media site Twitter which he later gave a new name known as X after buying it for USD $44 billion.[6]

    In November 2024, U.S. President-elect Donald Trump said that Musk would become a leader of the Department of Government Efficiency (DOGE)[7] alongside Vivek Ramaswamy but Ramaswamy stepped down before the Department was created.[8]
    """
    
    summary_template = """
    Given the information {information} about a person I want to create: 
    1. A short summary
    2. Two interesting facts about them
    """
    summary_prompt = PromptTemplate(template=summary_template, input_variables=["information"])
    summary_chain = summary_prompt | llm
    response = summary_chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
