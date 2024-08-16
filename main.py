from crewai import Agent, Task, Crew
from langchain_ollama import ChatOllama
import os
os.environ["OPENAI_API_KEY"] = "NA"

llm = ChatOllama(
    model = "llama3.1",
    base_url = "http://localhost:11434")

general_agent = Agent(role = "Stock Market Analyst",
                      goal = """Look for Tesla, Microsoft stock prices and Provide the latest stock prices of Tesla and Microsoft""",
                      backstory = """You are an excellent stock market specailist that has been working in the industry for over 10 years. You have a deep understanding of the stock market and have been following Tesla and Microsoft stocks for a long time.""",
                      allow_delegation = False,
                      verbose = True,
                      llm = llm)

task = Task(description="""What is current stock price of Tesla and Microsoft?""",
             agent = general_agent,
             expected_output="""The current stock price of Tesla is $700 and Microsoft is $300""")

crew = Crew(
            agents=[general_agent],
            tasks=[task],
            verbose=True
        )

result = crew.kickoff()

print(result)