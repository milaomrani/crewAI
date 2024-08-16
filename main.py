# from crewai import Agent, Task, Crew
# from langchain_ollama import ChatOllama
# import yfinance as yf
# import os

# os.environ["OPENAI_API_KEY"] = "NA"

# llm = ChatOllama(
#     model="llama3.1",
#     base_url="http://localhost:11434")

# def get_stock_price(ticker):
#     stock = yf.Ticker(ticker)
#     current_price = stock.info.get('regularMarketPrice') or stock.info.get('previousClose')
#     if current_price is None:
#         raise KeyError(f"Could not find a valid price for ticker {ticker}")
#     return current_price



# class StockPriceAgent(Agent):
#     def fetch_stock_prices(self, tickers):
#         prices = {}
#         for ticker in tickers:
#             price = get_stock_price(ticker)
#             prices[ticker] = price
#         return prices

# general_agent = StockPriceAgent(
#     role="Stock Market Analyst",
#     goal="Provide the latest stock prices of Tesla and Microsoft",
#     backstory="You are an excellent stock market specialist with over 10 years of experience. You have a deep understanding of the stock market and have been following Tesla and Microsoft stocks for a long time.",
#     allow_delegation=False,
#     verbose=True,
#     llm=llm,
#     max_iterations=300,
#     debug=False
# )

# # Step 1: Fetch Stock Prices
# stock_prices_task = Task(
#     description="Fetch the current stock prices of Tesla and Microsoft",
#     agent=general_agent,
#     expected_output="Current stock prices of Tesla and Microsoft",
#     max_iterations=100,  # Smaller iteration limit
#     max_time_seconds=100  # Smaller time limit
# )

# # Step 2: Analyze Stock Prices
# analysis_task = Task(
#     description="Analyze the fetched stock prices and provide insights including hold or sell recommendation",
#     agent=general_agent,
#     expected_output="Insights and recommendations for Tesla and Microsoft stocks",
#     max_iterations=200,
#     max_time_seconds=200
# )

# # Create Crew with Multiple Tasks
# crew = Crew(
#     agents=[general_agent],
#     tasks=[stock_prices_task, analysis_task],  # Split tasks
#     verbose=True,
#     max_iterations=300,
# )

# result = crew.kickoff()

# print("CrewAI Result:")
# print(result)

# # Direct usage of the StockPriceAgent
# print("\nDirect Stock Price Fetching:")
# prices = general_agent.fetch_stock_prices(['TSLA', 'MSFT'])
# print("Fetched Stock Prices: {prices}")
# print(f"Tesla (TSLA) current price: ${prices['TSLA']:.2f}")
# print(f"Microsoft (MSFT) current price: ${prices['MSFT']:.2f}")


from crewai import Agent, Task, Crew
from langchain_ollama import ChatOllama
import yfinance as yf
import os

os.environ["OPENAI_API_KEY"] = "NA"

llm = ChatOllama(
    model="llama3.1",
    base_url="http://localhost:11434")

def get_stock_price(ticker):
    stock = yf.Ticker(ticker)
    current_price = stock.info.get('regularMarketPrice') or stock.info.get('previousClose')
    if current_price is None:
        raise KeyError(f"Could not find a valid price for ticker {ticker}")
    return current_price

class StockPriceAgent(Agent):
    def fetch_stock_prices(self, tickers):
        prices = {}
        for ticker in tickers:
            price = get_stock_price(ticker)
            prices[ticker] = price
        return prices

    def perform_analysis(self, prices):
        analysis = {}
        for ticker, price in prices.items():
            # Simple analysis: Hold if price > 200, Sell otherwise (example logic)
            recommendation = "Hold" if price > 200 else "Sell"
            analysis[ticker] = recommendation
        return analysis

    def fetch_and_analyze_stock_prices(self, tickers):
        prices = self.fetch_stock_prices(tickers)
        analysis = self.perform_analysis(prices)
        return prices, analysis

general_agent = StockPriceAgent(
    role="Stock Market Analyst",
    goal="Provide the latest stock prices of Tesla and Microsoft, and offer recommendations.",
    backstory="You are an excellent stock market specialist with over 10 years of experience. You have a deep understanding of the stock market and have been following Tesla and Microsoft stocks for a long time.",
    allow_delegation=False,
    verbose=True,
    llm=llm,
    max_iterations=300,
    debug=False
)

# Step 1: Fetch Stock Prices
stock_prices_task = Task(
    description="Fetch the current stock prices of Tesla and Microsoft",
    agent=general_agent,
    expected_output="Current stock prices of Tesla and Microsoft",
    max_iterations=100,  # Smaller iteration limit
    max_time_seconds=100  # Smaller time limit
)

# Step 2: Analyze Stock Prices
analysis_task = Task(
    description="Analyze the fetched stock prices and provide insights including hold or sell recommendation",
    agent=general_agent,
    expected_output="Insights and recommendations for Tesla and Microsoft stocks",
    max_iterations=200,
    max_time_seconds=200
)

# Create Crew with Multiple Tasks
crew = Crew(
    agents=[general_agent],
    tasks=[stock_prices_task, analysis_task],  # Split tasks
    verbose=True,
    max_iterations=300,
)

result = crew.kickoff()

print("CrewAI Result:")
print(result)

# Direct usage of the StockPriceAgent
print("\nDirect Stock Price Fetching:")
prices, analysis = general_agent.fetch_and_analyze_stock_prices(['TSLA', 'MSFT'])
print(f"Fetched Stock Prices: {prices}")
print(f"Tesla (TSLA) current price: ${prices['TSLA']:.2f}")
print(f"Microsoft (MSFT) current price: ${prices['MSFT']:.2f}")

# Output the analysis
print("\nStock Analysis and Recommendations:")
for ticker, recommendation in analysis.items():
    print(f"{ticker}: {recommendation}")
