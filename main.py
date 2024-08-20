# Required Libraries: crewai, langchain_ollama, yfinance
from crewai import Agent, Task, Crew
from langchain_ollama import ChatOllama
import yfinance as yf
import os

# Not required for this example
os.environ["OPENAI_API_KEY"] = "NA"

# Initialize the LLaMA model
llm = ChatOllama(
    model="llama3.1",
    base_url="http://localhost:11434"
)

def get_stock_price(ticker):
    stock = yf.Ticker(ticker)
    current_price = stock.info.get('regularMarketPrice') or stock.info.get('previousClose')
    if current_price is None:
        raise KeyError(f"Could not find a valid price for ticker {ticker}")
    return current_price

class StockPriceAgent(Agent):
    def __init__(self, llm, role, goal, backstory, allow_delegation, verbose):
        super().__init__(role=role, goal=goal, backstory=backstory, allow_delegation=allow_delegation, verbose=verbose)
        self.llm = llm

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

# Instantiate the StockPriceAgent with the required parameters
# Instantiate the StockPriceAgent with the required parameters
general_agent = StockPriceAgent(
    llm=llm,
    role="Stock Market Analyst",
    goal="Provide the latest stock prices of Tesla, Microsoft, Apple, and Google, and offer recommendations.",
    backstory="You are an excellent stock market specialist with over 10 years of experience. You have a deep understanding of the stock market and have been following these stocks for a long time.",
    allow_delegation=False,
    verbose=True,
)


# Step 1: Fetch Stock Prices Task
stock_prices_task = Task(
    description="Fetch the current stock prices of Tesla, Microsoft, Apple, and Google",
    agent=general_agent,
    expected_output="Current stock prices of Tesla, Microsoft, Apple, and Google",
    max_iterations=200,
    max_time_seconds=200
)

# Step 2: Analyze Stock Prices Task
analysis_task = Task(
    description="Analyze the fetched stock prices and provide insights including hold or sell recommendation",
    agent=general_agent,
    expected_output="Insights and recommendations for Tesla, Microsoft, Apple, and Google stocks",
    max_iterations=400,
    max_time_seconds=400
)

# Create Crew with Multiple Tasks
crew = Crew(
    agents=[general_agent],
    tasks=[stock_prices_task, analysis_task],
    verbose=True,
    max_iterations=500,
)

# Run the CrewAI tasks
result = crew.kickoff()

print("CrewAI Result:")
print(result)

# Direct usage of the StockPriceAgent
print("\nDirect Stock Price Fetching:")
prices, analysis = general_agent.fetch_and_analyze_stock_prices(['TSLA', 'MSFT', 'AAPL', 'GOOGL'])
print(f"Fetched Stock Prices: {prices}")
print(f"Tesla (TSLA) current price: ${prices['TSLA']:.2f}")
print(f"Microsoft (MSFT) current price: ${prices['MSFT']:.2f}")
print(f"Apple (AAPL) current price: ${prices['AAPL']:.2f}")
print(f"Google (GOOGL) current price: ${prices['GOOGL']:.2f}")

# Output the analysis
print("\nStock Analysis and Recommendations:")
for ticker, recommendation in analysis.items():
    print(f"{ticker}: {recommendation}")
