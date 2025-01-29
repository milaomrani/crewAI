# Required Libraries: crewai, langchain_ollama, yfinance
from crewai import Agent, Task, Crew
from langchain_ollama import ChatOllama
import yfinance as yf
from typing import Dict, List, Tuple

class CustomChatOllama(ChatOllama):
    def supports_stop_words(self) -> bool:
        return False
    
    def supports_function_calling(self) -> bool:
        return False
        
    def call(self, prompt: str, **kwargs) -> str:
        response = self.invoke(prompt)
        return response.content

# Initialize the LLaMA model
llm = CustomChatOllama(
    model="deepseek-r1:8b",
    base_url="http://localhost:11434"
)

def get_stock_price(ticker: str) -> float:
    stock = yf.Ticker(ticker)
    current_price = stock.info.get('regularMarketPrice') or stock.info.get('previousClose')
    if current_price is None:
        raise ValueError(f"Could not find a valid price for ticker {ticker}")
    return current_price

class StockPriceAgent(Agent):
    def __init__(self, llm, role: str, goal: str, backstory: str, allow_delegation: bool, verbose: bool):
        super().__init__(role=role, goal=goal, backstory=backstory, allow_delegation=allow_delegation, verbose=verbose)
        self.llm = llm

    def fetch_stock_prices(self, tickers: List[str]) -> Dict[str, float]:
        return {ticker: get_stock_price(ticker) for ticker in tickers}

    def perform_analysis(self, prices: Dict[str, float]) -> Dict[str, str]:
        return {ticker: "Hold" if price > 200 else "Sell" for ticker, price in prices.items()}

    def fetch_and_analyze_stock_prices(self, tickers: List[str]) -> Tuple[Dict[str, float], Dict[str, str]]:
        prices = self.fetch_stock_prices(tickers)
        analysis = self.perform_analysis(prices)
        return prices, analysis

def create_stock_agent() -> StockPriceAgent:
    return StockPriceAgent(
        llm=llm,
        role="Stock Market Analyst",
        goal="Provide the latest stock prices of Tesla, Microsoft, Apple, and Google, Nvidia and offer recommendations.",
        backstory="You are an excellent stock market specialist with over 10 years of experience. You have a deep understanding of the stock market and have been following these stocks for a long time.",
        allow_delegation=False,
        verbose=True,
    )

def create_tasks(agent: StockPriceAgent) -> List[Task]:
    return [
        Task(
            description="Fetch the current stock prices of Tesla, Microsoft, Apple, and Google and Nvidia",
            agent=agent,
            expected_output="Current stock prices of Tesla, Microsoft, Apple, and Google and Nvidia",
            max_iterations=200,
            max_time_seconds=200
        ),
        Task(
            description="Analyze the fetched stock prices and provide insights including hold or sell recommendation",
            agent=agent,
            expected_output="Insights and recommendations for Tesla, Microsoft, Apple, and Google stocks",
            max_iterations=400,
            max_time_seconds=400
        )
    ]

def run_crew_ai_tasks(agent: StockPriceAgent, tasks: List[Task]) -> str:
    crew = Crew(
        agents=[agent],
        tasks=tasks,
        verbose=True,
        max_iterations=500,
    )
    return crew.kickoff()

def print_stock_info(prices: Dict[str, float], analysis: Dict[str, str]):
    print("\nDirect Stock Price Fetching:")
    print(f"Fetched Stock Prices: {prices}")
    for ticker in prices:
        print(f"{ticker} current price: ${prices[ticker]:.2f}")

    print("\nStock Analysis and Recommendations:")
    for ticker, recommendation in analysis.items():
        print(f"{ticker}: {recommendation}")

def main():
    agent = create_stock_agent()
    tasks = create_tasks(agent)

    print("CrewAI Result:")
    result = run_crew_ai_tasks(agent, tasks)
    print(result)

    tickers = ['TSLA', 'MSFT', 'AAPL', 'GOOGL', 'NVDA']
    prices, analysis = agent.fetch_and_analyze_stock_prices(tickers)
    print_stock_info(prices, analysis)

if __name__ == "__main__":
    main()