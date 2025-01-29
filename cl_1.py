from crewai import Agent, Task, Crew
from langchain_ollama import ChatOllama
import yfinance as yf
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_stock_data(ticker: str) -> Dict:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'price': info['currentPrice'] if 'currentPrice' in info else info['regularMarketPrice'],
            'volume': info['volume'] if 'volume' in info else info['regularMarketVolume'],
            'change': info['regularMarketChangePercent']
        }
    except KeyError as e:
        logger.error(f"Missing data for {ticker}: {e}")
        return {'price': None, 'volume': None, 'change': None}
    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        return {'price': None, 'volume': None, 'change': None}

class StockAgent(Agent):
    def analyze_stocks(self, tickers: List[str]) -> str:
        results = []
        for ticker in tickers:
            data = get_stock_data(ticker)
            if data['price']:
                results.append(f"{ticker}: ${data['price']:.2f} | Change: {data['change']:.2f}%")
        return "\n".join(results)

def create_tasks(agent: Agent, tickers: List[str]) -> List[Task]:
    return [
        Task(
            description=f"Analyze stocks: {', '.join(tickers)}",
            agent=agent,
            expected_output="Stock analysis report",
            context=["Analyze current market data"]
        )
    ]

def main():
    llm = ChatOllama(
        model="deepseek-r1:8b",
        base_url="http://localhost:11434"
    )
    
    stock_agent = StockAgent(
        role='Stock Analyst',
        goal='Analyze market data',
        backstory='Senior market analyst',
        llm=llm,
        verbose=True
    )
    
    tickers = ['TSLA', 'MSFT', 'AAPL', 'GOOGL', 'NVDA']
    
    try:
        tasks = create_tasks(stock_agent, tickers)
        crew = Crew(
            agents=[stock_agent],
            tasks=tasks,
            verbose=True
        )
        
        result = crew.kickoff()
        print("\nAnalysis Results:")
        print(result)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()