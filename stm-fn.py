import streamlit as st
import plotly.express as px
from main import CustomChatOllama, StockPriceAgent, get_stock_price
from typing import List, Dict, Any
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def get_ticker_from_company(company_name: str, llm) -> str:
    prompt = f"What is the stock ticker symbol for {company_name}?"
    response = llm.invoke(prompt)
    words = response.content.upper().split()
    for word in words:
        if word.isalpha() and len(word) <= 5:
            try:
                if yf.Ticker(word).info:
                    return word
            except:
                continue
    return None

def initialize_llm():
    return CustomChatOllama(
        model="deepseek-r1:8b",
        base_url="http://localhost:11434"
    )

def get_historical_data(ticker: str, days: int) -> Dict[str, Any]:
    stock = yf.Ticker(ticker)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    history = stock.history(start=start_date, end=end_date)
    return {
        'history': history,
        'stats': {
            'Price_Change': ((history['Close'][-1] - history['Close'][0]) / history['Close'][0] * 100),
            'Volume_Avg': history['Volume'].mean(),
            'High': history['High'].max(),
            'Low': history['Low'].min()
        }
    }

def create_agent(llm):
    return StockPriceAgent(
        llm=llm,
        role="Stock Market Analyst",
        goal="Analyze stock prices and provide detailed market insights",
        backstory="Expert stock market analyst with deep market understanding and technical analysis expertise",
        allow_delegation=False,
        verbose=True
    )

def get_llm_analysis(agent, ticker: str, hist_data: pd.DataFrame) -> str:
    company_info = yf.Ticker(ticker).info
    company_name = company_info.get('longName', ticker)
    
    prompt = f"""
    Analyze {company_name} ({ticker}) based on:
    - Current price trend
    - Volume analysis
    - Price movement patterns
    - Recent highs and lows
    
    Provide a concise market analysis and trading recommendation.
    """
    response = agent.llm.invoke(prompt)
    return response.content

def analyze_stocks(agent, company_names: List[str], analysis_days: int):
    tickers = []
    company_data = {}
    
    for company in company_names:
        ticker = get_ticker_from_company(company, agent.llm)
        if ticker:
            st.info(f"Found ticker {ticker} for {company}")
            tickers.append(ticker)
        else:
            st.warning(f"Could not find ticker for {company}")
    
    if not tickers:
        st.error("No valid tickers found")
        return pd.DataFrame(), {}, pd.DataFrame()
    
    historical_data = {}
    llm_analysis = {}
    all_historical_df = pd.DataFrame()
    
    for ticker in tickers:
        try:
            # Get current stock data
            stock_data = get_stock_price(ticker)
            company_data[ticker] = stock_data
            
            # Get historical data
            data = get_historical_data(ticker, analysis_days)
            historical_data[ticker] = data['stats']
            
            hist_df = data['history'].reset_index()
            hist_df['Ticker'] = ticker
            all_historical_df = pd.concat([all_historical_df, hist_df])
            
            llm_analysis[ticker] = get_llm_analysis(agent, ticker, data['history'])
            
        except Exception as e:
            st.error(f"Error analyzing {ticker}: {str(e)}")
            historical_data[ticker] = {
                'Price_Change': None,
                'Volume_Avg': None,
                'High': None,
                'Low': None
            }
            company_data[ticker] = None
            llm_analysis[ticker] = f"Error analyzing {ticker}: {str(e)}"
    
    df = pd.DataFrame({
        'Company': [company_data[t]['shortName'] for t in tickers],
        'Ticker': tickers,
        'Current_Price': [company_data[t]['currentPrice'] for t in tickers],
        'Market_Cap': [company_data[t]['marketCap'] for t in tickers],
        'Volume': [company_data[t]['volume'] for t in tickers],
        '52W_High': [company_data[t]['fiftyTwoWeekHigh'] for t in tickers],
        '52W_Low': [company_data[t]['fiftyTwoWeekLow'] for t in tickers],
        'Sector': [company_data[t]['sector'] for t in tickers],
        'Industry': [company_data[t]['industry'] for t in tickers],
        'Price_Change_%': [historical_data[t]['Price_Change'] for t in tickers]
    })
    
    return df, llm_analysis, all_historical_df

def main():
    st.title("Advanced Stock Market Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        companies = st.text_input("Enter company names (comma-separated)", "Apple, Microsoft, Google")
    with col2:
        analysis_days = st.number_input("Analysis period (days)", min_value=1, max_value=365*2, value=180)
    
    analyze_button = st.button("Analyze")
    
    if analyze_button:
        try:
            company_names = [c.strip() for c in companies.split(",")]
            
            with st.spinner("Initializing AI agent..."):
                llm = initialize_llm()
                agent = create_agent(llm)
            
            with st.spinner(f"Analyzing companies for the past {analysis_days} days..."):
                df, llm_analysis, historical_df = analyze_stocks(agent, company_names, analysis_days)
                
                # Summary Statistics
                st.subheader("Analysis Results")
                df_display = df.copy()
                for col in ['Current_Price', 'Period_High', 'Period_Low']:
                    df_display[col] = df_display[col].apply(lambda x: f'${x:.2f}' if x is not None else 'N/A')
                df_display['Price_Change_%'] = df_display['Price_Change_%'].apply(lambda x: f'{x:.2f}%' if x is not None else 'N/A')
                df_display['Avg_Volume'] = df_display['Avg_Volume'].apply(lambda x: f'{x:,.0f}' if x is not None else 'N/A')
                st.dataframe(df_display)
                
                # Historical Price Plot
                st.subheader("Historical Price Trends")
                fig = px.scatter(historical_df, x='Date', y='Close', color='Ticker',
                               title='Stock Price History',
                               labels={'Close': 'Price ($)', 'Date': 'Date'})
                fig.add_traces(px.line(historical_df, x='Date', y='Close', color='Ticker').data)
                st.plotly_chart(fig)
                
                # Volume Analysis
                st.subheader("Trading Volume Analysis")
                fig_volume = px.scatter(historical_df, x='Date', y='Volume', color='Ticker',
                                      title='Trading Volume History',
                                      labels={'Volume': 'Volume', 'Date': 'Date'})
                st.plotly_chart(fig_volume)
                
                # LLM Analysis
                st.subheader("AI Market Analysis")
                for ticker in df['Ticker']:
                    company = df[df['Ticker'] == ticker]['Company'].iloc[0]
                    with st.expander(f"{company} ({ticker}) Analysis"):
                        st.write(llm_analysis[ticker])
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()