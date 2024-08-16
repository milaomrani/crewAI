# Stock Price Checker

This project uses the CrewAI framework to create an agent that retrieves the latest stock prices for Tesla and Microsoft.

## Prerequisites

- Python 3.10+
- An Ollama server running locally

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/milaomrani/crewAI.git
   cd cerwAI
   ```

2. Install the required packages:
   ```
   pip install crewai langchain-ollama
   ```

3. Ensure you have Ollama set up and running locally with the `llama3.1` model.

## Usage

1. Run the script:
   ```
   python main.py
   ```

2. The script will output the latest stock prices for Tesla and Microsoft.

## How it works

1. The script sets up a CrewAI agent with the role of a Stock Market Analyst.
2. It uses the Ollama language model running locally.
3. A task is created to retrieve the current stock prices of Tesla and Microsoft.
4. The CrewAI framework executes the task using the defined agent.
5. The result is printed to the console.

## Configuration

- The `OPENAI_API_KEY` environment variable is set to "NA" as it's not used in this implementation.
- The Ollama model is set to "llama3.1" and connects to `http://localhost:11434`.

## Limitations

- This script doesn't actually fetch real-time stock prices. It's a demonstration of how to set up a CrewAI agent for such a task.
- The stock prices in the expected output are placeholder values and not real-time data.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).