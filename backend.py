# backend.py

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import os
from langchain_groq import ChatGroq 
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import requests
import random
from datetime import datetime

load_dotenv()

# -------------------
# 1. LLM Configuration
# -------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,  # Lower temperature for more reliable tool calls
    groq_api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=4096,
)

# -------------------
# 2. Tools Definition
# -------------------

# Search Tool
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> str:
    """
    Perform basic arithmetic operation on two numbers.
    
    Args:
        first_num: First number
        second_num: Second number
        operation: Operation to perform (add, sub, mul, div)
    
    Returns:
        String with the calculation result
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return "Error: Division by zero is not allowed"
            result = first_num / second_num
        else:
            return f"Error: Unsupported operation '{operation}'. Use: add, sub, mul, or div"
        
        return f"Calculation: {first_num} {operation} {second_num} = {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"


@tool
def get_stock_price(symbol: str) -> str:
    """
    Fetch latest stock price for a given symbol using Alpha Vantage API.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'GOOGL')
    
    Returns:
        String with stock price information
    """
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=7E5W43K0DV9AMWAC"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if "Global Quote" in data and data["Global Quote"]:
            quote = data["Global Quote"]
            price = quote.get("05. price", "N/A")
            change = quote.get("09. change", "N/A")
            change_percent = quote.get("10. change percent", "N/A")
            
            return f"Stock: {symbol}\nPrice: ${price}\nChange: {change} ({change_percent})"
        else:
            return f"Could not fetch stock data for {symbol}. Please check the symbol."
    except Exception as e:
        return f"Error fetching stock price: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """
    Get current weather information for a given city.
    
    Args:
        city: Name of the city (e.g., 'London', 'New York', 'Tokyo')
    
    Returns:
        String with weather information
    """
    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        current = data['current_condition'][0]
        temp_c = current['temp_C']
        temp_f = current['temp_F']
        condition = current['weatherDesc'][0]['value']
        humidity = current['humidity']
        wind = current['windspeedKmph']
        
        return f"Weather in {city}:\n" \
               f"Temperature: {temp_c}°C ({temp_f}°F)\n" \
               f"Condition: {condition}\n" \
               f"Humidity: {humidity}%\n" \
               f"Wind Speed: {wind} km/h"
    except Exception as e:
        return f"Error fetching weather: {str(e)}"


@tool
def get_current_time(timezone: str = "UTC") -> str:
    """
    Get current time in a specific timezone.
    
    Args:
        timezone: Timezone name (e.g., 'UTC', 'US/Eastern', 'Europe/London', 'Asia/Tokyo')
    
    Returns:
        String with current time information
    """
    try:
        import pytz
        
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        
        return f"Current time in {timezone}:\n" \
               f"{current_time.strftime('%Y-%m-%d %H:%M:%S')}\n" \
               f"Day: {current_time.strftime('%A')}"
    except Exception as e:
        return f"Error getting time (use format like 'US/Eastern'): {str(e)}"


@tool
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Convert currency from one type to another.
    
    Args:
        amount: Amount to convert
        from_currency: Source currency code (e.g., 'USD', 'EUR', 'GBP')
        to_currency: Target currency code (e.g., 'USD', 'EUR', 'GBP')
    
    Returns:
        String with conversion result
    """
    try:
        url = f"https://api.exchangerate-api.com/v4/latest/{from_currency.upper()}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if to_currency.upper() not in data['rates']:
            return f"Error: Currency {to_currency} not found"
        
        rate = data['rates'][to_currency.upper()]
        converted = amount * rate
        
        return f"Currency Conversion:\n" \
               f"{amount} {from_currency.upper()} = {converted:.2f} {to_currency.upper()}\n" \
               f"Exchange Rate: 1 {from_currency.upper()} = {rate:.4f} {to_currency.upper()}"
    except Exception as e:
        return f"Error converting currency: {str(e)}"


@tool
def generate_random_number(min_value: int, max_value: int) -> str:
    """
    Generate a random number between min and max values (inclusive).
    
    Args:
        min_value: Minimum value
        max_value: Maximum value
    
    Returns:
        String with the random number
    """
    try:
        if min_value > max_value:
            return "Error: min_value must be less than or equal to max_value"
        
        number = random.randint(min_value, max_value)
        return f"Random number between {min_value} and {max_value}: {number}"
    except Exception as e:
        return f"Error generating random number: {str(e)}"


@tool
def text_analyzer(text: str) -> str:
    """
    Analyze text and return statistics.
    
    Args:
        text: Text to analyze
    
    Returns:
        String with text statistics
    """
    try:
        words = text.split()
        sentences = [s for s in text.split('.') if s.strip()]
        
        char_count = len(text)
        word_count = len(words)
        sentence_count = len(sentences)
        avg_word_len = round(sum(len(word) for word in words) / len(words), 2) if words else 0
        
        return f"Text Analysis:\n" \
               f"Characters: {char_count}\n" \
               f"Words: {word_count}\n" \
               f"Sentences: {sentence_count}\n" \
               f"Average Word Length: {avg_word_len}"
    except Exception as e:
        return f"Error analyzing text: {str(e)}"


@tool
def get_crypto_price(symbol: str) -> str:
    """
    Get current cryptocurrency price in USD.
    
    Args:
        symbol: Cryptocurrency name (e.g., 'bitcoin', 'ethereum', 'cardano', 'solana')
    
    Returns:
        String with crypto price information
    """
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd&include_24hr_change=true"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if symbol.lower() not in data:
            return f"Error: Cryptocurrency '{symbol}' not found. Try: bitcoin, ethereum, cardano, solana"
        
        coin_data = data[symbol.lower()]
        price = coin_data['usd']
        change = round(coin_data.get('usd_24h_change', 0), 2)
        
        return f"Cryptocurrency: {symbol.capitalize()}\n" \
               f"Price: ${price:,.2f} USD\n" \
               f"24h Change: {change:+.2f}%"
    except Exception as e:
        return f"Error fetching crypto price: {str(e)}"


@tool
def joke_generator(category: str = "general") -> str:
    """
    Get a random joke.
    
    Args:
        category: Joke category ('general' or 'programming')
    
    Returns:
        String with a joke
    """
    try:
        if category.lower() == "programming":
            url = "https://official-joke-api.appspot.com/jokes/programming/random"
        else:
            url = "https://official-joke-api.appspot.com/random_joke"
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if isinstance(data, list):
            data = data[0]
        
        setup = data['setup']
        punchline = data['punchline']
        
        return f"{setup}\n{punchline}"
    except Exception as e:
        return f"Error fetching joke: {str(e)}"


# Combine all tools
tools = [
    search_tool,
    calculator,
    get_stock_price,
    get_weather,
    get_current_time,
    currency_converter,
    generate_random_number,
    text_analyzer,
    get_crypto_price,
    joke_generator
]

llm_with_tools = llm.bind_tools(tools)

# -------------------
# 3. State Definition
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Node Functions
# -------------------
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    try:
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    except Exception as e:
        print(f"ERROR in chat_node: {str(e)}")
        # Return error message to continue conversation
        error_response = AIMessage(
            content=f"I encountered an error: {str(e)}. Let me try to help you differently."
        )
        return {"messages": [error_response]}

# Tool node
tool_node = ToolNode(tools)

# -------------------
# 5. Checkpointer Setup
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 6. Graph Construction
# -------------------
graph = StateGraph(ChatState)

# Add nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

# Add edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')

# Compile graph
chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper Functions
# -------------------
def retrieve_all_threads():
    """Retrieve all conversation thread IDs from checkpointer."""
    all_threads = set()
    try:
        for checkpoint in checkpointer.list(None):
            all_threads.add(checkpoint.config["configurable"]["thread_id"])
    except Exception as e:
        print(f"Error retrieving threads: {e}")
    return list(all_threads)


def chat(user_input: str, thread_id: str = "default"):
    """
    Send a message to the chatbot and get a response.
    
    Args:
        user_input: User's message
        thread_id: Conversation thread ID (default: "default")
    
    Returns:
        String response from the chatbot
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        
        # Invoke chatbot
        result = chatbot.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        
        # Get the last message (assistant's response)
        last_message = result["messages"][-1]
        return last_message.content
    except Exception as e:
        return f"Error: {str(e)}"



    
