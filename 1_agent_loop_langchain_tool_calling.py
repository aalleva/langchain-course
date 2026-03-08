from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "gpt-4o-mini"

# --- Tools (LangChain @tool decorator) ---
@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalogue."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0.00)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers are: bronze, silver, gold.
    """
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"gold": 23, "silver": 12, "bronze": 5}
    discount = discount_percentages.get(discount_tier, 0.00)
    return round(price * (1 - discount / 100), 2)

# --- Model ---
model = init_chat_model(MODEL)

# --- Agent Loop ---
@traceable(name="LangChain Agent Loop")
def run_agent(question: str) -> str:
    pass

def main() -> None:
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop applying a gold discount?")

if __name__ == "__main__":
    main()
