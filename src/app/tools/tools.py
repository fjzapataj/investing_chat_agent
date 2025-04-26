from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
import numpy as np
import yfinance as yf
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from typing import Annotated

# This executes code locally, which can be unsafe
repl = PythonREPL()


@tool
def add(a, b):
    """
    Add two integer numbers together

    Args:
    a: First Integer
    b: Second Integer
    """
    return a + b


@tool
def multiply(a, b):
    """
    Multiply two integer numbers together

    Args:
    a: First Integer
    b: Second Integer
    """
    return a * b


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code and do math. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = (
        f"Successfully executed:\n\`\`\`python\n{code}\n\`\`\`\nStdout: {result}"
    )
    return result_str


@tool
def optimize_portfolio_kde_cvar(risk_profile: str = "medio"):
    """
    Optimiza un portafolio según el perfil de riesgo.

    Args:
    - risk_profile: Perfil de riesgo ('bajo', 'medio', 'alto').

    Returns:
    - optimal_weights: Pesos óptimos para cada acción.
    - portfolio_return: Retorno esperado del portafolio.
    - portfolio_var: cVaR del portafolio.
    """

    tickers = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "BTC-USD",
        "^GSPC",
        "^DJI",
        "^IXIC",
        "GC=F",
    ]
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    risk_profile = risk_profile.lower()
    # Ajustar el nivel de confianza según el perfil de riesgo
    if risk_profile == "bajo":
        confidence_level = 0.95
    elif risk_profile == "medio":
        confidence_level = 0.90
    elif risk_profile == "alto":
        confidence_level = 0.85
    else:
        confidence_level = 0.95  # valor por defecto

    # Descargar datos de Yahoo Finance
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    returns = np.log(data / data.shift(1)).dropna()
    n_assets = len(tickers)

    def calculate_cvar(weights):
        portfolio_returns = np.dot(returns, weights)
        kde = gaussian_kde(portfolio_returns)
        x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 1000)
        kde_values = kde(x)
        cumulative_values = np.cumsum(kde_values) / np.sum(kde_values)
        var_index = np.where(cumulative_values >= (1 - confidence_level))[0][0]
        var_value = x[var_index]
        cvar_estimation = np.mean(portfolio_returns[portfolio_returns <= var_value])
        return -cvar_estimation

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for asset in range(n_assets))
    initial_weights = np.ones(n_assets) / n_assets

    result = minimize(
        calculate_cvar,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    optimal_weights = result.x
    cvar_value = calculate_cvar(optimal_weights)

    return optimal_weights, -cvar_value
