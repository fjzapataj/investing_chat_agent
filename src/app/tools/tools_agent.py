### Tool Creation

from langchain_core.tools import tool

# Define the tools that will be used in the agent


class SetTools:
    def __init__(self):
        self.tools = []

    def __call__(self, tool):
        self.tools.append(tool)
        return tool

    # custom tools
    @tool
    def add(a, b):
        """
        Add two integer numbers together

        Args:
        a: First Integer
        b: Second Integer
        """
        return a + b
