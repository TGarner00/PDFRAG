from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import Dict, List, Any
import operator
import os

class CalculatorTool(BaseTool):
    """Base class for calculator tools that handle string inputs."""
    
    def _parse_input(self, input_str: str) -> tuple[float, float]:
        """Parse a string input into two numbers."""
        try:
            # Split the input string and convert to floats
            nums = [float(x.strip()) for x in input_str.split(',')]
            if len(nums) != 2:
                raise ValueError("Input must be exactly two numbers separated by a comma")
            return nums[0], nums[1]
        except Exception as e:
            raise ValueError(f"Invalid input format. Please provide two numbers separated by a comma. Error: {str(e)}")

class AddTool(CalculatorTool):
    name = "add"
    description = "Add two numbers. Input should be two numbers separated by a comma (e.g., '2, 3')"
    
    def _run(self, input_str: str) -> float:
        a, b = self._parse_input(input_str)
        return a + b

class SubtractTool(CalculatorTool):
    name = "subtract"
    description = "Subtract second number from first number. Input should be two numbers separated by a comma (e.g., '5, 3')"
    
    def _run(self, input_str: str) -> float:
        a, b = self._parse_input(input_str)
        return a - b

class MultiplyTool(CalculatorTool):
    name = "multiply"
    description = "Multiply two numbers. Input should be two numbers separated by a comma (e.g., '2, 3')"
    
    def _run(self, input_str: str) -> float:
        a, b = self._parse_input(input_str)
        return a * b

class DivideTool(CalculatorTool):
    name = "divide"
    description = "Divide first number by second number. Input should be two numbers separated by a comma (e.g., '6, 2')"
    
    def _run(self, input_str: str) -> float:
        a, b = self._parse_input(input_str)
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

class PowerTool(CalculatorTool):
    name = "power"
    description = "Raise first number to the power of second number. Input should be two numbers separated by a comma (e.g., '2, 3')"
    
    def _run(self, input_str: str) -> float:
        a, b = self._parse_input(input_str)
        return a ** b

# Create instances of calculator tools
calculator_tools = [
    AddTool(),
    SubtractTool(),
    MultiplyTool(),
    DivideTool(),
    PowerTool()
]

# Define the ReAct prompt template
REACT_TEMPLATE = """You are a calculator assistant that solves mathematical expressions step by step.
You have access to tools for basic arithmetic operations.

For each step:
1. Think about what operation needs to be performed next based on order of operations (PEMDAS)
2. Format the numbers as comma-separated values for the tool (e.g., "2, 3" for 2 + 3)
3. Use the appropriate tool to perform the calculation
4. Keep track of intermediate results for the next step

Question: {input}

{agent_scratchpad}"""

class CalculatorAgent:
    """An agent that can solve multi-step calculator operations using langchain's ReAct approach."""
    
    def __init__(self, api_key=None):
        """Initialize the calculator agent with langchain ReAct components."""
        # Use provided API key or get from environment
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        # Initialize the LLM with OpenAI
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo-16k",
            temperature=0,
            api_key=api_key
        )
        
        # Create the prompt
        prompt = PromptTemplate.from_template(REACT_TEMPLATE)
        
        # Create the ReAct agent
        agent = create_react_agent(
            llm=self.llm,
            tools=calculator_tools,
            prompt=prompt
        )
        
        # Create the agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=calculator_tools,
            verbose=True,
            max_iterations=10,
            return_intermediate_steps=True
        )
        
        # Store steps
        self.steps = []
    
    def calculate(self, expression: str) -> Dict[str, Any]:
        """Solve a mathematical expression using the ReAct approach."""
        # Set up the user query
        query = f"""Solve this mathematical expression step by step: {expression}
        Remember to:
        1. Follow order of operations (PEMDAS)
        2. Show your work for each step
        3. Format numbers as comma-separated values for the tools
        """
        
        # Run the agent to solve the expression
        response = self.agent_executor.invoke({"input": query})
        
        # Store the steps
        self.steps = response.get("intermediate_steps", [])
        
        # Return the result along with the expression
        return {
            "expression": expression,
            "result": response.get("output"),
            "steps": self.steps,
            "reasoning": self._extract_reasoning()
        }
    
    def _extract_reasoning(self) -> List[str]:
        """Extract the reasoning steps from the agent's thought process."""
        reasoning = []
        
        for step in self.steps:
            # Extract the thought process (which is in the first position for ReAct agents)
            action = step[0]
            if hasattr(action, "log"):
                reasoning.append(action.log)
            
        return reasoning


# Example usage
if __name__ == "__main__":
    # Create a calculator agent
    calculator = CalculatorAgent()
    
    # Test some expressions
    expressions = [
        "2 + 3 * 4",
        "(10 - 2) / 4",
        "2^3 + 5"
    ]
    
    for expression in expressions:
        result = calculator.calculate(expression)
        print(f"\nExpression: {expression}")
        print(f"Result: {result['result']}")
        print(f"Number of steps: {len(result['steps'])}")
        
        # Display detailed reasoning for this calculation
        print("\nReasoning Process:")
        for i, reasoning in enumerate(result.get("reasoning", [])):
            if reasoning:
                print(f"Step {i+1} Reasoning: {reasoning}")
        
        print("\n" + "-"*50)
    
    # You can also view the raw steps with tool usage
    print("\nDetailed tool usage for the last calculation:")
    for i, step in enumerate(calculator.steps):
        action = step[0]
        observation = step[1]
        print(f"Step {i+1}:")
        print(f"  Action: {action}")
        print(f"  Observation: {observation}")
        print()
