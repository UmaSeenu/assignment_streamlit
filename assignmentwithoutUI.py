from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()


# --------------------------------------------------
# Tool 1: Requirement Analysis
# --------------------------------------------------

@tool
def requirement_analysis(requirement: str) -> str:
    """
    Analyze the user's requirement and identify the key
    functionality, inputs, validations, and possible
    test scenarios.
    """

    return f"""
    Analyze the following requirement:

    {requirement}

    Identify:
    - Main functionality
    - Required fields
    - Business rules
    - Validations
    - Possible positive scenarios
    - Possible negative scenarios
    - Boundary conditions
    """


# --------------------------------------------------
# Tool 2: Test Case Generator
# --------------------------------------------------

@tool
def test_case_generator(
    requirement: str,
    test_type: str,
    number_of_test_cases: int
) -> str:
    """
    Generate test cases based on the requirement,
    test type and requested number of test cases.
    """

    return f"""
    Generate {number_of_test_cases} {test_type} test cases
    for the following requirement:

    {requirement}

    Each test case should contain:

    - Test Case ID
    - Scenario
    - Test Steps
    - Test Data
    - Expected Result
    - Priority
    """


# --------------------------------------------------
# System Prompt
# --------------------------------------------------

system_prompt = """
You are a Senior QA Engineer with 12+ years of experience
in software testing and test automation.

You generate high-quality test cases from software
requirements.

Follow this workflow:

1. First understand and analyze the requirement.
2. Use the requirement_analysis tool to analyze the requirement.
3. Use the test_case_generator tool to generate test cases.
4. Generate only the requested number of test cases.
5. Avoid duplicate scenarios.
6. Ensure the test cases match the requested test type.
"""


# --------------------------------------------------
# Create Agent
# --------------------------------------------------

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[
        requirement_analysis,
        test_case_generator
    ],
    system_prompt=system_prompt
)


# --------------------------------------------------
# User Inputs
# --------------------------------------------------

requirement = input(
    "Enter the requirement: "
)

test_type = input(
    "Enter Test Type "
    "(Functional / Positive / Negative / Boundary / All): "
)

number_of_test_cases = int(
    input("Enter number of test cases: ")
)


# --------------------------------------------------
# User Prompt using f-string
# --------------------------------------------------

user_prompt = f"""
Generate test cases for the following requirement.

Requirement:
{requirement}

Test Type:
{test_type}

Number of Test Cases:
{number_of_test_cases}

First analyze the requirement and then generate
the requested test cases.
"""


# --------------------------------------------------
# Invoke Agent
# --------------------------------------------------

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    }
)


# --------------------------------------------------
# Output
# --------------------------------------------------

print("\n===== GENERATED TEST CASES =====\n")

print(result["messages"][-1].content)