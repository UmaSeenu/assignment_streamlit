import os

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from streamlit.errors import StreamlitSecretNotFoundError

load_dotenv()


@tool
def requirement_analysis(requirement: str) -> str:
    """Analyze a software requirement for test planning."""
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


@tool
def test_case_generator(
    requirement: str,
    test_type: str,
    number_of_test_cases: int,
) -> str:
    """Generate test cases for a requirement and test type."""
    return f"""
Generate {number_of_test_cases} {test_type} test cases for the following requirement:

{requirement}

Each test case should contain:
- Test Case ID
- Scenario
- Test Steps
- Test Data
- Expected Result
- Priority
"""


SYSTEM_PROMPT = """
You are a Senior QA Engineer with 12+ years of experience in software testing
and test automation.

Generate high-quality test cases from software requirements.

Follow this workflow:
1. First understand and analyze the requirement.
2. Use the requirement_analysis tool to analyze the requirement.
3. Use the test_case_generator tool to generate test cases.
4. Generate only the requested number of test cases.
5. Avoid duplicate scenarios.
6. Ensure the test cases match the requested test type.
"""


def generate_test_cases(requirement: str, test_type: str, count: int) -> str:
    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[requirement_analysis, test_case_generator],
        system_prompt=SYSTEM_PROMPT,
    )
    user_prompt = f"""
Generate test cases for the following requirement.

Requirement:
{requirement}

Test Type:
{test_type}

Number of Test Cases:
{count}

First analyze the requirement and then generate the requested test cases.
"""
    result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
    content = result["messages"][-1].content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def configure_api_key() -> bool:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except (KeyError, StreamlitSecretNotFoundError):
            return False
        os.environ["OPENAI_API_KEY"] = api_key
    return True


st.set_page_config(page_title="QA Test Case Generator", page_icon="✅", layout="wide")
st.title("QA Test Case Generator")
st.caption("Turn a software requirement into focused, structured test cases.")

with st.form("test_case_form"):
    requirement = st.text_area(
        "Software requirement",
        height=180,
        placeholder="Example: A user can reset their password using a registered email address.",
    )
    left_column, right_column = st.columns(2)
    with left_column:
        test_type = st.selectbox(
            "Test type", ["Functional", "Positive", "Negative", "Boundary", "All"]
        )
    with right_column:
        count = st.number_input(
            "Number of test cases", min_value=1, max_value=50, value=5, step=1
        )
    submitted = st.form_submit_button("Generate test cases", type="primary")

if submitted:
    if not requirement.strip():
        st.error("Enter a software requirement before generating test cases.")
    elif not configure_api_key():
        st.error("OPENAI_API_KEY is missing. Add it in Streamlit Cloud app settings under Secrets.")
    else:
        with st.spinner("Analyzing the requirement and generating test cases..."):
            try:
                st.session_state["latest_result"] = generate_test_cases(
                    requirement.strip(), test_type, int(count)
                )
            except Exception as error:
                st.error(f"Generation failed: {error}")

if "latest_result" in st.session_state:
    st.subheader("Generated test cases")
    st.markdown(st.session_state["latest_result"])
    st.download_button(
        "Download results",
        data=st.session_state["latest_result"],
        file_name="generated_test_cases.md",
        mime="text/markdown",
    )