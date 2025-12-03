import streamlit as st
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# -----------------------------
# Async agent function
# -----------------------------
async def run_trading_agent(user_message):
    client = MultiServerMCPClient({
        "Server-1": {
            "transport": "streamable_http",
            "url": "http://localhost:8000/mcp",
        },
        "Server-2": {
            "transport": "streamable_http",
            "url": "http://localhost:8001/mcp",
        }
    })

    # Load tools from MCP servers
    tools = await client.get_tools()

    # LLM
    llm = ChatOpenAI(
        model='gpt-5-nano',
        temperature=0,
        service_tier="flex"
    )

    # Build agent
    agent = create_agent(
        model=llm,
        tools=tools
    )

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ('system', 'You are a helpful assistant. Use MCP tools when needed.'),
        ('human', '{user_message}')
    ])

    chain = prompt | agent

    # Run chain with timeout
    result = await asyncio.wait_for(chain.ainvoke({'user_message' : user_message}), timeout=300)

    return result['messages'][-1].content


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 MCP Trading Agent UI")
st.write("Ask anything. The agent will use MCP tools when required.")

user_input = st.text_input("Enter your message:")

if st.button("Run Agent"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        with st.spinner("Agent thinking..."):
            response = asyncio.run(run_trading_agent(user_input))
        st.subheader("### Agent Response:")
        st.write(response)
