"""
Debate Arena — Agentic Fact-Check Engine
Streamlit frontend that connects to mcp_server.py via stdio, binds MCP tools
to a Gemini-powered ReAct agent, and streams every tool call to the UI.
"""

import asyncio
import os
import sys
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Debate Arena", page_icon="⚖️", layout="wide")

# ---------------------------------------------------------------------------
# Custom CSS — dark-mode polish
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {max-width: 920px; padding-top: 1.5rem;}
    h1 {letter-spacing: -0.4px;}
    .tool-card {
        background: rgba(108, 99, 255, 0.07);
        border-left: 3px solid #6C63FF;
        padding: 0.55rem 1rem;
        border-radius: 0 6px 6px 0;
        margin: 0.35rem 0;
        font-size: 0.92rem;
    }
    .data-label {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.3rem;
        margin-bottom: 0.15rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("⚖️ Debate Arena")
st.caption(
    "Submit an argument and watch the AI referee autonomously fact-check "
    "every claim using live data from Wikipedia, the Web, and ArXiv."
)
st.divider()

# ---------------------------------------------------------------------------
# Sidebar — API key & how-it-works
# ---------------------------------------------------------------------------
api_key = os.environ.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.header("Configuration")
    if not api_key:
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Required to power the Gemini LLM referee.",
        )
    else:
        st.success("API key loaded from environment.")
    st.divider()
    st.subheader("How it works")
    st.markdown(
        "1. You submit an argument or claim.\n"
        "2. The agent identifies each factual assertion.\n"
        "3. It **autonomously** queries Wikipedia, DuckDuckGo, and ArXiv.\n"
        "4. A structured, cited fact-check verdict is produced."
    )

if not api_key:
    st.warning(
        "Enter your **GEMINI_API_KEY** in the sidebar or set it as an "
        "environment variable to continue."
    )
    st.stop()

# ---------------------------------------------------------------------------
# LLM initialisation
# ---------------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.3,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are the Debate Arena referee — a meticulous, impartial fact-checker.\n"
    "A user will submit an argument or set of claims.\n\n"
    "YOUR PROCESS:\n"
    "1. Identify every distinct factual claim in the argument.\n"
    "2. For EACH claim, use one or more of your tools to gather real-world evidence:\n"
    "   • search_wikipedia  — established facts, history, definitions\n"
    "   • search_current_web — recent events, news, current data\n"
    "   • search_arxiv_papers — scientific or academic claims\n"
    "3. You MUST call at least two different tools. Be thorough.\n"
    "4. After gathering evidence, write the structured report below.\n\n"
    "REPORT FORMAT:\n"
    "## ⚖️ Fact-Check Report\n\n"
    "### Claims Identified\n"
    "Numbered list of each distinct claim.\n\n"
    "### Evidence & Findings\n"
    "For each claim, present evidence with source citations.\n\n"
    "### Verdict\n"
    "Rate each claim: ✅ **SUPPORTED**, ⚠️ **PARTIALLY SUPPORTED**, or ❌ **REFUTED**\n\n"
    "### Overall Assessment\n"
    "Brief summary of the argument's factual accuracy.\n\n"
    "### Sources\n"
    "Numbered list of every source with URLs."
)

# ---------------------------------------------------------------------------
# Input area
# ---------------------------------------------------------------------------
user_argument = st.text_area(
    "Enter your argument to fact-check:",
    placeholder=(
        "e.g., 'The Great Wall of China is visible from space, "
        "and it was built entirely during the Qin Dynasty.'"
    ),
    height=130,
)

run_clicked = st.button(
    "⚖️  Fact-Check This Argument", type="primary", use_container_width=True
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
TOOL_ICONS = {
    "search_wikipedia": "📚",
    "search_current_web": "🌐",
    "search_arxiv_papers": "🎓",
}

SERVER_SCRIPT = str(Path(__file__).parent / "mcp_server.py")


def _extract_text(content) -> str:
    """Extract plain text from LLM message content.

    Gemini returns content as a list of block dicts
    (e.g. [{'type': 'text', 'text': '…', 'extras': {…}}])
    rather than a plain string.  This normalises both forms.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


async def execute_agent(argument: str, ui_status):
    """Spin up the MCP server, build the agent, and stream steps to *ui_status*."""

    client = MultiServerMCPClient(
        {
            "debate_arena": {
                "command": sys.executable,
                "args": [SERVER_SCRIPT],
                "transport": "stdio",
                "env": dict(os.environ),
            }
        }
    )

    tools = await client.get_tools()
    agent = create_react_agent(llm, tools)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=argument),
    ]

    final_text = ""
    step_counter = 0

    async for chunk in agent.astream(
        {"messages": messages}, stream_mode="updates"
    ):
        # --- Agent node: LLM produced a message --------------------------------
        if "agent" in chunk:
            for msg in chunk["agent"]["messages"]:
                if not isinstance(msg, AIMessage):
                    continue

                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        step_counter += 1
                        name = tc["name"]
                        query = tc.get("args", {}).get("query", "")
                        icon = TOOL_ICONS.get(name, "🔧")
                        ui_status.markdown(
                            f'<div class="tool-card">'
                            f"<strong>Step {step_counter}</strong> &mdash; "
                            f"{icon} <code>{name}</code>"
                            f"&nbsp; → &nbsp;<em>{query}</em></div>",
                            unsafe_allow_html=True,
                        )

                if msg.content and not msg.tool_calls:
                    final_text = _extract_text(msg.content)

        # --- Tools node: tool returned data ------------------------------------
        elif "tools" in chunk:
            for msg in chunk["tools"]["messages"]:
                content = _extract_text(msg.content)
                snippet = (content[:400] + " …") if len(content) > 400 else content
                tool_name = getattr(msg, "name", "tool")
                icon = TOOL_ICONS.get(tool_name, "📊")
                ui_status.markdown(
                    f'<p class="data-label">{icon} Data from <code>{tool_name}</code>:</p>',
                    unsafe_allow_html=True,
                )
                ui_status.code(snippet, language=None)

    return final_text


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if run_clicked:
    if not user_argument.strip():
        st.warning("Please enter an argument before submitting.")
    else:
        with st.status(
            "🔍  Agent reasoning and querying databases…", expanded=True
        ) as status:
            try:
                result = asyncio.run(execute_agent(user_argument, status))
                status.update(label="✅  Analysis complete!", state="complete")
            except Exception as exc:
                status.update(label="❌  Error during analysis", state="error")
                st.error(f"Something went wrong: {exc}")
                st.stop()

        if result:
            st.divider()
            st.markdown(result)
        else:
            st.warning(
                "The agent did not produce a final report. "
                "Try rephrasing your argument."
            )
