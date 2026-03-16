# Debate Arena: The Agentic Fact-Check Engine

An AI-powered fact-checking application that uses an **agentic workflow** to autonomously research and verify claims using live data. Built with the **Model Context Protocol (MCP)** to cleanly separate tool logic from reasoning.

---

## Architecture

```
User  -->  Streamlit UI (app.py)  -->  Gemini 2.5 Flash (ReAct Agent)
                                            |
                                            |  stdio
                                            v
                                     MCP Server (mcp_server.py)
                                       ├── search_wikipedia
                                       ├── search_current_web
                                       └── search_arxiv_papers
```

| Layer | File | Role |
|---|---|---|
| **MCP Server** | `mcp_server.py` | Exposes three real-data tools over stdio via FastMCP |
| **Client & UI** | `app.py` | Streamlit frontend + LangChain/LangGraph agent orchestration |

---

## MCP Tools

| Tool | Source | Returns |
|---|---|---|
| `search_wikipedia` | Wikipedia API | Page summary + source URL |
| `search_current_web` | DuckDuckGo (ddgs) | Top 3 results with title, snippet, URL |
| `search_arxiv_papers` | ArXiv API | Top 2 papers with title, authors, abstract, URL |

All tools include error handling for disambiguation, missing pages, and network failures.

---

## Tech Stack

- **LLM** — Google Gemini 2.5 Flash (`langchain-google-genai`)
- **Agent Framework** — LangGraph ReAct agent (`langgraph`)
- **MCP Integration** — `langchain-mcp-adapters` (stdio transport)
- **MCP Server** — `fastmcp`
- **Data Sources** — `wikipedia`, `ddgs`, `arxiv`
- **Frontend** — Streamlit (dark-mode theme)

---

## Setup

### 1. Install dependencies

```bash
cd DebateArena
pip install -r requirements.txt
```

### 2. Set your API key

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

Or enter it directly in the app sidebar.

### 3. Run

```bash
streamlit run app.py
```

---

## How It Works

1. **Submit** an argument or set of claims.
2. The ReAct agent **identifies** each factual assertion.
3. It **autonomously selects and calls** MCP tools — Wikipedia, DuckDuckGo, ArXiv — to gather real-world evidence.
4. Every tool call is **streamed live** to the UI inside an expandable status panel (tool name, search query, data snippet).
5. A structured **Fact-Check Report** is rendered with per-claim verdicts and cited sources.

---

## Project Structure

```
DebateArena/
├── .streamlit/
│   └── config.toml        # Dark-mode theme configuration
├── app.py                  # Streamlit client + LangGraph agent
├── mcp_server.py           # FastMCP server (3 tools, stdio)
└── requirements.txt        # Python dependencies
```

---

## Example

**Input:**
> "The Great Wall of China is visible from space, and it was built entirely during the Qin Dynasty."

**Agent workflow (visible in UI):**
```
Step 1 — search_wikipedia  →  "Great Wall of China"
Step 2 — search_current_web →  "Great Wall visible from space myth"
Step 3 — search_wikipedia  →  "Qin Dynasty Great Wall construction"
```

**Output:** A structured report rating each claim as SUPPORTED, PARTIALLY SUPPORTED, or REFUTED — with sources.
