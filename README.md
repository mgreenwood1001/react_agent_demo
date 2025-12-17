# ReAct Agent

A minimal implementation of the [ReAct (Reasoning + Acting)](https://arxiv.org/abs/2210.03629) pattern using OpenAI's function calling API.

## Features

- **Wikipedia Search** — Look up factual information from Wikipedia
- **Calculator** — Safely evaluate mathematical expressions
- **ReAct Loop** — The agent reasons step-by-step, calling tools as needed until it reaches a final answer

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the agent interactively:

```bash
python react_agent.py
```

Or import and use programmatically:

```python
from react_agent import run

answer = run("What year was Python created and what is 2024 minus that year?")
print(answer)
```

## How It Works

The agent follows the ReAct pattern:

1. **Reason** — The model thinks about what information it needs
2. **Act** — It calls a tool (search or calc) to gather information
3. **Observe** — The tool result is added to the conversation
4. **Repeat** — Until the model has enough information to provide a final answer

## Available Tools

| Tool | Description |
|------|-------------|
| `search(query)` | Search Wikipedia and return a summary |
| `calc(expression)` | Evaluate a math expression (e.g., `(3+4)*5`) |

## Example

```
Question: When was the first public demonstration of the World Wide Web? Also compute 1991 - 1989.

Agent calls: search("first public demonstration World Wide Web")
Agent calls: calc("1991 - 1989")

Final Answer: The World Wide Web was first publicly demonstrated in December 1990. 1991 - 1989 = 2.
```

## License

MIT

