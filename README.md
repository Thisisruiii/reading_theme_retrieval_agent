# Reading Theme Retrieval Agent

Reading Theme Retrieval Agent is a simple command-line AI agent for an Information Retrieval lab. It retrieves relevant context from local reading notes, a theme guide, recommendation rules, and user memory before asking an OpenAI-compatible LLM to answer.

The project uses women's literature as a thematic reading taxonomy. The goal is to demonstrate retrieval and agent architecture, not to build a political discussion bot.

## Why this is an AI agent

This program is more than a normal chatbot because it follows an agent-like workflow:

1. It receives a user command.
2. It chooses an action such as `ask`, `search`, `remember`, or `themes`.
3. For questions, it retrieves relevant local context first.
4. It checks retrieval confidence.
5. It calls the LLM only when the retrieved context is strong or weak enough to support an answer.
6. It returns the answer together with retrieval metadata.

## Information Retrieval method

The project uses TF-IDF retrieval with cosine similarity.

- `documents/book_notes.txt` contains sample book notes.
- `documents/theme_guide.txt` explains the theme taxonomy.
- `documents/recommendation_rules.txt` gives behavior rules for recommendations.
- `memory_store/memory.json` stores user memories added with the `remember` command.

All these texts are converted into searchable chunks.

## How TF-IDF retrieval works here

`retriever.py` loads `.txt` files from the `documents` folder and splits them into small overlapping chunks. It also receives memory chunks from `memory.py`.

The retriever uses `scikit-learn`:

- `TfidfVectorizer` converts chunks into TF-IDF vectors.
- The user query is converted into another vector.
- `cosine_similarity` compares the query vector with every chunk vector.
- The top-k most similar chunks are returned.

The `search <query>` command shows this process without calling the LLM.

## How memory works

The command:

```bash
remember <text>
```

saves a user note into `memory_store/memory.json`.

Memory items are loaded into the same TF-IDF index as the document chunks. This means a later `ask` or `search` command can retrieve both book notes and personal memory.

## Confidence-aware retrieval

The agent uses a two-step confidence mechanism.

First, it computes a numeric retrieval confidence score from the highest TF-IDF cosine similarity score:

- `top score >= 0.25`: base score strength is `strong`
- `0.10 <= top score < 0.25`: base score strength is `weak`
- `top score < 0.10`: base score strength is `unreliable`

Second, it applies a source-aware confidence check. Since this project is designed for book-note retrieval, book discussion questions are only treated as reliable when the retrieved sources include `book_notes.txt`. This prevents the agent from answering confidently when the query only matches general guide files such as `theme_guide.txt` or `recommendation_rules.txt`.

The final output therefore includes:

- numeric retrieval confidence score
- base score strength
- final context strength
- retrieved source names

## Project structure

```text
main.py
agent.py
retriever.py
memory.py
llm_client.py
documents/
  book_notes.txt
  theme_guide.txt
  recommendation_rules.txt
memory_store/
  memory.json
.env.example
.gitignore
requirements.txt
README.md
```

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Create `.env`

Copy `.env.example` to `.env`:

```bash
copy .env.example .env
```

Then edit `.env`:

```text
OPENAI_API_KEY=your_berget_api_key_here
OPENAI_BASE_URL=https://api.berget.ai/v1
OPENAI_MODEL=openai/gpt-oss-120b
```

You can replace this with another Berget.AI chat model, such as `google/gemma-4-31b-it`, if it is available in your account.

## Run the project

```bash
python main.py
```

## Example commands

```text
help
themes
search sisterhood and healing
ask Which books discuss writing and independence?
remember I am especially interested in mother-daughter relationships and memory.
search mother daughter memory
memory
exit
```

The `ask` command prints:

- answer
- retrieval confidence score
- context strength
- retrieved source names

## Suggested video demo

The demo can show:

1. Run `python main.py`.
2. Run `themes` to show the theme taxonomy.
3. Run `search writing independence` to show TF-IDF retrieval.
4. Run `ask Which books are useful for discussing sisterhood and healing?` to show retrieved-context answering.
5. Run `remember I want to focus on books about family memory.`
6. Run `memory` or `search family memory` to show memory retrieval.
7. Run `ask Which books discuss space travel?` to show that unrelated context is downgraded to `unreliable`.

## Berget.AI and OpenAI-compatible API notes

This project uses the official OpenAI Python SDK, but the API base URL is configurable. For Berget.AI, set:

```text
OPENAI_BASE_URL=https://api.berget.ai/v1
OPENAI_MODEL=openai/gpt-oss-120b
```

Any provider with an OpenAI-compatible chat completions endpoint should work if the API key, base URL, and model name are correct.

## Possible future extensions

- Add web search as an optional tool.
- Add conversation compaction for long sessions.
- Add a heartbeat loop that checks whether the agent needs to refresh memory.
- Add a visualization of TF-IDF scores and retrieved chunks.
- Add more books and richer note metadata.
- Add evaluation queries for measuring retrieval quality.
