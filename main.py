from agent import ReadingThemeAgent


THEMES = [
    "friendship",
    "sisterhood",
    "family",
    "mother-daughter relationship",
    "marriage",
    "motherhood",
    "work and career",
    "money and independence",
    "education",
    "writing and creativity",
    "girlhood",
    "memory",
    "violence",
    "healing",
    "freedom",
    "social expectations",
    "gender norms",
    "identity",
    "survival",
]


HELP_TEXT = """
Commands:
  ask <question>       Retrieve context, then ask the LLM to answer.
  search <query>       Show retrieved chunks without calling the LLM.
  remember <text>      Save a note into local memory.
  memory               Show all saved memory items.
  themes               Show the theme taxonomy.
  help                 Show this help message.
  exit                 Quit the program.
"""


def print_search_results(results):
    if not results:
        print("No matching chunks found.")
        return

    for index, result in enumerate(results, start=1):
        print(f"\n[{index}] source: {result['source']}")
        print(f"score: {result['score']:.3f}")
        print(result["text"])


def main():
    print("Reading Theme Retrieval Agent")
    print("Type 'help' to see available commands.")

    agent = ReadingThemeAgent()

    while True:
        raw_input = input("\n> ").strip()
        if not raw_input:
            continue

        command, _, argument = raw_input.partition(" ")
        command = command.lower()
        argument = argument.strip()

        if command == "exit":
            print("Goodbye.")
            break

        if command == "help":
            print(HELP_TEXT)

        elif command == "themes":
            print("\nThemes:")
            for theme in THEMES:
                print(f"- {theme}")

        elif command == "memory":
            items = agent.memory.load()
            if not items:
                print("Memory is empty.")
            else:
                for item in items:
                    print(f"{item.get('id', '?')}. {item.get('text', '')}")

        elif command == "remember":
            if not argument:
                print("Usage: remember <text>")
                continue
            item = agent.remember(argument)
            print(f"Saved memory item {item['id']}.")

        elif command == "search":
            if not argument:
                print("Usage: search <query>")
                continue
            print_search_results(agent.search(argument))

        elif command == "ask":
            if not argument:
                print("Usage: ask <question>")
                continue
            try:
                response = agent.ask(argument)
            except ValueError as error:
                print(f"\nConfiguration error: {error}")
                print("Copy .env.example to .env and add your API key.")
                continue
            print("\nAnswer:")
            print(response["answer"])
            print(f"\nRetrieval confidence score: {response['confidence']:.3f}")
            print(f"Context strength: {response['strength']}")
            if response["sources"]:
                print("Retrieved source names: " + ", ".join(response["sources"]))
            else:
                print("Retrieved source names: none")

        else:
            print("Unknown command. Type 'help' to see available commands.")


if __name__ == "__main__":
    main()
