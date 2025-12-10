#!/usr/bin/env python3
import asyncio
import json
import httpx
from typing import Dict, Any

# ------------------------------
# VISUALIZATION IMPORTS & MAC SETUP
# ------------------------------
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

# MacOS often requires a specific backend to avoid crashing if not using the system Python
try:
    matplotlib.use('TkAgg')
except Exception:
    pass # Fallback to default if TkAgg isn't available

# Import pipeline and query classes
# Ensure 'causal_slm_scaffold-1.py' is renamed to 'causal_slm_scaffold.py' 
# or that the import matches your filename.
from causal_slm_scaffold import (
    CausalReasoningPipeline,
    InterventionalQuery,
    CounterfactualQuery,
)

# ------------------------------
# CONFIGURATION
# ------------------------------
OLLAMA_MODEL = "phi4-mini-reasoning"
# Standard Ollama URL for macOS
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"


# ------------------------------
# OLLAMA CALL FUNCTION
# ------------------------------
async def ollama_call_fn(prompt: str) -> str:
    """
    Sends a prompt to Ollama using /api/chat.
    """

    print("\n" + "=" * 20 + " PROMPT TO SLM " + "=" * 20)
    print(prompt)
    print("=" * 55 + "\n")

    # Increased timeout to 300s for local inference on Apple Silicon/Intel
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }

            response = await client.post(OLLAMA_CHAT_URL, json=payload)
            response.raise_for_status()

            data = response.json()
            model_output = data["message"]["content"]

            print("\n" + "=" * 20 + " SLM RESPONSE " + "=" * 20)
            print(model_output)
            print("=" * 56 + "\n")

            return model_output

        except httpx.ConnectError:
            print(f"\n❌ ERROR: Could not connect to Ollama at {OLLAMA_CHAT_URL}")
            print("Action: Open Terminal and run `ollama serve`")
            raise
        except Exception as e:
            print(f"\n❌ ERROR contacting Ollama: {e}")
            raise e


# ------------------------------
# UTILITY PRINT FUNCTION
# ------------------------------
def pretty_print_json(obj: Any):
    print(json.dumps(obj, indent=2))


# ------------------------------
# GRAPH DRAWING FUNCTION
# ------------------------------
def draw_causal_graph(graph):
    """
    Visualizes the SCM DAG.
    NOTE: On macOS, this requires a working window manager.
    """
    try:
        G = nx.DiGraph()

        # Add nodes
        for v in graph.variables:
            G.add_node(v.name)

        # Add edges
        for e in graph.edges:
            G.add_edge(e.source, e.target)

        # Layout
        pos = nx.spring_layout(G, seed=42)

        plt.figure(figsize=(10, 7))
        nx.draw(
            G, pos,
            with_labels=True,
            node_size=2500,
            node_color="skyblue",
            font_size=10,
            arrows=True,
            font_weight="bold"
        )
        plt.title("Causal Graph (SCM DAG)")
        print("Opening graph window...")
        plt.show()
    except Exception as e:
        print(f"Could not open visual graph window: {e}")


# ------------------------------
# MAIN INTERACTIVE LOOP
# ------------------------------
async def main():
    print("--- Causal Reasoning Interactive Chat (macOS) ---")
    print(f"Targeting Ollama at: {OLLAMA_CHAT_URL}")

    problem_description = input("Enter the causal problem description:\n> ")

    pipeline = CausalReasoningPipeline(slm_call_fn=ollama_call_fn)

    # ------------------------------
    # SETUP: VARIABLE & GRAPH EXTRACTION
    # ------------------------------
    try:
        print("\n--- STAGE 1: Extracting Variables ---")
        await pipeline.stage_1_extract_variables(problem_description)
        print(f"Extracted {len(pipeline.extracted_variables)} variables.")

        print("\n--- STAGE 2: Constructing Graph ---")
        await pipeline.stage_2_construct_graph(problem_description)
        print("✔ Causal graph constructed successfully.")

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        return

    print("\n--- You may now ask questions. ---")

    # ------------------------------
    # INTERACTIVE MENU
    # ------------------------------
    while True:
        print("\nWhat would you like to do?")
        print("  1. Interventional query (e.g., do(X=1))")
        print("  2. Counterfactual query")
        print("  3. Show causal graph (Adjacency List DAG)")
        print("  4. Exit")
        choice = input("> ")

        # ------------------------------
        # INTERVENTIONAL QUERY
        # ------------------------------
        if choice == "1":
            print("\n--- Interventional Query ---")
            var = input("Intervene on variable: ")
            val = input(f"Set {var} to: ")
            query = input("Query effect on variable: ")

            iq = InterventionalQuery(
                intervention_var=var,
                intervention_value=val,
                query_var=query
            )

            try:
                result = await pipeline.stage_3_interventional_reasoning(
                    problem_description, iq
                )
                print("\n--- RESULT ---")
                pretty_print_json(result)
            except Exception as e:
                print(f"❌ Error: {e}")

        # ------------------------------
        # COUNTERFACTUAL QUERY
        # ------------------------------
        elif choice == "2":
            print("\n--- Counterfactual Query ---")
            print("Enter observed values (empty name to finish):")

            observed = {}
            while True:
                name = input("Observed variable: ")
                if not name:
                    break
                val = input(f"Value for {name}: ")
                observed[name] = val

            if not observed:
                print("You must enter at least 1 observation.")
                continue

            cf_var = input("Counterfactual variable: ")
            cf_val = input(f"What if {cf_var} = ")
            target = input("Query effect on variable: ")

            cfq = CounterfactualQuery(
                observed_values=observed,
                counterfactual_var=cf_var,
                counterfactual_value=cf_val,
                query_var=target
            )

            try:
                result = await pipeline.stage_4_counterfactual_reasoning(
                    problem_description, cfq
                )
                print("\n--- RESULT ---")
                pretty_print_json(result)
            except Exception as e:
                print(f"❌ Error: {e}")

        # ------------------------------
        # SHOW GRAPH (DAG ADJACENCY LIST)
        # ------------------------------
        elif choice == "3":
            print("\n--- Displaying Causal Graph Adjacency List ---")
            if pipeline.causal_graph:
                # Returns the DAG as a dictionary {Node: [Children]}
                adj_list = pipeline.causal_graph.to_adjacency_list() 
                pretty_print_json(adj_list)
                
                # To see the visual plot, uncomment the line below:
                # draw_causal_graph(pipeline.causal_graph)
            else:
                print("Graph not available.")

        # ------------------------------
        # EXIT
        # ------------------------------
        elif choice == "4":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Enter 1–4.")


# ------------------------------
# ENTRY POINT
# ------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting.")