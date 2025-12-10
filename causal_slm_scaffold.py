"""
Causal Reasoning Scaffolding for SLMs (e.g., Phi-4 Mini-Reasoning)

Implements:
  1. Variable Extraction
  2. Causal Graph / SCM Construction
  3. Interventional Reasoning (do-calculus)
  4. Counterfactual Reasoning

All interaction with the SLM is done via a single injected function:
    slm_call_fn(prompt: str) -> str (can be async)
"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable, Awaitable


# ============================================================================
# STAGE 1: DATA STRUCTURES & ENUMS
# ============================================================================

class VariableRole(Enum):
    """Causal variable roles in a graph."""
    CAUSE = "cause"
    EFFECT = "effect"
    CONFOUNDER = "confounder"
    MEDIATOR = "mediator"
    COLLIDER = "collider"
    INDEPENDENT = "independent"


@dataclass
class CausalVariable:
    """Typed variable representation."""
    name: str
    role: VariableRole
    domain: str          # e.g., "binary {0,1}", "continuous [0,100]"
    description: str     # semantic meaning

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role.value,
            "domain": self.domain,
            "description": self.description,
        }


@dataclass
class CausalEdge:
    """Directed causal relationship."""
    source: str          # cause
    target: str          # effect
    justification: str
    mechanism: Optional[str] = None   # "direct" | "mediated" | "confounded" | ...
    strength: Optional[str] = None    # "weak" | "moderate" | "strong"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "justification": self.justification,
            "mechanism": self.mechanism,
            "strength": self.strength,
        }


@dataclass
class CausalGraph:
    """Structural Causal Model (SCM) representation."""
    variables: List[CausalVariable]
    edges: List[CausalEdge]
    scm_equations: Optional[Dict[str, str]] = None  # e.g., {"Y": "f(X,Z) + U_Y"}

    def to_adjacency_list(self) -> Dict[str, List[str]]:
        """Convert to adjacency list for traversal."""
        adj = {v.name: [] for v in self.variables}
        for e in self.edges:
            if e.source in adj:
                adj[e.source].append(e.target)
            else:
                adj[e.source] = [e.target]
        return adj

    def get_parents(self, var: str) -> List[str]:
        return [e.source for e in self.edges if e.target == var]

    def get_children(self, var: str) -> List[str]:
        return [e.target for e in self.edges if e.source == var]

    def to_dag_representation(self) -> List[str]:
        return [f"{e.source} -> {e.target}" for e in self.edges]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variables": [v.to_dict() for v in self.variables],
            "edges": [e.to_dict() for e in self.edges],
            "scm_equations": self.scm_equations,
        }


@dataclass
class InterventionalQuery:
    """Intervention-level causal question: do(X = x')."""
    intervention_var: str
    intervention_value: str
    query_var: str
    expected_effect: Optional[str] = None        # optional label
    reasoning_path: Optional[List[str]] = None   # optional path(s)


@dataclass
class CounterfactualQuery:
    """Counterfactual-level causal question."""
    observed_values: Dict[str, str]  # actual observations
    counterfactual_var: str
    counterfactual_value: str
    query_var: str
    expected_cf_value: Optional[str] = None      # optional label


# ============================================================================
# STAGE 2: PROMPTS FOR SLM SCAFFOLDING
# ============================================================================

class CausalPrompts:
    """Prompt templates for each scaffolding stage."""

    @staticmethod
    def prompt_1_variable_extraction(problem: str) -> str:
        """Prompt 1: Causal Variable Extraction."""
        return f"""
Task: Causal Variable Extraction
You are given a causal problem. Extract ALL relevant variables and their roles.

Problem:
{problem}

Output a JSON list with this structure (valid JSON only):
[
  {{
    "name": "variable_name",
    "role": "cause|effect|confounder|mediator|collider|independent",
    "domain": "description of possible values (e.g., binary {{0,1}}, continuous [0,100])",
    "description": "semantic meaning in the problem"
  }},
  ...
]

Rules:
1. Be exhaustive: include all entities mentioned or clearly implied.
2. Domain must be specific and realistic (not vague).
3. Role must match the variable's position in causal chains.
4. Output ONLY valid JSON, no extra text, no comments.

Extract variables now:
""".strip()

    @staticmethod
    def prompt_2_graph_construction(
        problem: str,
        variables: List[CausalVariable],
    ) -> str:
        """Prompt 2: Causal Graph / SCM Sketch."""
        vars_str = json.dumps([v.to_dict() for v in variables], indent=2)
        return f"""
Task: Causal Graph Construction
You have extracted these variables:
{vars_str}

Problem context:
{problem}

Now propose a DIRECTED causal graph (a DAG: no directed cycles).

Output JSON with this structure:
{{
  "edges": [
    {{
      "source": "var1",
      "target": "var2",
      "justification": "why this causal relationship exists",
      "mechanism": "direct|mediated|confounded",
      "strength": "weak|moderate|strong"
    }},
    ...
  ],
  "scm_equations": {{
    "var_name": "mathematical or functional form, e.g., f(parents) + noise",
    ...
  }}
}}

Rules:
1. Each edge is a directed relationship (cause → effect).
2. Justify EVERY edge from the problem context.
3. Avoid cycles (must form a DAG).
4. Include SCM-style equations where possible (even informal).
5. Output ONLY valid JSON, no extra text.

Construct the causal graph:
""".strip()

    @staticmethod
    def prompt_3_interventional_reasoning(
        problem: str,
        graph: CausalGraph,
        intervention: InterventionalQuery,
    ) -> str:
        """Prompt 3: Interventional Reasoning (do-calculus)."""
        graph_str = json.dumps(graph.to_dict(), indent=2)
        dag_representation = graph.to_dag_representation()
        dag_str = json.dumps(dag_representation, indent=2)
        return f"""
Task: Interventional Causal Reasoning (do-calculus)
You have this causal graph (SCM):
{graph_str}

DAG representation (for reference):
{dag_str}

Problem context:
{problem}

Query: Under the intervention do({intervention.intervention_var} = {intervention.intervention_value}),
what do you expect to happen to {intervention.query_var}?

Instructions:
1. Use ONLY the causal graph above (do NOT use any external or real-world knowledge).
2. Trace all directed paths from {intervention.intervention_var} to {intervention.query_var}.
3. Explain the causal mechanism step-by-step along each path.
4. Identify confounders or mediators that might affect this relationship.
5. Conclude with the expected change in {intervention.query_var}.

Output JSON:
{{
  "causal_paths": [
    "path1: {intervention.intervention_var} -> ... -> {intervention.query_var}",
    ...
  ],
  "mechanisms": [
    "explanation of how each path affects {intervention.query_var}",
    ...
  ],
  "confounders": ["list of backdoor confounders if any"],
  "expected_effect": "increase|decrease|no_change|unknown",
  "reasoning_summary": "2-3 sentence conclusion"
}}

Reason about the intervention now and output ONLY valid JSON:
""".strip()

    @staticmethod
    def prompt_4_counterfactual_reasoning(
        problem: str,
        graph: CausalGraph,
        cf_query: CounterfactualQuery,
    ) -> str:
        """Prompt 4: Counterfactual-style reasoning (Level 3)."""
        graph_str = json.dumps(graph.to_dict(), indent=2)
        obs_str = json.dumps(cf_query.observed_values, indent=2)
        return f"""
Task: Counterfactual Reasoning (anchored to the SCM)
You have this causal graph (SCM):
{graph_str}

Problem context:
{problem}

Observed data:
{obs_str}

Counterfactual Query:
Given we observed these values, what would {cf_query.query_var} have been
if {cf_query.counterfactual_var} had been {cf_query.counterfactual_value} instead?

Instructions:
1. Use the SCM structure to reason about counterfactuals.
2. Identify which variables would change given the counterfactual intervention.
3. Trace through the graph to determine how {cf_query.query_var} would be affected.
4. Keep exogenous noise terms (U_*) constant, as in standard SCM counterfactuals.
5. Provide clear, step-by-step reasoning.

Output JSON:
{{
  "changed_variables": {{"var": "new_value_under_counterfactual", ...}},
  "unchanged_variables": ["vars that remain the same"],
  "reasoning_steps": ["step 1", "step 2", ...],
  "counterfactual_value": "predicted value of {cf_query.query_var} under the counterfactual",
  "confidence": "high|medium|low"
}}

Reason about the counterfactual and output ONLY valid JSON:
""".strip()


# ============================================================================
# STAGE 3: PIPELINE ORCHESTRATOR
# ============================================================================

class CausalReasoningPipeline:
    """
    End-to-end pipeline for SLM-based causal reasoning.

    slm_call_fn: async function(prompt: str) -> str
       - wraps your Phi-4 mini (or other) model.
    """

    def __init__(self, slm_call_fn: Callable[[str], Awaitable[str]]):
        self.slm_call = slm_call_fn
        self.extracted_variables: Optional[List[CausalVariable]] = None
        self.causal_graph: Optional[CausalGraph] = None
        self.interventional_results: List[Dict[str, Any]] = []
        self.counterfactual_results: List[Dict[str, Any]] = []

    # ---------------- Stage 1 ----------------

    async def stage_1_extract_variables(self, problem: str) -> List[CausalVariable]:
        """Stage 1: Variable extraction via SLM."""
        prompt = CausalPrompts.prompt_1_variable_extraction(problem)
        response = await self.slm_call(prompt)

        try:
            raw = self._extract_json(response)

            # Accept both:
            #   - a bare list: [ {...}, {...} ]
            #   - an object with a "variables" field: { "variables": [ {...}, ... ] }
            if isinstance(raw, dict):
                if "variables" in raw and isinstance(raw["variables"], list):
                    var_dicts = raw["variables"]
                else:
                    # If the dict itself looks like a single variable, wrap it
                    var_dicts = [raw]
            elif isinstance(raw, list):
                var_dicts = raw
            else:
                raise TypeError(f"Unexpected JSON type for variables: {type(raw)}")

            self.extracted_variables = [
                CausalVariable(
                    name=v["name"],
                    role=VariableRole(v["role"]),
                    domain=v["domain"],
                    description=v["description"],
                )
                for v in var_dicts
            ]
            return self.extracted_variables
        except Exception as e:
            raise ValueError(f"Failed to parse variables from SLM output: {e}")

    # ---------------- Stage 2 ----------------

    async def stage_2_construct_graph(self, problem: str) -> CausalGraph:
        """Stage 2: Causal graph construction via SLM."""
        if not self.extracted_variables:
            raise ValueError("stage_1_extract_variables must be run before stage_2_construct_graph")

        prompt = CausalPrompts.prompt_2_graph_construction(problem, self.extracted_variables)
        response = await self.slm_call(prompt)

        try:
            graph_dict = self._extract_json(response)

            # Accept either full object or nested under a key
            if not isinstance(graph_dict, dict):
                raise TypeError(f"Expected dict for graph JSON, got {type(graph_dict)}")

            edges_raw = graph_dict.get("edges", [])
            if not isinstance(edges_raw, list):
                raise TypeError("graph_dict['edges'] must be a list")

            edges = [
                CausalEdge(
                    source=e["source"],
                    target=e["target"],
                    justification=e["justification"],
                    mechanism=e.get("mechanism"),
                    strength=e.get("strength"),
                )
                for e in edges_raw
            ]
            scm_eqs = graph_dict.get("scm_equations")
            self.causal_graph = CausalGraph(
                variables=self.extracted_variables,
                edges=edges,
                scm_equations=scm_eqs,
            )
            self._validate_dag()
            return self.causal_graph
        except Exception as e:
            raise ValueError(f"Failed to parse graph from SLM output: {e}")

    # ---------------- Stage 3 ----------------

    async def stage_3_interventional_reasoning(
        self,
        problem: str,
        intervention: InterventionalQuery,
    ) -> Dict[str, Any]:
        """Stage 3: Interventional query answering."""
        if not self.causal_graph:
            raise ValueError("stage_2_construct_graph must be run before stage_3_interventional_reasoning")

        prompt = CausalPrompts.prompt_3_interventional_reasoning(
            problem, self.causal_graph, intervention
        )
        response = await self.slm_call(prompt)

        try:
            result = self._extract_json(response)
            if not isinstance(result, dict):
                raise TypeError(f"Expected dict for interventional result, got {type(result)}")
            self.interventional_results.append(result)
            return result
        except Exception as e:
            raise ValueError(f"Failed to parse interventional reasoning from SLM output: {e}")

    # ---------------- Stage 4 ----------------

    async def stage_4_counterfactual_reasoning(
        self,
        problem: str,
        cf_query: CounterfactualQuery,
    ) -> Dict[str, Any]:
        """Stage 4: Counterfactual query answering."""
        if not self.causal_graph:
            raise ValueError("stage_2_construct_graph must be run before stage_4_counterfactual_reasoning")

        prompt = CausalPrompts.prompt_4_counterfactual_reasoning(
            problem, self.causal_graph, cf_query
        )
        response = await self.slm_call(prompt)

        try:
            result = self._extract_json(response)
            if not isinstance(result, dict):
                raise TypeError(f"Expected dict for counterfactual result, got {type(result)}")
            self.counterfactual_results.append(result)
            return result
        except Exception as e:
            raise ValueError(f"Failed to parse counterfactual reasoning from SLM output: {e}")

    # ---------------- Utilities ----------------

    @staticmethod
    def _extract_json(text: str) -> Any:
        """
        Extract JSON from an SLM response.

        Handles:
        - pure JSON
        - ```json ... ``` fenced blocks
        - largest {...} or [...] block in text
        """
        # 1) Try direct JSON
        try:
            return json.loads(text)
        except Exception:
            pass

        # 2) Try fenced code block ```json ... ```
        fence_pattern = r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```"
        m = re.search(fence_pattern, text, flags=re.DOTALL)
        if m:
            candidate = m.group(1)
            try:
                return json.loads(candidate)
            except Exception:
                pass

        # 3) Try largest [...] block
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                candidate = text[start: end + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    continue

        raise ValueError("Could not extract valid JSON from SLM response")

    def _validate_dag(self) -> None:
        """Check for cycles (DAG constraint) using DFS."""
        if not self.causal_graph:
            return

        adj = self.causal_graph.to_adjacency_list()
        visited = set()
        stack = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            stack.add(node)
            for nxt in adj.get(node, []):
                if nxt not in visited:
                    if dfs(nxt):
                        return True
                elif nxt in stack:
                    return True
            stack.remove(node)
            return False

        for node in adj:
            if node not in visited:
                if dfs(node):
                    raise ValueError("Causal graph contains a directed cycle (not a DAG)")

    def to_report(self) -> Dict[str, Any]:
        """Return a comprehensive JSON-like report of the whole run."""
        return {
            "extracted_variables": [v.to_dict() for v in (self.extracted_variables or [])],
            "causal_graph": self.causal_graph.to_dict() if self.causal_graph else None,
            "interventional_results": self.interventional_results,
            "counterfactual_results": self.counterfactual_results,
        }


# ============================================================================
# STAGE 4: EVALUATION & METRICS
# ============================================================================

class CausalEvaluator:
    """Evaluate correctness of causal reasoning against a gold standard."""

    @staticmethod
    def graph_f1(
        predicted_edges: List[Tuple[str, str]],
        gold_edges: List[Tuple[str, str]],
    ) -> float:
        """Compute F1 score for edge recovery."""
        pred_set = set(predicted_edges)
        gold_set = set(gold_edges)

        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        if tp + fp == 0 or tp + fn == 0:
            return 0.0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def evaluate_interventional(
        predicted_effect: str,
        gold_effect: str,
        reasoning_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate interventional reasoning accuracy.

        predicted_effect / gold_effect in {"increase","decrease","no_change","unknown"}.
        """
        effect_match = predicted_effect.lower().strip() == gold_effect.lower().strip()
        return {
            "effect_accuracy": 1.0 if effect_match else 0.0,
            "reasoning_quality": len(reasoning_paths) if reasoning_paths else 0,
        }

    @staticmethod
    def evaluate_counterfactual(
        predicted_value: str,
        gold_value: str,
        reasoning_steps: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate counterfactual reasoning accuracy.

        Tries numeric comparison first, falls back to string equality.
        """
        match = False
        try:
            p = float(str(predicted_value).strip())
            g = float(str(gold_value).strip())
            match = abs(p - g) < 1e-1
        except Exception:
            match = str(predicted_value).strip().lower() == str(gold_value).strip().lower()

        return {
            "value_accuracy": 1.0 if match else 0.0,
            "reasoning_depth": len(reasoning_steps) if reasoning_steps else 0,
        }


# ============================================================================
# OPTIONAL: SIMPLE SMOKE TEST
# ============================================================================

if __name__ == "__main__":
    print("Causal Reasoning Scaffolding module loaded.")
    print("Import CausalReasoningPipeline and inject your Phi-4 mini call function.")
