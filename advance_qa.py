import pickle
import networkx as nx
import numpy as np
import re
import json
import os  
import sys
import time
from google.api_core import exceptions as google_exceptions
import google.generativeai as genai 

try:
    genai.configure(api_key="AIzaSyCPCA8vt_yO26JIVcytqo-hxlIGpHmY1zc")
except KeyError:
    print("Error: GEMINI_API_KEY issuse.")

model = genai.GenerativeModel(
    'gemini-2.5-pro',
)

json_generation_config = genai.GenerationConfig(
    temperature=0.0,
    response_mime_type="application/json",
)

code_generation_config = genai.GenerationConfig(
    temperature=0.0,
    response_mime_type="text/plain",  
)
class SetEncoder(json.JSONEncoder):
    """ A custom JSON encoder that converts set objects to lists. """
    def default(self, obj):
        if isinstance(obj, set):
            # Convert set to list
            return list(obj)
        # Let the base class default method handle other types
        return super().default(obj)

def generate_with_retry(model, messages, generation_config, max_retries=3):
    """
    Wraps model.generate_content with automatic retry logic for 429 errors.
    """
    retries = 0
    while retries < max_retries:
        try:
            response = model.generate_content(
                messages,
                generation_config=generation_config
            )
            return response
        except google_exceptions.ResourceExhausted as e:
            print(f"Warning: Rate limit hit. {e.message}", file=sys.stderr)
            
            retry_delay = 30  # default wait time
            match = re.search(r'retry_delay {\s*seconds: (\d+)\s*}', str(e))
            if match:
                retry_delay = int(match.group(1)) + 1
            else:
                match = re.search(r'Please retry in (\d+\.?\d*)s', str(e.message))
                if match:
                    retry_delay = int(float(match.group(1))) + 1

            if retries < max_retries - 1:
                print(f"Waiting for {retry_delay} seconds before retrying...", file=sys.stderr)
                time.sleep(retry_delay)
                retries += 1
            else:
                print("Max retries reached. Raising exception.", file=sys.stderr)
                raise e
        except Exception as e:
            print(f"An unexpected error occurred: {e}", file=sys.stderr)
            raise e

graph_struct='''***GRAPH STRUCTURE
1. **Node Types and Attributes**:
    - task (ID: task_<hash>):
      - node_type: 'task'
      - label: 'Task: <description_snippet>...'
      - description: Full task description string.
      - trajectories: Set of trajectory IDs (e.g., {1, 2, 5}).

    - trajectory (ID: traj_<traj_id>):
      - node_type: 'trajectory'
      - label: 'Trajectory <traj_id> (<status>)'
      - traj_id: The ID of the trajectory.
      - status: Final status of the trajectory (e.g., 'Success', 'Error').
      - final_output: The final output string of the trajectory.

    - step (ID: <traj_id>_step<step_id>):
      - node_type: 'step'
      - label: 'Step <step_id> - <agent_name>'
      - node: Name of the agent performing the step (e.g., 'End', 'Planner').
      - traj_id: The trajectory ID this step belongs to.
      - status: Status of the step.
      - total_time: Placeholder (e.g., 0.0).

    - thought / action / observation (ID: <step_id>_<type> or merged ID):
      - node_type: 'thought', 'action', or 'observation'.
      - label: '<Type>: <text_snippet>...'
      - full_text: The complete text content.
      - referenced_by: Set of step IDs that link to this node (e.g., {'1_step1', '2_step3'}).
      - statuses: Set of statuses from all referencing steps.
      - texts: List of raw text variants that were merged into this node.

    - tool (ID: tool_<tool_name>):
      - node_type: 'tool'
      - label: 'Tool: <tool_name>'
      - tool_name: The name of the tool (e.g., 'search_google').
      - calls: A **list of dictionaries**, where each dict is a specific call instance.
               (e.g., [{'traj_id': 1, 'step_id': 2, 'status': 'Success', 'input_query': '...', 'output_content_raw': '...'}, ...])
      - statuses: Set of all statuses from all calls (e.g., {'Success', 'Error'}).
      - referenced_by: Set of step IDs that called this tool.

2. **Edge Specifications**:
    | Edge Type         | From → To                      | Description                               |
    |-------------------|--------------------------------|-------------------------------------------|
    | has_trajectory    | Task → Trajectory              | Connects a task to its runs.              |
    | has_step          | Trajectory → Step              | Connects a trajectory to its steps.       |
    | next_step         | Step → Step                    | Connects steps in sequential order.       |
    | has_thought       | Step → Thought                 | Connects a step to its thought.           |
    | has_action        | Step → Action                  | Connects a step to its action.            |
    | has_observation   | Step → Observation             | Connects a step to its observation.       |
    | uses_tool         | Step → Tool                    | Connects a step to the tool it called.    |
*****'''

def parse_node_id(node_id):
    """Extracts components from node IDs"""
    
    if node_id.startswith('task_'):
        return {"type": "task"}
    
    if node_id.startswith('traj_'):
        try:
            return {"type": "trajectory", "traj_id": int(node_id.split('_')[1])}
        except (ValueError, IndexError):
            return {"type": "trajectory"}

    if '_step' in node_id and not (node_id.endswith('_thought') or node_id.endswith('_action') or node_id.endswith('_observation')):
        try:
            parts = node_id.split('_step')
            return {
                "type": "step", 
                "traj_id": int(parts[0]), 
                "step_id": int(parts[1])
            }
        except (ValueError, IndexError, TypeError):
             return {"type": "step"}

    if node_id.endswith('_thought') or node_id.endswith('_action') or node_id.endswith('_observation'):
        return {"type": "subnode"} # General category

    if node_id.startswith('tool_'):
        return {"type": "tool", "tool_name": node_id.split('_', 1)[1]}
    
    return {"type": "unknown"}

def get_nodes_by_type(graph, node_type):
    return [(nid, attrs) for nid, attrs in graph.nodes(data=True) 
            if attrs.get('node_type') == node_type]

def get_nodes_by_attribute(graph, attribute, value):
    return [(nid, attrs) for nid, attrs in graph.nodes(data=True) 
            if attrs.get(attribute) == value]

def get_nodes_by_status(graph, status):
    """Finds nodes that have the given status in their 'status' or 'statuses' attribute."""
    results = []
    for nid, attrs in graph.nodes(data=True):
        node_status = attrs.get('status')
        node_statuses = attrs.get('statuses')
        if node_status == status:
            results.append((nid, attrs))
        elif isinstance(node_statuses, set) and status in node_statuses:
            results.append((nid, attrs))
    return results

def get_step_nodes(graph, traj_id, step_id):
    """Finds a specific step node by its traj_id and step_id."""
    # The ID is a string combination
    node_id = f"{traj_id}_step{step_id}"
    if node_id in graph.nodes:
        return [(node_id, graph.nodes[node_id])]
    return []

def get_tool_nodes_by_name(graph, tool_name):
    """Finds a tool node by its exact name."""
    # The ID is 'tool_' + tool_name
    node_id = f"tool_{tool_name}"
    if node_id in graph.nodes:
        return [(node_id, graph.nodes[node_id])]
    return []

def get_outgoing_nodes(graph, source, edge_type=None):
    source_id = source[0] if isinstance(source, tuple) else source
    connected = []
    if source_id not in graph:
        return []
    for _, tgt, data in graph.out_edges(source_id, data=True):
        if edge_type is None or data.get('edge_type') == edge_type:
            connected.append((tgt, graph.nodes[tgt]))
    return connected

def get_outgoing_tool_nodes(graph, source):
    """DEPRECATED - use get_outgoing_nodes(graph, source, 'uses_tool')"""
    # This function is kept for compatibility with older prompts,
    # but new prompts should favor get_outgoing_nodes.
    return get_outgoing_nodes(graph, source, edge_type='uses_tool')


def get_referencing_steps(graph, node_id):
    """For merged nodes (thought, action, tool), gets the step nodes that reference it."""
    node_data = graph.nodes.get(node_id)
    if not node_data:
        return []
    
    steps = []
    for step_id in node_data.get('referenced_by', []):
        if step_id in graph.nodes:
            steps.append((step_id, graph.nodes[step_id]))
    return steps

def safe_get_attribute(node_attrs, attribute, default=None):
    if not isinstance(node_attrs, dict):
        return default
    value = node_attrs.get(attribute, default)
    
    # Check for NaN (which is a float)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    
    return value

def load_graph_and_index(graph_pkl_path):
    with open(graph_pkl_path, "rb") as f:
        return pickle.load(f)

# --- Agent Functions (Refactored for Gemini) ---

# Agent 1: Planning Agent
def plan_traversal_path(query):
    system_prompt = (
        f'''You are a graph traversal planner. You create detailed, step-by-step plans to query a graph based on the schema provided.

**GRAPH SCHEMA:***
{graph_struct}

### Critical Rules for Planning:
1.  **Use `get_nodes_by_attribute` to find starting nodes.** For example, to find trajectory 2, plan to use `get_nodes_by_attribute(graph, 'traj_id', 2)`. To find a tool, use `get_nodes_by_attribute(graph, 'tool_name', 'tool_name_here')`.
2.  **Use `get_outgoing_nodes` to traverse.** For example, `get_outgoing_nodes(step_node, 'has_thought')` or `get_outgoing_nodes(traj_node, 'has_step')`.
3.  **To find steps that used a tool**, first find the tool node (e.g., `get_nodes_by_attribute(graph, 'tool_name', '...')`), then use `get_referencing_steps(tool_node)`.
4.  **To get specific tool call details (like failure reasons)**, you must plan to get the `tool` node and iterate through its `calls` list attribute.

**Your Instructions:**
Generate a structured JSON traversal plan. The plan must include:

-   `query`: The original natural language query.
-   `reasoning`: A clear explanation of *how* you will use the schema and helper functions to answer the query.
-   `traversal_plan`: A list of steps. Each step must include:
    -   `step`: Step number (1, 2, 3...).
    -   `description`: What this step does (e.g., "Find the node for trajectory 2.").
    -   `function_call`: The exact helper function to use (e.g., "get_nodes_by_attribute").
    -   `parameters`: The parameters for the function (e.g., {{'attribute': 'traj_id', 'value': 2}}).
    -   `data_to_collect`: Attributes to get from the resulting node(s) (e.g., ["full_text", "calls"]).
    -   `data_dependency`: (Optional) Which previous step's output is used.
-   `steps_important`: A list of step numbers (e.g., [3]) that generate the *final* data required to answer the user query.
'''
    )

    user_prompt = (
        f"Query: {query}\n\n"
        f"Create a detailed traversal plan in JSON format based on the provided schema."
    )
    messages = [system_prompt, user_prompt]
    
    response = generate_with_retry(
        model,
        messages,
        generation_config=json_generation_config
    )

    json_text = response.text.strip().lstrip('```json').rstrip('```')
    return json.loads(json_text)

# Agent 2: Code Generation Agent
def generate_traversal_code(query, plan):
    system_prompt = (
        f'''You are a Python expert that converts a *complete* traversal plan into a *single* executable Python script.
        
**GRAPH SCHEMA:***
{graph_struct}

The following helper functions are ALREADY DEFINED and available to you.
DO NOT redefine them. CALL them directly.

AVAILABLE FUNCTIONS (CALL THESE, DO NOT RE-IMPLEMENT):
***1. parse_node_id(node_id)
**2. get_nodes_by_type(graph, node_type)
**3. get_nodes_by_attribute(graph, attribute, value)
**4. get_nodes_by_status(graph, status)
**5. get_step_nodes(graph, traj_id, step_id)
**6. get_tool_nodes_by_name(graph, tool_name) -> DEPRECATED, use get_nodes_by_attribute('tool_name', ...)
**7. get_outgoing_nodes(graph, source, edge_type=None)
**8. get_referencing_steps(graph, node_id)
**9. safe_get_attribute(node_attrs, attribute)

CRITICAL RULES:
1.  **DO NOT** define any functions (e.g., `def my_function(): ...`). Write a direct script.
2.  **CALL** the "AVAILABLE FUNCTIONS" listed above.
3.  **NEVER** hardcode node IDs (e.g., "2_step1" or "tool_search_google"). Use the helper functions to find them.
4.  **ALWAYS** loop through results, as functions return lists (e.g., `for node, attrs in get_nodes_by_attribute(...)`).
5.  Use the 'graph' variable (which is preloaded).
6.  Store the final result(s) in a dictionary named 'final_result'.
7.  Use the 'steps_important' list from the plan to determine what to store in 'final_result'. If 'steps_important' is [3], then `final_result['step_3'] = ...`.
8.  Handle empty results gracefully (e.g., `final_result = {{'step_3': []}}`).
9.  **Only** output a single Python code block. Do not add any explanation.
'''
    )

    user_prompt = (
        f"Query: {query}\n\n"
        f"Traversal Plan:\n{json.dumps(plan, indent=2)}\n\n"
        f"Generate a single, complete Python script that implements this plan and stores its final answer(s) in a dictionary named 'final_result'."
    )
    messages = [system_prompt, user_prompt]
    
    response = generate_with_retry(
        model,
        messages,
        generation_config=code_generation_config
    )
    return response.text

# --- Code Execution ---

def clean_code_block(code_str):
    """Cleans the ```python markdown fences from code output."""
    lines = code_str.strip().splitlines()
    if lines and lines[0].startswith("```python"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)

def execute_traversal_code(code_str, graph):
    """Executes the generated traversal code in a safe environment."""
    local_env = {
        "graph": graph,
        "final_result": {}, # IMPORTANT: Initialize the result dict
        "nx": nx,
        "parse_node_id": parse_node_id,
        "get_nodes_by_type": get_nodes_by_type,
        "get_nodes_by_attribute": get_nodes_by_attribute,
        "get_nodes_by_status": get_nodes_by_status,
        "get_step_nodes": get_step_nodes,
        "get_tool_nodes_by_name": get_tool_nodes_by_name, # Kept for compatibility
        "get_outgoing_nodes": get_outgoing_nodes,
        "get_outgoing_tool_nodes": get_outgoing_tool_nodes, # Kept for compatibility
        "get_referencing_steps": get_referencing_steps,
        "safe_get_attribute": safe_get_attribute
    }
    
    try:
        exec(code_str, local_env, local_env)
        # The code is expected to populate 'final_result' directly
        return local_env["final_result"] 
    except Exception as e:
        import traceback
        print(f"--- Code Execution Error ---", file=sys.stderr)
        print(f"Code:\n{code_str}", file=sys.stderr)
        print(f"Error: {e}\n{traceback.format_exc()}", file=sys.stderr)
        return {"error": f"Execution Error: {str(e)}", "traceback": traceback.format_exc()}

# --- Simplified Main Agent Pipeline ---

def query_graph_agent(query, graph_pkl_path):
    """
    Runs the simplified 2-step (Plan, Code) agent pipeline.
    """
    print(f"\n🔍 Starting query: {query}", file=sys.stderr)
    
    try:
        graph = load_graph_and_index(graph_pkl_path)
    except FileNotFoundError:
        print(f"Error: Graph file not found at {graph_pkl_path}", file=sys.stderr)
        return {"error": "Graph file not found"}, "", {}
    except Exception as e:
        print(f"Error loading graph: {e}", file=sys.stderr)
        return {"error": f"Error loading graph: {e}"}, "", {}

    
    # Planning 
    print("Generating plan...", file=sys.stderr)
    plan = plan_traversal_path(query)
    #print(f"Generated Plan:\n{json.dumps(plan, indent=2)}\n", file=sys.stderr)
    # Code Generation 
    print("Generating code...", file=sys.stderr)
    traversal_code = generate_traversal_code(query, plan)
    cleaned_code = clean_code_block(traversal_code)
    print(f"Generated Code:\n{cleaned_code}\n", file=sys.stderr)
    # Execution 
    print("Executing code...", file=sys.stderr)
    result = execute_traversal_code(cleaned_code, graph)
    return plan, cleaned_code, result

def graph_agent(query):
    """
    Main entry point for the agent.
    """
    path_to_graph="trajectory_graph.pkl" 
    
    if not os.path.exists(path_to_graph):
        print(f"Error: Graph file not found at {path_to_graph}", file=sys.stderr)
        return json.dumps({
            "error": "Graph file not found",
            "plan": {},
            "final_code": "",
            "final_response": {}
        }, indent=2)

    # pipeline call
    plan, final_code, final_response = query_graph_agent(query, path_to_graph)
    
    #final output
    return json.dumps({
        "plan": plan,
        "final_code": final_code,
        "final_response": final_response
    }, indent=2, cls=SetEncoder)
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
        
    user_query=sys.argv[1]
    response=graph_agent(user_query)
    print(response)