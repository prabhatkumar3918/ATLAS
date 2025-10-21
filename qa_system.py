import pickle
import networkx as nx
from collections import Counter

class GraphQueryEngine:
    def __init__(self, graph_path="trajectory_graph.pkl"):
        try:
            with open(graph_path, "rb") as f:
                self.graph = pickle.load(f)
            print(f"Knowledge graph loaded successfully from '{graph_path}'.")
            print(f"   - Nodes: {len(self.graph.nodes)}")
            print(f"   - Edges: {len(self.graph.edges)}\n")
            self._prepare_lookups()
        except FileNotFoundError:
            print(f"Error: The graph file was not found at '{graph_path}'.")
            print("Please make sure the file is in the same directory as this script.")
            self.graph = None
    
    def _prepare_lookups(self):
        """Creates dictionaries for fast access to tasks and trajectories."""
        self.tasks = {}
        self.trajectories = {}
        for node, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'task':
                # Use a simple integer ID for easier user input
                task_num = len(self.tasks) + 1
                self.tasks[str(task_num)] = {
                    'node_id': node,
                    'description': data.get('description', 'N/A')
                }
            elif data.get('node_type') == 'trajectory':
                traj_id = str(data.get('traj_id', 'N/A'))
                self.trajectories[traj_id] = node

    def list_tasks(self):
        """Lists all unique tasks found in the graph."""
        if not self.tasks:
            return "No tasks found in the graph."
        
        output = "--- Available Tasks ---\n"
        print(f"Found {len(self.tasks)} tasks.")
        for task_num, data in self.tasks.items():
            desc = data['description'].replace('\n', ' ').strip()
            output += f"  [{task_num}] {desc[:80]}...\n"
        return output

    def get_task_details(self, task_num):
        """Gets the status of all trajectories for a given task."""
        if task_num not in self.tasks:
            return f"Error: Task number '{task_num}' not found. Use 'list_tasks' to see available tasks."

        task_node_id = self.tasks[task_num]['node_id']
        task_desc = self.tasks[task_num]['description']
        
        output = f"--- Details for Task {task_num}: {task_desc[:80]}... ---\n"
        
        traj_nodes = [n for n in self.graph.successors(task_node_id) if self.graph.nodes[n].get('node_type') == 'trajectory']
        
        if not traj_nodes:
            return output + "No trajectories found for this task."
            
        for traj_node in traj_nodes:
            data = self.graph.nodes[traj_node]
            output += f"  - Trajectory {data.get('traj_id')}: Status = {data.get('status')}\n"
        return output

    def get_trajectory_steps(self, traj_id):
        if traj_id not in self.trajectories:
            return f"Error: Trajectory ID '{traj_id}' not found."

        traj_node_id = self.trajectories[traj_id]
        output = f"--- Steps for Trajectory {traj_id} ---\n"

        step_nodes = [n for n in self.graph.successors(traj_node_id) if self.graph.nodes[n].get('node_type') == 'step']
        
        if not step_nodes:
            return output + "No steps found for this trajectory."
            
        # sort steps by Step_ID
        step_nodes.sort(key=lambda n: int(n.split('_step')[-1]))

        for i, step_node in enumerate(step_nodes):
            step_data = self.graph.nodes[step_node]
            output += f"\n[ Step {step_data['label'].split(' ')[1]} ] - Status: {step_data.get('status', 'N/A')}\n"
            
            # Find and format subnodes
            for sub_node in self.graph.successors(step_node):
                sub_data = self.graph.nodes[sub_node]
                sub_type = sub_data.get('node_type', '')
                if sub_type in ['thought', 'action', 'observation']:
                    text = sub_data.get('full_text', '').replace('\n', '\n      ')
                    output += f"  - {sub_type.capitalize()}: {text}\n"
        return output

    def find_common_errors(self, top_n=5):
        """Finds the most frequent error messages (observations)."""
        error_counter = Counter()
        
        for node, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'observation':
                text = data.get('full_text', '').lower()
                if 'error' in text or 'failed' in text or 'traceback' in text:
                    count = len(data.get('referenced_by', []))
                    error_counter[data['full_text']] += count

        if not error_counter:
            return "No common errors found."
            
        output = f"--- Top {top_n} Most Common Errors ---\n"
        for i, (error, count) in enumerate(error_counter.most_common(top_n)):
            output += f"\n{i+1}. (Occurred {count} times)\n"
            output += f"   - Observation: {error.strip()}\n"
        return output
        
    def find_trajectories_by_tool(self, tool_name):
        """Finds all trajectories that used a specific tool."""
        trajectories_found = set()
        
        # Find the tool node
        tool_node_id = f"tool_{tool_name}"
        if not self.graph.has_node(tool_node_id):
            return f"Error: Tool '{tool_name}' not found in the graph."
            
        # Get all steps that used this tool (predecessors of the tool node)
        step_nodes = self.graph.predecessors(tool_node_id)
        
        for step_node in step_nodes:
            traj_id = self.graph.nodes[step_node].get('traj_id')
            if traj_id:
                trajectories_found.add(traj_id)

        if not trajectories_found:
            return f"No trajectories found using the tool '{tool_name}'."
        
        output = f"--- Trajectories using Tool '{tool_name}' ---\n"
        for traj_id in sorted(list(trajectories_found)):
            output += f"  - Trajectory {traj_id}\n"
        return output

def print_help():
    print("Available commands:")
    print("  list_tasks                      - Show all unique tasks.")
    print("  task_details <task_num>         - Show trajectory statuses for a task (e.g., task_details 1).")
    print("  show_traj <traj_id>             - Display the full steps for a trajectory (e.g., show_traj 1).")
    print("  common_errors [top_n]           - Find the most common errors (e.g., common_errors 3).")
    print("  find_tool <tool_name>           - Find all trajectories that used a specific tool (e.g., find_tool python).")
    print("  help                            - Show this help message.")
    print("  exit / quit                     - Exit the program.")

def main():
    engine = GraphQueryEngine()
    if not engine.graph:
        return

    print("Welcome to the Trajectory QA System. Type 'help' for a list of commands.")
    
    while True:
        try:
            user_input = input("qa> ").strip().lower()
            if not user_input:
                continue
                
            parts = user_input.split()
            command = parts[0]
            
            if command in ["exit", "quit"]:
                print("Exiting. Goodbye!")
                break
            elif command == "help":
                print_help()
            elif command == "list_tasks":
                print(engine.list_tasks())
            elif command == "task_details":
                if len(parts) > 1:
                    print(engine.get_task_details(parts[1]))
                else:
                    print("Usage: task_details <task_num>")
            elif command == "show_traj":
                if len(parts) > 1:
                    print(engine.get_trajectory_steps(parts[1]))
                else:
                    print("Usage: show_traj <traj_id>")
            elif command == "common_errors":
                top_n = int(parts[1]) if len(parts) > 1 else 5
                print(engine.find_common_errors(top_n))
            elif command == "find_tool":
                if len(parts) > 1:
                    print(engine.find_trajectories_by_tool(parts[1]))
                else:
                    print("Usage: find_tool <tool_name>")
            else:
                print(f"Unknown command: '{command}'. Type 'help' for options.")

        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
    
