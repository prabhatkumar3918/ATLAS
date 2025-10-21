import json
import os
import pandas as pd

class TrajectoryParser:
    """
    Parses a directory of trajectory JSON files into structured data tables.
    """

    def __init__(self, agent_name="swe-agent"):
        """
        Initializes the parser with data containers and a trajectory ID counter.
        """
        self.agent_name = agent_name
        self.traj_id_counter = 0
        self.trajectory_data = []
        self.steps_data = []
        self.tool_calls_data = []
        self.agent_metadata = {
            "Agent_name": self.agent_name,
            "Agent Description": "An autonomous agent for software engineering tasks.",
            "Agent Tools": set() 
        }
        self.tool_metadata = {} 

    def parse_file(self, file_path):
        """
        Parses a single trajectory JSON file and updates the data tables.
        
        Args:
            file_path (str): The path to the JSON trajectory file.
        """
        self.traj_id_counter += 1
        traj_id = self.traj_id_counter

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading {file_path}: {e}")
            return

        # Trajectory Table
        self.trajectory_data.append({
            'Traj_ID': traj_id,
            'Task': data.get('task', 'N/A'),
            'Output': data.get('final_answer', 'N/A'),
            'Correctness': data.get('evaluation', 'N/A')
        })

        # Steps and ToolCalls Tables
        trajectory_steps = data.get('trajectory_steps', [])
        for i, step in enumerate(trajectory_steps):
            step_id = i + 1
            action = step.get('action', '')
            observation = step.get('observation', '')

            tool_name, tool_input = self._parse_action(action)
            status = self._infer_status(observation)
            self._update_tool_metadata(tool_name, tool_input)

        
            self.steps_data.append({
                'Traj_ID': traj_id,
                'Step_ID': step_id,
                'Agent_name': self.agent_name,
                'Thought': step.get('thought', ''),
                'Action': action,
                'Observation': observation
            })

            
            self.tool_calls_data.append({
                'Traj_ID': traj_id,
                'Step_ID': step_id,
                'Tool Name': tool_name,
                'Tool Input': tool_input,
                'Tool Output': observation,
                'Status': status
            })

    def _parse_action(self, action_string):
        """
        Extracts the tool name and input from an action string.
        """
        if not action_string:
            return 'N/A', ''
        parts = action_string.strip().split(maxsplit=1)
        tool_name = parts[0]
        tool_input = parts[1] if len(parts) > 1 else ''
        return tool_name, tool_input

    def _infer_status(self, observation):
        """
        Infers the status of a tool call based on its output (observation).
        """
        if observation is None:
            return 'Success'
        
        # heuristic to detect errors
        error_keywords = ['error', 'failed', 'traceback', 'not found']
        observation_lower = observation.lower()
        if any(keyword in observation_lower for keyword in error_keywords):
            return 'Failure'
        return 'Success'
        
    def _update_tool_metadata(self, tool_name, tool_input):
        """
        Updates the set of known tools and their descriptions.
        """
        if tool_name not in self.tool_metadata:
            self.tool_metadata[tool_name] = {
                "Tool Name": tool_name,
                "Tool Description": f"Executes the '{tool_name}' command.",
                "Tool Arguments": "Varies based on usage." # A more sophisticated parser could infer this
            }
        self.agent_metadata["Agent Tools"].add(tool_name)

    def save_tables_to_csv(self, output_dir):
        """
        Saves all the processed data into CSV files in the specified directory.

        Args:
            output_dir (str): The directory to save the CSV files in.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Convert agent tools set to a comma-separated string
        self.agent_metadata["Agent Tools"] = ', '.join(sorted(list(self.agent_metadata["Agent Tools"])))
        
        # Create DataFrames
        df_trajectory = pd.DataFrame(self.trajectory_data)
        df_steps = pd.DataFrame(self.steps_data)
        df_tool_calls = pd.DataFrame(self.tool_calls_data)
        df_agents = pd.DataFrame([self.agent_metadata])
        df_tools = pd.DataFrame(list(self.tool_metadata.values()))
        
        # Save to CSV
        df_trajectory.to_csv(os.path.join(output_dir, 'trajectory.csv'), index=False)
        df_steps.to_csv(os.path.join(output_dir, 'steps.csv'), index=False)
        df_tool_calls.to_csv(os.path.join(output_dir, 'tool_calls.csv'), index=False)
        df_agents.to_csv(os.path.join(output_dir, 'agents.csv'), index=False)
        df_tools.to_csv(os.path.join(output_dir, 'tools.csv'), index=False)

        print(f"Successfully processed {self.traj_id_counter} trajectory files.")
        print(f"CSV files saved in '{output_dir}' directory.")
