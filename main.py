import os
from parser import TrajectoryParser

def main():
    """
    Main function to process trajectory files and generate CSV reports.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, 'swe_parsed_data')
    output_dir = os.path.join(current_dir, 'output')
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    # list all JSON files indirectory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No .json files found in '{input_dir}'.")
        return

    parser = TrajectoryParser(agent_name="swe-agent")

    # Process each file
    print(f"Found {len(json_files)} trajectory files to process...")
    for file_name in sorted(json_files): 
        file_path = os.path.join(input_dir, file_name)
        print(f"Processing {file_name}...")
        parser.parse_file(file_path)

    parser.save_tables_to_csv(output_dir)

if __name__ == "__main__":
    main()
