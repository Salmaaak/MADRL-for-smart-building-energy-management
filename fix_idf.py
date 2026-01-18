import os

IDF_FILE = "5ZoneAirCooled.idf"

def fix_idf():
    if not os.path.exists(IDF_FILE):
        print(f"Error: {IDF_FILE} not found.")
        return

    with open(IDF_FILE, 'r') as f:
        lines = f.readlines()

    new_lines = []
    run_period_found = False
    
    # 1. Scan file to fix RunPeriod (Extend to 31 days)
    # Looking for: "RunPeriod," then modifying the end day
    skip_next = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Detect RunPeriod block start
        if "RunPeriod," in stripped:
            run_period_found = True
            new_lines.append(line)
            continue
        
        if run_period_found:
            # We are inside the RunPeriod block.
            # Typical format: 
            #   1, !- Begin Month
            #   1, !- Begin Day...
            #   ,  !- Begin Year
            #   1, !- End Month
            #   1, !- End Day ... <--- WE NEED TO CHANGE THIS TO 31
            
            # Simple heuristic: Look for lines with "End Day of Month"
            if "End Day of Month" in line:
                # Replace 1 with 31, keeping formatting roughly same
                parts = line.split(',')
                if len(parts) > 0:
                    # Construct new line: "    31,   !- End Day of Month\n"
                    new_lines.append("    31,                      !- End Day of Month\n")
                    print(">>> Fixed: RunPeriod extended to 31 days.")
                    continue
        
        new_lines.append(line)

    # 2. Append Missing Output Variables
    # These are required by hvac_env.py
    required_vars = [
        "Output:Variable,*,Site Direct Solar Radiation Rate per Area,hourly;",
        "Output:Variable,*,Site Diffuse Solar Radiation Rate per Area,hourly;",
        "Output:Variable,*,Zone Air System Sensible Heating Energy,hourly;"
    ]
    
    print(">>> Appending required Output:Variables...")
    new_lines.append("\n! --- Added by fix_idf.py for RL Agent ---\n")
    for var in required_vars:
        new_lines.append(f"{var}\n")

    # 3. Write back to file
    with open(IDF_FILE, 'w') as f:
        f.writelines(new_lines)
    
    print(f"Success! {IDF_FILE} has been patched.")

if __name__ == "__main__":
    fix_idf()