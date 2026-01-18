import os

IDF_FILE = "5ZoneAirCooled.idf"

def setup_simulation():
    if not os.path.exists(IDF_FILE):
        print(f"Error: {IDF_FILE} not found.")
        return

    print(f"Reading {IDF_FILE}...")
    with open(IDF_FILE, 'r') as f:
        lines = f.readlines()

    new_lines = []
    run_period_found = False
    variables_already_exist = False
    
    # 1. FIX DURATION: Scan file to extend RunPeriod to 31 days
    for line in lines:
        stripped = line.strip()
        
        # Check if variables already exist to avoid double appending
        if "Site Direct Solar Radiation Rate per Area" in line:
            variables_already_exist = True

        # Detect RunPeriod block start
        if "RunPeriod," in stripped:
            run_period_found = True
            new_lines.append(line)
            continue
        
        if run_period_found:
            # Look for the End Day line inside RunPeriod
            if "End Day of Month" in line:
                # Force it to 31
                new_lines.append("    31,                      !- End Day of Month\n")
                print(">>> FIXED: RunPeriod extended to 31 days.")
                continue
            
            # End of RunPeriod object
            if ";" in line:
                run_period_found = False
        
        new_lines.append(line)

    # 2. FIX VARIABLES: Append Missing Output Variables if not present
    if not variables_already_exist:
        print(">>> FIXED: Appending missing Output:Variables...")
        new_lines.append("\n! --- Added by setup_simulation.py for RL Agent ---\n")
        new_lines.append("Output:Variable,*,Site Direct Solar Radiation Rate per Area,hourly;\n")
        new_lines.append("Output:Variable,*,Site Diffuse Solar Radiation Rate per Area,hourly;\n")
        new_lines.append("Output:Variable,*,Zone Air System Sensible Heating Energy,hourly;\n")
    else:
        print(">>> SKIP: Output:Variables already present.")

    # 3. Write back to file
    with open(IDF_FILE, 'w') as f:
        f.writelines(new_lines)
    
    print(f"SUCCESS: {IDF_FILE} is fully patched and ready for training.")

if __name__ == "__main__":
    setup_simulation()