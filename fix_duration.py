import os

IDF_FILE = "5ZoneAirCooled.idf"

def extend_simulation_time():
    if not os.path.exists(IDF_FILE):
        print(f"Error: {IDF_FILE} not found.")
        return

    with open(IDF_FILE, 'r') as f:
        lines = f.readlines()

    new_lines = []
    inside_run_period = False
    
    print(f"Scanning {IDF_FILE}...")
    
    for line in lines:
        # Detect the RunPeriod block
        if "RunPeriod," in line:
            inside_run_period = True
            new_lines.append(line)
            continue
        
        # Inside the block, look for the End Day line
        if inside_run_period:
            # We are looking for the line that defines the End Day (usually "1, !- End Day of Month")
            if "End Day of Month" in line:
                # Force it to 31
                parts = line.split(',')
                # Reconstruct the line preserving comments
                new_line = f"    31,                      !- End Day of Month\n"
                new_lines.append(new_line)
                print(">>> SUCCESS: Updated 'End Day of Month' to 31.")
                
                # We assume the block ends after this or nearby, but we just flag off safely
                # (EnergyPlus objects end with a semicolon, but usually on the last field)
                continue
            
            # If we hit a semicolon, we are done with this object
            if ";" in line:
                inside_run_period = False

        new_lines.append(line)

    # Write back
    with open(IDF_FILE, 'w') as f:
        f.writelines(new_lines)
    
    print("IDF file updated. Ready for training.")

if __name__ == "__main__":
    extend_simulation_time()