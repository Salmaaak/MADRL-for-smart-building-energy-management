import os

IDF_FILE = "5ZoneAirCooled.idf"

def force_end_date():
    if not os.path.exists(IDF_FILE):
        print("IDF not found.")
        return

    with open(IDF_FILE, 'r') as f:
        lines = f.readlines()

    new_lines = []
    inside_run_period = False
    
    for line in lines:
        if "RunPeriod," in line:
            inside_run_period = True
            new_lines.append(line)
            continue
        
        if inside_run_period:
            # Change End Month to 1
            if "End Month" in line:
                new_lines.append("    1,                       !- End Month\n")
                print(">>> Set End Month to 1")
                continue
            # Change End Day to 31
            if "End Day of Month" in line:
                new_lines.append("    31,                      !- End Day of Month\n")
                print(">>> Set End Day to 31")
                continue
            
            if ";" in line:
                inside_run_period = False
        
        new_lines.append(line)

    with open(IDF_FILE, 'w') as f:
        f.writelines(new_lines)
    print("IDF Date Fixed.")

if __name__ == "__main__":
    force_end_date()