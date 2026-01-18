import os
import glob

# Defines the path to your results
LOG_DIR = "results"

def check_errors():
    # Find the latest episode folder
    list_of_dirs = glob.glob(f"{LOG_DIR}/ep_out_*")
    if not list_of_dirs:
        print("No result folders found.")
        return

    latest_dir = max(list_of_dirs, key=os.path.getctime)
    err_file = os.path.join(latest_dir, "eplusout.err")

    if not os.path.exists(err_file):
        print(f"No error file found in {latest_dir}")
        return

    print(f"--- READING ERROR LOG: {err_file} ---")
    with open(err_file, 'r') as f:
        content = f.read()
        print(content)
    print("---------------------------------------")

if __name__ == "__main__":
    check_errors()