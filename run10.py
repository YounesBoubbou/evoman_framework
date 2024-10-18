# Runs the code of the specialist optimisation ten times (if neeeded the results do not already exist).

import os, sys

# runs the EA ten times if needed
def run10(EA,experiment_name, enemy):
    for i in range(10):
        # If results do not already exist:
        if not os.path.exists(experiment_name+str(i)):
            # Note: in order to run the EA ten times, we need it to accept an experiment name as the first argument, the enemy id as the second argument
            commandname = "python " + EA + ' ' + experiment_name + str(i) + ' ' + enemy 
            os.system(commandname)
        else:
            print("directory '" + experiment_name + str(i) + "' already exists, will be skipped")

try:
    commandline = sys.argv[3]
except:
    print("usage: 'python run10.py EA_executable.py save_file_name enemy_id'")
    exit()
executable, experiment_name, enemy_number = sys.argv[1], sys.argv[2], sys.argv[3]

# Automatically adding the enemy id to the name for easier organisation
run10(executable,experiment_name + enemy_number, enemy_number)
