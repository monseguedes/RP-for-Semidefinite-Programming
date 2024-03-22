import os
import pickle

for file_name in [file for file in os.listdir() if ".pkl" in file]:
    print(file_name)
    with open(file_name, "rb") as file:
        results = pickle.load(file)

    results["original"].pop('X_sol')
    
    for projector_type in list(results.keys())[1:]:
        for projection_size in results[projector_type].keys():
            results[projector_type][projection_size].pop("X_sol")

    with open(file_name, "wb") as f:
        pickle.dump(results, f)


    

