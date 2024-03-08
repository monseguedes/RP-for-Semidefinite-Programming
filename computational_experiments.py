"""
File to run computational experiments, potentially from the server.

1. Download the datasets from websites if not available here. There 
datsets are:
    - Gi.txt files from https://web.stanford.edu/~yyye/yyye/Gset/
    - Biq Mac files from https://biqmac.aau.at/biqmaclib.html

2. Process all files into the right format (if not already there), 
and also generate graphs for stable set problem. This includes:
    - Generalised petersen for some values of n and k. NOTE: pickle 
    might not handle these graphs well.
    - Cordones graphs for some values of n.
    - Complement of these graphs. 

3. Run the experiments and store them in a file. We want to make it
so that results are saves after each run, and not at the end.
    Structure is:
    - File structure:
      - Dictionary-like object
        - Keys are the graph names
        - Values are sol_dictionaries
            - Keys are L1, L2, projections
            - Values are sol_dictionaries
                - Keys are the parameters: size, solution, cpu time
                - Values are the results

"""

from urllib.request import urlretrieve
import yaml
import generate_graphs
import os
import process_graphs
import pickle
import first_level_stable_set
import second_level_stable_set
from generate_graphs import Graph
import random_projections as rp
import maxcut
from process_graphs import File
import json
import maxsat


def download_datasets(config):
    """
    Download the datasets from websites if indicated in config. There
    datsets are:
        - Gi.txt files from https://web.stanford.edu/~yyye/yyye/Gset/
        - Biq Mac files from https://biqmac.aau.at/biqmaclib.html

    """

    if config["download"]:
        for website in config["websites"]:
            print(f"Downloading from {website}")
            url = config["websites"][website]
            filename = "graphs/maxcut/" + website + ".zip"
            urlretrieve(url, filename)

            raise NotImplementedError("Unzipping not implemented yet")


def process_datasets_stable_set(config):
    """
    Process all files into the right format (if not already there),
    and also generate graphs for stable set problem. This includes:
        - Generalised petersen for some values of n and k. NOTE: pickle
        might not handle these graphs well.
        - Cordones graphs for some values of n.
        - Complement of these graphs.

    """

    for graph in config["petersen_n_k"]:
        if not os.path.isfile(
            f"graphs/generalised_petersen_{graph['n']}_{graph['k']}/graph.pkl"
        ):
            generate_graphs.generate_generalised_petersen(graph["n"], graph["k"])
        if not os.path.isfile(
            f"graphs/generalised_petersen_{graph['n']}_{graph['k']}_complement/graph.pkl"
        ):
            generate_graphs.generate_generalised_petersen(
                graph["n"], graph["k"], complement=True
            )

    for graph in config["cordones_n"]:
        if not os.path.isfile(f"graphs/cordones_{graph['n']}/graph.pkl"):
            generate_graphs.generate_cordones(graph["n"])
        if not os.path.isfile(f"graphs/cordones_{graph['n']}_complement/graph.pkl"):
            generate_graphs.generate_cordones(graph["n"], complement=True)


def process_datasets_maxcut(config):
    """
    Process all files into the right format (if not already there),
    and also generate graphs for maxcut problem. This includes:
        - Converting maxcut graphs into stored class instances.

    """

    for name in [file for file in os.listdir("graphs/maxcut") if file.endswith(".txt")]:
        if not os.path.isfile(f"graphs/maxcut/{name.split('.')[0]}/graph.pkl"):
            file = process_graphs.File(name)
            if file.n <= config["maxcut"]["max_vertices"]:
                file.store_graph(name)


def run_stable_set_experiments(config):
    """
    Run the experiments and store them in files. We want to make it
    so that results are saves after each run, and not at the end.
    Structure is:
    - File structure:
      - Dictionary-like object
        - Keys are L1, L2, projections
        - Values are sol_dictionaries
            - Keys are the parameters: size, solution, cpu time
            - Values are the results

    """

    # Create folder for stable set results
    if not os.path.exists("results/stable_set"):
        os.makedirs("results/stable_set")

    # Create folder for petersen results
    os.makedirs("results/stable_set/petersen", exist_ok=True)

    for i, graph in enumerate(config["petersen_n_k"]):
        n = graph["n"]
        k = graph["k"]
        print(f"Scanning petersen graph {i + 1} of {len(config['petersen_n_k'])}")
        if 2 * graph["n"] <= config["stable_set"]["max_vertices"]:
            print(f"    Running experiments for graph {n}_{k}")
            if not os.path.isfile(f"graphs/generalised_petersen_{n}_{k}/graph.pkl"):
                generate_graphs.generate_generalised_petersen(n, k)

            directory = f"graphs/generalised_petersen_{n}_{k}"
            file_path = directory + "/graph.pkl"

            with open(file_path, "rb") as file:
                graph = pickle.load(file)

            stable_set_experiments_graph(
                f"results/stable_set/petersen/{n}_{k}.pkl", graph
            )

            print(f"    Running experiments for graph {n}_{k} complement")
            if not os.path.isfile(
                f"graphs/generalised_petersen_{n}_{k}_complement/graph.pkl"
            ):
                generate_graphs.generate_generalised_petersen(n, k, complement=True)

            directory = f"graphs/generalised_petersen_{n}_{k}_complement"
            file_path = directory + "/graph.pkl"

            with open(file_path, "rb") as file:
                graph = pickle.load(file)

            stable_set_experiments_graph(
                f"results/stable_set/petersen/{n}_{k}_complement.pkl", graph, 
                complement=True
            )

    # Create folder for cordones results
    os.makedirs("results/stable_set/cordones", exist_ok=True)

    for i, graph in enumerate(config["cordones"]):
        n = graph["n"]
        print(f"Scanning cordones graph {i + 1} of {len(config['cordones'])}")
        if 2 * graph["n"] <= config["stable_set"]["max_vertices"]:
            print(f"    Running experiments for graph {n}")
            if not os.path.isfile(f"graphs/cordones_{n}/graph.pkl"):
                generate_graphs.generate_cordones(n)

            directory = f"graphs/cordones_{n}"
            file_path = directory + "/graph.pkl"

            with open(file_path, "rb") as file:
                graph = pickle.load(file)

            stable_set_experiments_graph(f"results/stable_set/cordones/{n}.pkl", graph)

            print(f"    Running experiments for graph {n} complement")
            if not os.path.isfile(f"graphs/cordones_{n}_complement/graph.pkl"):
                generate_graphs.generate_cordones(n, complement=True)

            directory = f"graphs/cordones_{n}_complement"
            file_path = directory + "/graph.pkl"

            with open(file_path, "rb") as file:
                graph = pickle.load(file)

            stable_set_experiments_graph(
                f"results/stable_set/cordones/{n}_complement.pkl", graph,
                complement=True
            )


def stable_set_experiments_graph(directory, graph, complement=False):
    """
    Run the experiments and store them in files. We want to make it
    so that results are saves after each run, and not at the end.
    Structure is:
    - File structure:
      - Dictionary-like object
        - Keys are L1, L2, projections
        - Values are sol_dictionaries
            - Keys are the parameters: size, solution, cpu time
            - Values are the results

    """

    L1_results = first_level_stable_set.stable_set_problem_sdp(graph)
    sol_dict = {"L1": L1_results}

    L2_results = second_level_stable_set.second_level_stable_set_problem_sdp(graph)
    sol_dict["L2"] = L2_results

    for projector_type in config["stable_set"]["projector"]:
        projections = config["stable_set"]["projection"]
        if complement:
            projections = config["stable_set"]["c_projection"]
        for projection in projections:
            projector = rp.RandomProjector(
                round(projection * L2_results["size_psd_variable"]),
                L2_results["size_psd_variable"],
                projector_type,
            )
            results = (
                second_level_stable_set.projected_second_level_stable_set_problem_sdp(
                    graph, projector
                )
            )

            sol_dict[projection] = results

    # Save as pickle
    with open(directory, "wb") as f:
        pickle.dump(sol_dict, f)


def run_maxcut_experiments(config):
    """ """

    # Create folder for stable set results
    if not os.path.exists("results/maxcut"):
        os.makedirs("results/maxcut")

    # Create folder for maxcut results
    os.makedirs("results/maxcut", exist_ok=True)

    folders = [
        folder
        for folder in os.listdir("graphs/maxcut")
        if os.path.isdir(os.path.join("graphs/maxcut", folder))
        and not folder in ["rudy", "ising"]
    ]
    for i, name in enumerate(folders):
        print(f"Scanning graph {i + 1} of {len(folders)}")
        directory = f"graphs/maxcut"
        file_path = directory + f"/{name}/graph.pkl"

        with open(file_path, "rb") as file:
            graph = pickle.load(file)

        if graph.n <= config["maxcut"]["max_vertices"]:
            print(f"    Running experiments for graph {name}")
            maxcut_experiments_graph(f"results/maxcut/{name}.pkl", graph)


def maxcut_experiments_graph(directory, graph):
    """
    Run the experiments and store them in files.

    """

    results = maxcut.sdp_relaxation(graph)

    sol_dict = {"original": results}

    for projector_type in config["maxcut"]["projector"]:
        sol_dict[projector_type] = {}
        for projection in config["maxcut"]["projection"]:
            projector = rp.RandomProjector(
                round(projection * results["size_psd_variable"]),
                results["size_psd_variable"],
                projector_type,
            )
            p_results = maxcut.projected_sdp_relaxation(graph, projector)
           
            sol_dict[projector_type][projection] = p_results

    # Save as pickle
    with open(directory, "wb") as f:
        pickle.dump(sol_dict, f)


def run_max_sat_experiments(config):
    """ 
    
    """

    # Create folder for max sat results
    if not os.path.exists("results/maxsat"):
        os.makedirs("results/maxsat", exist_ok=True)

    for variables in config["maxsat"]["variables"]:
        for C in config["maxsat"]["C"]:
            print(f"Running maxsat instance {variables} variables and {variables * C} clauses, with C = {C}")
            # Create maxsat intance.
            formula = maxsat.Formula(variables, variables * C)
            # Solve maxsat instance.
            results = maxsat.sdp_relaxation(formula)
            sol_dict = {"original": results}
            for projector_type in config["maxsat"]["projector"]:
                sol_dict[projector_type] = {}
                for projection in config["maxsat"]["projection"]:
                    projector = rp.RandomProjector(
                        round(projection * results["size_psd_variable"]),
                        results["size_psd_variable"],
                        projector_type,
                    )
                    p_results = maxsat.projected_sdp_relaxation(formula, projector)
                
                    sol_dict[projector_type][projection] = p_results

            # Store maxsat instance.
            with open(f"results/maxsat/{variables}_{C}.pkl", "wb") as f:
                pickle.dump(sol_dict, f)
        


if __name__ == "__main__":
    with open("config.yml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # run_stable_set_experiments(config)
    # run_maxcut_experiments(config)
    run_max_sat_experiments(config)
