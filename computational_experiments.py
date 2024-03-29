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
import datetime
import time


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
            print(f"    Running experiments for petersen {n}_{k} complement...")

            graph = generate_graphs.generate_generalised_petersen(
                n, k, complement=True, save=False
            )

            stable_set_experiments_graph(
                f"results/stable_set/petersen/{n}_{k}_complement.pkl",
                graph,
                complement=True,
            )
            print("Done!")

    # Create folder for cordones results
    os.makedirs("results/stable_set/cordones", exist_ok=True)
    for i, graph in enumerate(config["cordones"]):
        n = graph["n"]
        print(f"Scanning cordones graph {i + 1} of {len(config['cordones'])}")
        if 2 * graph["n"] <= config["stable_set"]["max_vertices"]:
            print(f"    Running experiments for cordones {n} complement")

            graph = generate_graphs.generate_cordones(n, complement=True, save=False)

            stable_set_experiments_graph(
                f"results/stable_set/cordones/{n}_complement.pkl",
                graph,
                complement=True,
            )
            print("Done!")


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

    if os.path.exists(directory):  # Load previous results if available
        with open(directory, "rb") as f:
            sol_dict = pickle.load(f)
    else:
        sol_dict = {}

    if "L1" not in sol_dict:  # Run original sdp stable set only if not stored
        L1_results = first_level_stable_set.stable_set_problem_sdp(graph)
        sol_dict = {"L1": L1_results}

    if "L2" not in sol_dict:  # Run second level sdp stable set only if not stored
        L2_results = second_level_stable_set.second_level_stable_set_problem_sdp(graph)
        sol_dict["L2"] = L2_results
        
    # Save as pickle
    with open(directory, "wb") as f:
        pickle.dump(sol_dict, f)

    if complement:
        projections = config["stable_set"]["c_projection"]
    else:
        projections = config["stable_set"]["projection"]

    projector_type = config["densities"][min(config["densities"], key=lambda x:abs(x - sol_dict["L2"]["size_psd_variable"]))][0] 
    sol_dict[projector_type] = {}
    for projection in projections:
        projector = rp.RandomProjector(
            round(projection * sol_dict["L2"]["size_psd_variable"]),
            sol_dict["L2"]["size_psd_variable"],
            projector_type,
        )
        results = (
            second_level_stable_set.projected_second_level_stable_set_problem_sdp(
                graph, projector
            )
        )
        sol_dict[projector_type][projection] = results

    # Save as pickle
    with open(directory, "wb") as f:
        pickle.dump(sol_dict, f)


def run_maxcut_experiments(config):
    """ """

    # Create folder for maxcut results
    if not os.path.exists("results/maxcut"):
        os.makedirs("results/maxcut")

    folders = [
        folder
        for folder in os.listdir("graphs/maxcut")
        if os.path.isdir(os.path.join("graphs/maxcut", folder))
        and not folder in ["out", "plot"]
    ]
    for i, name in enumerate(folders):
        print(f"Scanning graph {i + 1} of {len(folders)}        ")
        directory = f"graphs/maxcut"
        file_path = directory + f"/{name}/graph.pkl"

        with open(file_path, "rb") as file:
            graph = pickle.load(file)

        if (
            graph.n <= config["maxcut"]["max_vertices"]
            and graph.n >= config["maxcut"]["min_vertices"]
        ):
            print(
                f"    Running experiments for graph {name}, starting at {datetime.datetime.now()}"
            )
            start = time.time()
            maxcut_experiments_graph(f"results/maxcut/{name}.pkl", graph)
            print(
                f"    Finished experiments for graph {name}, took {time.time() - start} seconds"
            )


def maxcut_experiments_graph(directory, graph):
    """
    Run the experiments and store them in files.

    """

    if os.path.exists(directory):  # Load previous results if available
        with open(directory, "rb") as f:
            sol_dict = pickle.load(f)
    else:
        sol_dict = {}

    if "original" not in sol_dict:  # Run original sdp maxcut only if not stored
        start = time.time()
        results = maxcut.sdp_relaxation(graph)
        sol_dict = {"original": results}
        print(f"    Finished original sdp maxcut, took {time.time() - start} seconds")
        # Save as pickle
        with open(directory, "wb") as f:
            pickle.dump(sol_dict, f)
    else:
        results = sol_dict["original"]

    for projector_type in list(config["densities"][sol_dict["original"]["size_psd_variable"]]):  # Run projection for different projectors only if not stored
        if projector_type not in sol_dict:
            sol_dict[projector_type] = {}
        gen_projection = (
            projection
            for projection in config["maxcut"]["projection"]
            if projection not in sol_dict[projector_type]
        )
        for projection in gen_projection:
            projector = rp.RandomProjector(
                round(projection * results["size_psd_variable"]),
                results["size_psd_variable"],
                projector_type,
            )
            start = time.time()
            p_results = maxcut.projected_sdp_relaxation(graph, projector)
            print(
                f"    Finished {projection} for projector {projector.type} sdp maxcut, took {time.time() - start} seconds"
            )

            sol_dict[projector_type][projection] = p_results

    # Save as pickle
    with open(directory, "wb") as f:
        pickle.dump(sol_dict, f)


def run_max_sat_experiments(config):
    """ """

    # Create folder for max sat results
    if not os.path.exists("results/maxsat"):
        os.makedirs("results/maxsat", exist_ok=True)

    if config["maxsat"]["random"]:
        for variables in config["maxsat"]["variables"]:
            for C in config["maxsat"]["C"]:
                print(
                    f"Running maxsat instance {variables} variables and {variables * C} clauses, with C = {C}"
                )
                file_name = f"{variables}_{C}.pkl"
                c = variables * C
                max_sat_experiments_formula(file_name, variables, c)

    else:
        for i, file_name in enumerate([file for file in os.listdir("datasets/pmax2sat") if config["maxsat"]["min_variables"] <= file.split("_")[1] <= config["maxsat"]["max_variables"]]):
            with open(f"datasets/pmax2sat/{file_name}", "rb") as f:
                formula = pickle.load(f)
            print(f"Running maxsat instance {file_name}, number {i + 1} of {len(os.listdir('datasets/pmax2sat'))}")
            max_sat_experiments_formula(file_name, formula.n, formula.c, formula.list_of_clauses)

def max_sat_experiments_formula(file_name, n, c, clauses_list=[]):
    """
    Run the experiments and store them in files.

    """

    # Create maxsat intance.
    formula = maxsat.Formula(n, c, list_of_clauses=clauses_list)

    if os.path.exists(f"results/maxsat/{file_name}"):
        sol_dict = pickle.load(open(f"results/maxsat/{file_name}", "rb"))
    else:
        sol_dict = {}

    if "original" not in sol_dict:
        results = maxsat.sdp_relaxation(formula)
        sol_dict = {"original": results}
        with open(f"results/maxsat/{file_name}", "wb") as f:
            pickle.dump(sol_dict, f)
    
    for projector_type in config["densities"][results["size_psd_variable"] - 1]:
        if projector_type not in sol_dict:
            sol_dict[projector_type] = {}
        gen = (
            projection
            for projection in config["maxsat"]["projection"]
            if projection not in sol_dict[projector_type]
        )
        for projection in gen:
            projector = rp.RandomProjector(
                round(projection * results["size_psd_variable"]),
                results["size_psd_variable"],
                projector_type,
            )
            p_results = maxsat.projected_sdp_relaxation(formula, projector)
            sol_dict[projector_type][projection] = p_results
            with open(f"results/maxsat/{file_name}", "wb") as f:
                pickle.dump(sol_dict, f)

    # Store maxsat instance.
    with open(f"results/maxsat/{file_name}", "wb") as f:
        pickle.dump(sol_dict, f)

def sat_feasibility(config):
    """

    """

    # Create folder for max sat results
    if not os.path.exists("results/sat"):
        os.makedirs("results/sat", exist_ok=True)

    for variables in config["sat"]["variables"]:
        for C in config["sat"]["C"]:
            print("-" * 40)
            print(
                f"Running SAT instance {variables} variables and {variables * C} clauses, with C = {C}"
            )
            for i in range(config["sat"]["repetitions"]):
                print("-" * 20)
                print(f"Repetition {i + 1} of {config['sat']['repetitions']}")
                # Create intance.
                formula = maxsat.Formula(variables, variables * C, seed=i)

                if os.path.exists(f"results/sat/{variables}_{C}_{i}.pkl"):
                    sol_dict = pickle.load(open(f"results/sat/{variables}_{C}_{i}.pkl", "rb"))
                else:
                    sol_dict = {}

                if "original" not in sol_dict:
                    print("Running original SAT instance")
                    results = maxsat.satisfiability_feasibility(formula)
                    sol_dict = {"original": results}
                    print("Finished original SAT instance with size {}, took {} seconds".format(results["size_psd_variable"], results["computation_time"]))
                
                for projector_type in config["densities"][sol_dict["original"]["size_psd_variable"] - 1]:
                    # projector_type = "sparse"
                    if projector_type not in sol_dict:
                        sol_dict[projector_type] = {}
                    gen = (
                        projection
                        for projection in config["sat"]["projection"]
                        if projection not in sol_dict[projector_type]
                    )
                    for projection in gen:
                        projector = rp.RandomProjector(
                            round(projection * results["size_psd_variable"]),
                            results["size_psd_variable"],
                            projector_type,
                        )
                        print(f"Running SAT instance with projector {projector_type} and projection {projection}")
                        p_results = maxsat.projected_sat_feasibility(formula, projector)
                        print("Finished SAT instance with size {}, took {} seconds".format(p_results["size_psd_variable"], p_results["computation_time"]))
                        sol_dict[projector_type][projection] = p_results

                # Store sat instance.
                with open(f"results/sat/{variables}_{C}_{i}.pkl", "wb") as f:
                    pickle.dump(sol_dict, f)

            print()


def quality_plot_computational_experiments_maxcut():
    # Create folder for maxcut results
    if not os.path.exists("results/maxcut/plot"):
        os.makedirs("results/maxcut/plot")

    names = [name for name in os.listdir("graphs/maxcut/plot") if ".txt" in name]

    for i, name in enumerate(names):
        print(f"Scanning graph {i + 1} of {len(names)}        ")
        if not os.path.exists(f"results/maxcut/plot/{name}.pkl"):
            directory = f"graphs/maxcut"
            file_name = "graphs/maxcut/plot/" + name
            graph = File(file_name)

            print(
                f"    Running experiments for graph {name}, starting at {datetime.datetime.now()}"
            )

            start = time.time()
            results = maxcut.sdp_relaxation(graph)
            print(
                f"    Finished original sdp maxcut, took {time.time() - start} seconds"
            )
            sol_dict = {"original": results}
            size = sol_dict["original"]["size_psd_variable"]

            if 0 <= size <= 1000:
                projector_type = "sparse"
            elif 2000 <= size <= 3500:
                projector_type = "0.2_density"
            elif size == 4000:
                projector_type = "0.1_density"
            elif 5000 <= size <= 6000:
                projector_type = "0.05_density"
            elif 7000 <= size <= 8000:
                projector_type = "0.04_density"

            sol_dict[projector_type] = {}
            projection = 0.1
            projector = rp.RandomProjector(
                round(projection * results["size_psd_variable"]),
                results["size_psd_variable"],
                projector_type,
            )
            start = time.time()
            p_results = maxcut.projected_sdp_relaxation(graph, projector)
            print(
                f"    Finished {projection} for projector {projector.type} sdp maxcut, took {time.time() - start} seconds"
            )

            sol_dict[projector_type][projection] = p_results

            # Save as pickle
            with open(f"results/maxcut/plot/{name}.pkl", "wb") as f:
                pickle.dump(sol_dict, f)

            print(
                f"    Finished experiments for graph {name}, took {time.time() - start} seconds"
            )


if __name__ == "__main__":
    with open("config.yml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # run_stable_set_experiments(config)
    # run_maxcut_experiments(config)
    # run_max_sat_experiments(config)
    # quality_plot_computational_experiments_maxcut()
    sat_feasibility(config)
