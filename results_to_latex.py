"""
Convert computational results to LaTeX tables.

"""

import os
import pickle
import yaml


def maxcut_to_latex(directory, config, projector_type="sparse", percentage=[0.1, 0.2]):
    """
    Convert the results of the maxcut problem to a LaTeX table.
    """

    # Create the table header
    table_header = r"""
    \begin{table}[!htbp]
        \captionof{table}{Computational results \textsc{maxcut}} 
        \resizebox{\textwidth}{!}{%
        \begin{tabular}{lrrrrrrrrrrrrrrr} 
            \toprule
            & & & \multicolumn{2}{c}{original} && \multicolumn{4}{c}{10\% projection} && \multicolumn{4}{c}{20\% projection} \\
            \cmidrule{4-5} \cmidrule{7-10} \cmidrule{12-15}
            \rule{0pt}{10pt} % Adding space of 10pt between lines and text below
            Instance & n & m & Value & Time && Size & Value & Time & Qlt && Size & Value & Time & Qlt \\
            \midrule
    """

    print(table_header)

    alphabetical_dir = sorted(
        [
            file
            for file in os.listdir(directory)
            if file.endswith(".pkl")
            if config["maxcut"]["name"] in file
        ],
        key=lambda x: int("".join([i for i in x if i.isdigit()])),
    )

    for name in alphabetical_dir:
        file_path = os.path.join(directory, name)
        with open(file_path, "rb") as file:
            results = pickle.load(file)
        if (
            results["original"]["size_psd_variable"]
            >= config["maxcut"]["results"]["min_vertices"]
            and results["original"]["size_psd_variable"]
            <= config["maxcut"]["results"]["max_vertices"]
        ):
            first_ratio = (
                results[projector_type][percentage[0]]["objective"]
                / results["original"]["objective"]
                * 100
            )
            second_ratio = (
                results[projector_type][percentage[1]]["objective"]
                / results["original"]["objective"]
                * 100
            )

            if first_ratio > -10000000000 and second_ratio > -1000000000:
                # results[projector_type][percentage[0]]["objective"] = 0
                # results[projector_type][percentage[0]]["computation_time"] = 0
                # first_ratio = 0

                # if second_ratio < 0:
                # results[projector_type][percentage[1]]["objective"] = 0
                # results[projector_type][percentage[1]]["computation_time"] = 0
                # second_ratio = 0

                name = name.strip(".pkl")
                print(
                    "             {:8} & {:8} & {:8} & {:8.2f} & {:8.2f} \
                    & & {:8} & {:8.2f} & {:8.2f} & {:8.2f} \
                    & & {:8} & {:8.2f} & {:8.2f} & {:8.2f} \\\\".format(
                        name.replace("_", "-"),
                        results["original"]["size_psd_variable"],
                        results["original"]["edges"],
                        results["original"]["objective"],
                        results["original"]["computation_time"],
                        results[projector_type][percentage[0]]["size_psd_variable"],
                        results[projector_type][percentage[0]]["objective"],
                        results[projector_type][percentage[0]]["computation_time"],
                        first_ratio,
                        results[projector_type][percentage[1]]["size_psd_variable"],
                        results[projector_type][percentage[1]]["objective"],
                        results[projector_type][percentage[1]]["computation_time"],
                        second_ratio,
                    )
                )
    table_footer = r"""
            \bottomrule
        \end{tabular}}
        \label{tab:my_label}
    \end{table}
    """
    print(table_footer)


def maxcut_to_latex_single(directory, config, projector_type="sparse", percentage=0.1):
    """
    Convert the results of the maxcut problem to a LaTeX table.
    """

    # Create the table header
    table_header = r"""
    \begin{table}[!htbp]
        \captionof{table}{Computational results \textsc{maxcut}} 
        \resizebox{\textwidth}{!}{%
        \begin{tabular}{lrrrrrrrrrrrrrrr} 
            \toprule
            & & & \multicolumn{2}{c}{original} && \multicolumn{4}{c}{10\% projection} \\
            \cmidrule{4-5} \cmidrule{7-10}
            \rule{0pt}{10pt} % Adding space of 10pt between lines and text below
            Instance & n & m & Value & Time && Size & Value & Time & Qlt \\
            \midrule
    """

    print(table_header)

    alphabetical_dir = sorted(
        [
            file
            for file in os.listdir(directory)
            if file.endswith(".pkl")
            if config["maxcut"]["name"] in file
        ],
        key=lambda x: int("".join([i for i in x if i.isdigit()])),
    )

    for name in alphabetical_dir:
        file_path = os.path.join(directory, name)
        with open(file_path, "rb") as file:
            results = pickle.load(file)
        if (
            results["original"]["size_psd_variable"]
            >= config["maxcut"]["results"]["min_vertices"]
            and results["original"]["size_psd_variable"]
            <= config["maxcut"]["results"]["max_vertices"]
        ):
            first_ratio = (
                results[projector_type][percentage]["objective"]
                / results["original"]["objective"]
                * 100
            )
            name = name.strip(".pkl")
            print(
                "             {:8} & {:8} & {:8} & {:8.2f} & {:8.2f} \
                & & {:8} & {:8.2f} & {:8.2f} & {:8.2f} \\\\".format(
                    name.replace("_", "-"),
                    results["original"]["size_psd_variable"],
                    results["original"]["edges"],
                    results["original"]["objective"],
                    results["original"]["computation_time"],
                    results[projector_type][percentage]["size_psd_variable"],
                    results[projector_type][percentage]["objective"],
                    results[projector_type][percentage]["computation_time"],
                    first_ratio,
                )
            )
    table_footer = r"""
            \bottomrule
        \end{tabular}}
        \label{tab:my_label}
    \end{table}
    """
    print(table_footer)


def maxcut_to_latex_single_simplified(
    directory, config, projector_type="sparse", percentage=0.1
):
    """
    Convert the results of the maxcut problem to a LaTeX table.
    """

    # Create the table header
    table_header = r"""
    \begin{table}[!htbp]
        \captionof{table}{Computational time in seconds and quality of projection \textsc{maxcut} with respect to SDP relaxation for FILL vertices rudy graphs using FILL projector} 
        \begin{tabular}{lrrrrrrrrrrrrrrr} 
            \toprule
            & & && \multicolumn{2}{c}{10\% projection} \\
            \cmidrule{5-6}
            \rule{0pt}{10pt} % Adding space of 10pt between lines and text below
            Instance & m & Density && Time & Quality \\
            \midrule
    """

    print(table_header)

    alphabetical_dir = sorted(
        [
            file
            for file in os.listdir(directory)
            if file.endswith(".pkl")
            if config["maxcut"]["name"] in file
        ],
        key=lambda x: int("".join([i for i in x if i.isdigit()])),
    )

    for name in alphabetical_dir:
        file_path = os.path.join(directory, name)
        with open(file_path, "rb") as file:
            results = pickle.load(file)
        if (
            results["original"]["size_psd_variable"]
            == config["maxcut"]["results"]["min_vertices"]
        ):
            objective_ratio = (
                results[projector_type][percentage]["objective"]
                / results["original"]["objective"]
                * 100
            )
            time_ratio = (
                results[projector_type][percentage]["computation_time"]
                / results["original"]["computation_time"]
                * 100
            )
            name = name.strip(".pkl").replace("_", "-")
            print(
                "             {:8} & {:10} & {:2} \
                & & {:2.2f} & {:2.2f} \\\\".format(
                    name,  # Instance
                    results["original"]["edges"],  # m
                    name.split("-")[2],  # Density
                    time_ratio,  # Projection time
                    objective_ratio,  # Quality
                )
            )
    table_footer = r"""
            \bottomrule
        \end{tabular}
        \label{tab:FILL}
    \end{table}
    """
    print(table_footer)


def stable_set_to_latex(directory):
    """
    Convert the results of the stable set problem to a LaTeX table.
    """

    # Create the table header
    table_header = r"""
    \begin{table}[!htbp]
        \centering
        \captionof{table}{Relative time (\%) and relative quality (\%) with respect to the second level of the Stable Set Problem for dense imperfect graphs with $n$ vertices, $m$ edges and $c$ constraints}
       \begin{tabular}{lrrrrrrrrrrr} 
        \toprule
        & & & & \multicolumn{1}{c}{original 2\textsuperscript{nd} level} && \multicolumn{3}{c}{projected 2\textsuperscript{nd} level} \\
        \cmidrule{5-5} \cmidrule{7-9}
        \rule{0pt}{10pt} % Adding space of 10pt between lines and text below
        Graph & n & m & c  & Size && Size & Qlt & Time \\
        \midrule
    """
    print(table_header)

    graphs = ["petersen", "cordones", "helm", "jahangir"]

    for graph in graphs:
        dir = os.path.join(directory, graph)

        alphabetical_dir = sorted(
            [file for file in os.listdir(dir) if file.endswith(".pkl")],
            key=lambda x: int("".join([i for i in x if i.isdigit()])),
        )
        for name in alphabetical_dir:
            file_path = os.path.join(dir, name)
            with open(file_path, "rb") as file:
                results = pickle.load(file)

            projector_type = config["densities"][
                min(
                    config["densities"],
                    key=lambda x: abs(x - results["L2"]["size_psd_variable"]),
                )
            ][0]

            key = 0.1

            quality = (
                results["L2"]["objective"]
                / results[projector_type][key]["objective"]
                * 100
            )
            time_relative = (
                results[projector_type][key]["computation_time"]
                / results["L2"]["computation_time"]
                * 100
            )

            print(
                "             {:8} & {:8} & {:8} & {:8} & {:8} && {:8} & {:8} & {:8.2f} \\\\".format(
                    "c-"
                    + graph
                    + "-"
                    + name.strip(".pkl").replace("_", "-").strip("-complement"),
                    results["L1"]["size_psd_variable"] - 1,
                    results["L2"]["edges"],
                    results["L2"]["no_constraints"],
                    results["L2"]["size_psd_variable"],
                    results[projector_type][key]["size_psd_variable"],
                    int(quality),
                    time_relative,
                )
            )

    table_footer = r"""
            \bottomrule
        \end{tabular}
        \label{tab:my_label}
    \end{table}
    """
    print(table_footer)


def maxsat_to_latex(directory, projector_type="sparse", percentage=[0.1, 0.2]):
    """
    Convert the results of the maxsat problem to a LaTeX table.
    """

    # Create the table header
    table_header = r"""
    \begin{table}[!htbp]
        \centering
        \captionof{table}{Computational results \textsc{max-2-sat}}
        \resizebox{\textwidth}{!}{%
        \begin{tabular}{lrrrrrrrrrrrrrrr} 
            \toprule
            & & & & \multicolumn{2}{c}{original} && \multicolumn{4}{c}{10\% projection} && \multicolumn{4}{c}{20\% projection} \\
            \cmidrule{5-6} \cmidrule{8-11} \cmidrule{13-16}
            \rule{0pt}{10pt} % Adding space of 10pt between lines and text below
            Instance & n & C & clauses & Value & Time && Size & Value & Time & Qlt && Size & Value & Time & Qlt \\
            \midrule
    """
    print(table_header)

    alphabetical_dir = sorted(
        [file for file in os.listdir(directory) if file.endswith(".pkl")],
        key=lambda x: int("".join([i for i in x if i.isdigit()])),
    )
    alphabetical_dir = [
        file
        for file in os.listdir(directory)
        if file.endswith(".pkl") and "1000" in file
    ]

    for name in alphabetical_dir:
        file_path = os.path.join(directory, name)
        with open(file_path, "rb") as file:
            results = pickle.load(file)
        name = name.strip(".pkl").replace("_", "-")
        first_quality = (
            results[projector_type][percentage[0]]["objective"]
            / results["original"]["objective"]
            * 100
        )
        second_quality = (
            results[projector_type][percentage[1]]["objective"]
            / results["original"]["objective"]
            * 100
        )
        print(
            "             {:8} & {:8} & {:8} & {:8f} & {:8.2f} & {:8.2f} & & {:8f} & {:8.2f} & {:8.2f} & {:8.2f} & & {:8f} & {:8.2f} & {:8.2f} & {:8.2f}  \\\\".format(
                "f-" + name,
                int(name.split("-")[0]),
                results["original"]["C"],
                int(int(results["original"]["C"]) * float(name.split("-")[0])),
                results["original"]["objective"],
                results["original"]["computation_time"],
                int(results[projector_type][percentage[0]]["size_psd_variable"]),
                results[projector_type][percentage[0]]["objective"],
                results[projector_type][percentage[0]]["computation_time"],
                first_quality,
                int(results[projector_type][percentage[1]]["size_psd_variable"]),
                results[projector_type][percentage[1]]["objective"],
                results[projector_type][percentage[1]]["computation_time"],
                second_quality,
            )
        )

    table_footer = r"""
            \bottomrule
        \end{tabular}}
        \label{tab:my_label}
    \end{table}
    """
    print(table_footer)


def maxsat_to_latex_simplified(directory, percentage=[0.1, 0.2]):
    """
    Convert the results of the maxsat problem to a LaTeX table.
    """

    # Create the table header
    table_header = r"""
    \begin{table}[!htbp]
        \centering
        \captionof{table}{Computational results \textsc{max-2-sat}}
        \begin{tabular}{lrrrrrrrrrrrrrrr} 
            \toprule
            & & && \multicolumn{2}{c}{10\% projection} && \multicolumn{2}{c}{20\% projection} \\
            \cmidrule{5-6} \cmidrule{8-9}
            \rule{0pt}{10pt} % Adding space of 10pt between lines and text below
            Instance & n & C && Time & Quality && Time & Quality \\
            \midrule
    """
    print(table_header)

    alphabetical_dir = sorted(
        [file for file in os.listdir(directory) if file.endswith(".pkl")],
        key=lambda x: int("".join([i for i in x if i.isdigit()])),
    )
    gen = (name for name in alphabetical_dir if "urand" in name)
    for name in gen:
        file_path = os.path.join(directory, name)
        with open(file_path, "rb") as file:
            results = pickle.load(file)
        try:
            n = int(name.split("_")[0])
            c = int(float(results["original"]["C"]) * float(name.split("_")[0]))
            name = "f-" + name.split(".pkl")[0].replace("_", "-")
        except:
            name = name.split(".pkl")[0]
            n = name.split("_")[1]
            c = name.split("_")[2].split(".")[0]
            name = name.replace("_", "-")

        first_projector_type = config["densities"][
            results["original"]["size_psd_variable"] - 1
        ][0]
        # second_projector_type = config["densities"][min(config["densities"], key=lambda x:abs(x - results["original"]["size_psd_variable"] + 1000))][0] # Flawed
        second_projector_type = "0.03_density"
        try:
            first_quality = (
                results[first_projector_type][percentage[0]]["objective"]
                / results["original"]["objective"]
                * 100
            )
            first_time = (
                results[first_projector_type][percentage[0]]["computation_time"]
                / results["original"]["computation_time"]
                * 100
            )
            second_quality = (
                results[second_projector_type][percentage[1]]["objective"]
                / results["original"]["objective"]
                * 100
            )
            second_time = (
                results[second_projector_type][percentage[1]]["computation_time"]
                / results["original"]["computation_time"]
                * 100
            )
            print(
                "             {:8} & {:5} & {:8} && {:6.2f} & {:2.2f} && {:6.2f} & {:2.2f}  \\\\".format(
                    name,
                    n,
                    results["original"]["C"],
                    # c,
                    first_time,
                    first_quality,
                    second_time,
                    second_quality,
                )
            )
        except:
            pass

    table_footer = r"""
            \bottomrule
        \end{tabular}
        \label{tab:my_label}
    \end{table}
    """
    print(table_footer)


def sat_to_latex_simplified(config, percentage=[0.1, 0.2]):
    """
    Convert the results of the 2-SAT problem to a LaTeX table.
    """

    # Create the table header
    table_header = r"""
    \begin{table}[!htbp]
    \centering
    \captionof{table}{Proportion (\%) of feasible (satisfiable) instances of \textsc{2-sat} that are also feasible after projection}
    \begin{tabular}{lrrrrrrrrrrrrrrr} 
        \toprule
        n & C & 20\% projection & 50\% projection \\
    """
    print(table_header)

    def number(string):
        try:
            return int(string)
        except:
            return float(string)

    dir_list = [file for file in os.listdir("results/sat") if file.endswith(".pkl")]
    n_list = set([int(file.split("_")[0]) for file in dir_list])
    C_list = set([number(file.split("_")[1]) for file in dir_list])

    for n in sorted(n_list):
        epoch = 0
        print("             \midrule")
        for C in sorted(C_list):
            seeds = [file.split("_")[2] for file in dir_list if f"{n}_{C}" in file]
            no_feasible = 0
            first_no_projected_feasible = 0
            second_no_projected_feasible = 0
            for seed in seeds:
                file_path = os.path.join("results/sat", f"{n}_{C}_{seed}")
                with open(file_path, "rb") as file:
                    results = pickle.load(file)
                if results["original"]["objective"] == 1:
                    no_feasible += 1
                    projector_type = config["densities"][
                        results["original"]["size_psd_variable"] - 1
                    ][0]
                    if results[projector_type][percentage[0]]["objective"] == 1:
                        first_no_projected_feasible += 1
                    if results[projector_type][percentage[1]]["objective"] == 1:
                        second_no_projected_feasible += 1

            if no_feasible == 0:
                first_proportion = "N/A"
                second_proportion = "N/A"
            else:
                first_proportion = round(
                    first_no_projected_feasible / no_feasible * 100
                )
                second_proportion = round(
                    second_no_projected_feasible / no_feasible * 100
                )

            if epoch == 0:
                print(
                    "             {:8} & {:8} & {:8} & {:8} \\\\".format(
                        n,
                        C,
                        str(first_proportion)
                        + " ({}/{})".format(first_no_projected_feasible, no_feasible),
                        str(second_proportion)
                        + " ({}/{})".format(second_no_projected_feasible, no_feasible),
                    )
                )
            else:
                print(
                    "             {:8} & {:8} & {:8} & {:8} \\\\".format(
                        " ",
                        C,
                        str(first_proportion)
                        + " ({}/{})".format(first_no_projected_feasible, no_feasible),
                        str(second_proportion)
                        + " ({}/{})".format(second_no_projected_feasible, no_feasible),
                    )
                )

            epoch += 1

    table_footer = r"""
            \bottomrule
        \end{tabular}
        \label{tab:sat feasibility}
    \end{table}
    """

    print(table_footer)


def sparsity_test_to_latex(directory, percentage=[0.05, 0.1]):
    table_header = r"""
    \begin{table}[!htbp]
    \captionof{table}{Comparison of relative time (\%) and relative quality (\%) with respect to the original SDP relaxation for different sparsities (\%) of sparse projectors from \cite{DAmbrosio2020} for a 5\% and 10\% projection of a 10.000 nodes  rudy graph with 20\% edge density} 
    \begin{tabular}{rrrrrrrrrrrrrrrr} 
    \toprule
        && \multicolumn{2}{c}{5\% projection} && \multicolumn{2}{c}{10\% projection} \\
        \cmidrule{3-4}\cmidrule{6-7}
        \rule{0pt}{10pt} % Adding space of 10pt between lines and text below
        Sparsity && Time & Quality && Time & Quality \\
        \midrule
        """
    print(table_header)

    with open(os.path.join(directory, "mcp_10000_20_1.pkl"), "rb") as file:
        results = pickle.load(file)

    for sparsity in sorted(
        [density for density in results.keys() if density != "original"]
    ):
        print(
            "             {:8} && {:8.2f} & {:8.2f} && {:8.2f} & {:8.2f} \\\\".format(
                int((1 - float(sparsity.split("_")[0])) * 100),
                results[sparsity][percentage[0]]["computation_time"]
                / results["original"]["computation_time"]
                * 100,
                results[sparsity][percentage[0]]["objective"]
                / results["original"]["objective"]
                * 100,
                results[sparsity][percentage[1]]["computation_time"]
                / results["original"]["computation_time"]
                * 100,
                results[sparsity][percentage[1]]["objective"]
                / results["original"]["objective"]
                * 100,
            )
        )

    table_footer = r"""
        \bottomrule
        \end{tabular}
        \label{tab:main rudy-7000}
    \end{table}"""

    print(table_footer)


def qcqp_to_latex(directory):
    table_header = r"""
    \begin{table}[!htbp]
    \captionof{table}{Comparison of relative time (\%) and relative quality (\%) with respect to the original SDP relaxation for different sparsities (\%) of sparse projectors from \cite{DAmbrosio2020}} 
    \begin{tabular}{rrrrrrrrrrrrrrrr} 
    \toprule
        && \multicolumn{2}{c}{50\% projection} && \multicolumn{2}{c}{70\% projection} \\
        \cmidrule{3-4}\cmidrule{6-7}
        \rule{0pt}{10pt} % Adding space of 10pt between lines and text below
        Instance && Time & Quality && Time & Quality \\
        \midrule
        """
    print(table_header)

    alphabetical_dir = sorted(
        [file for file in os.listdir(directory) if file.endswith(".pkl")],
        key=lambda x: int("".join([i for i in x if i.isdigit()])),
    )

    for name in alphabetical_dir:
        file_path = os.path.join(directory, name)
        with open(file_path, "rb") as file:
            results = pickle.load(file)

        # projector_type = config["densities"][
        #     results["original"]["size_psd_variable"] - 1
        # ][0]
        # Pick projector type from config
        matrix_size = results["original"]["size_psd_variable"]
        pick_key = [key for key in config["densities"] if matrix_size in range(int(key.split(",")[0]), int(key.split(",")[1]))]
        projector_type = config["densities"][pick_key[0]][0]

        first_projector_type = projector_type
        second_projector_type = projector_type

        if results["original"]["objective"] == 0:
            if results[first_projector_type][0.5]["objective"] != 0:
                first_symbol = "*"
            else:
                first_symbol = "-"

            if results[second_projector_type][0.7]["objective"] != 0:
                second_symbol = "*"
            else:
                second_symbol = "-"

            print(
            "             {:8} && {:8} & {:8} && {:8} & {:8} \\\\".format(
                name.strip(".pkl").replace("_", "-"),
                first_symbol,
                first_symbol,
                second_symbol,
                second_symbol,
            )
        )

        else:
            # print(
            #     "             {:8} && {:8.2f} & {:8.2f} && {:8.2f} & {:8.2f} \\\\".format(
            #         name.strip(".pkl").replace("_", "-"),
            #         results[first_projector_type][0.5]["computation_time"]
            #         / results["original"]["computation_time"]
            #         * 100,
            #         (
            #             results[first_projector_type][0.5]["objective"]
            #             - results["original"]["objective"]
            #         )
            #         / results["original"]["objective"]
            #         * 100,
            #         results[second_projector_type][0.7]["computation_time"]
            #         / results["original"]["computation_time"]
            #         * 100,
            #         (
            #             results[second_projector_type][0.7]["objective"]
            #             - results["original"]["objective"]
            #         )
            #         / results["original"]["objective"]
            #         * 100,
            #     )
            # )

            print(
                "             {:8} && {:8.2f} & {:8.2f} && {:8.2f} & {:8.2f} \\\\".format(
                    name.strip(".pkl").replace("_", "-"),
                    results[first_projector_type][0.5]["computation_time"]
                    / results["original"]["computation_time"]
                    * 100,
                    (
                        results[first_projector_type][0.5]["objective"]
                        # - results["original"]["objective"]
                    )
                    / results["original"]["objective"]
                    * 100,
                    results[second_projector_type][0.7]["computation_time"]
                    / results["original"]["computation_time"]
                    * 100,
                    (
                        results[second_projector_type][0.7]["objective"]
                        # - results["original"]["objective"]
                    )
                    / results["original"]["objective"]
                    * 100,
                )
            )

    table_footer = r"""
        \bottomrule
        \end{tabular}
        \label{tab:main qcqp}
    \end{table}"""

    print(table_footer)

def unit_sphere_to_latex(projector_type, projection):
    """
        \begin{table}[!htbp]
    \centering
    \captionof{table}{Optimization over unit sphere X \% projection}
    \begin{tabular}{lrrrrrrrrrrr} 
        \toprule
        & & && \multicolumn{2}{c}{VR} && \multicolumn{2}{c}{CA} &&  \multicolumn{2}{c}{VR + CA}\\
        \cmidrule{5-6} \cmidrule{8-9} \cmidrule{11-12} \cmidrule{11-12}
        \rule{0pt}{10pt} % Adding space of 10pt between lines and text below
        instance & $X$  & $m$ && Qlt & Time && Qlt & Time && Qlt & Time \\
        \midrule
             form-4-70-1 &       60 &     1680 &&      152 &     1831 &&      183 &      100 &&     2.24 &  \\
             form-4-70-1 &       60 &     1680 &&      152 &     1831 &&      183 &      100 &&     2.24 &  \\
        \bottomrule
    \end{tabular}
    \label{tab: unit sphere}
    \end{table}
    """
    header = r"""
     \begin{table}[!htbp]
    \centering
    """

    print(header)
    print(r"\captionof{table}{Optimization over unit sphere ", str(int(projection * 100)), r"\% projection}")
    
    header = r"""
    \begin{adjustbox}{width=\textwidth}
    \begin{tabular}{lrrrrrrrrrrrrrrrrr} 
        \toprule
        & & && \multicolumn{2}{c}{original} && \multicolumn{2}{c}{VR} && \multicolumn{2}{c}{CA} &&  \multicolumn{2}{c}{VR + CA} \\
        \cmidrule{5-6} \cmidrule{8-9} \cmidrule{11-12} \cmidrule{14-15} 
        \rule{0pt}{10pt} % Adding space of 10pt between lines and text below
        instance & $X$  & $m$ && Value & Time && Value & Time && Value & Time && Value & Time \\
        \midrule
    """
    print(header)

    directory = "results/unit_sphere"

    alphabetical_dir = sorted(
        [file for file in os.listdir(directory) if file.endswith(".pkl")],
        key=lambda x: int("".join([i for i in x if i.isdigit()])),
    )

    for name in alphabetical_dir:
        file_path = os.path.join(directory, name)
        with open(file_path, "rb") as file:
            results = pickle.load(file)

        original_value = results["original"]["objective"]
        original_time = results["original"]["computation_time"]


        # print(
        #     "             {:8} & {:8} & {:8} && {:8} & {:8} && {:8} & {:8} && {:8} & {:8} \\\\".format(
        #         name.strip(".pkl"),
        #         results["original"]["size_psd_variable"],
        #         results["original"]["no_constraints"],
        #         results[projector_type][projection]["variable_reduction"]["objective"] / original_value * 100,
        #         results[projector_type][projection]["variable_reduction"]["computation_time"] / original_time * 100,
        #         results[projector_type][projection]["constraint_aggregation"]["objective"] / original_value * 100,
        #         results[projector_type][projection]["constraint_aggregation"]["computation_time"] / original_time * 100,
        #         results[projector_type][projection]["combined_projection"]["objective"] / original_value * 100,
        #         results[projector_type][projection]["combined_projection"]["computation_time"] / original_time * 100,
        #     )
        # )

        print(
            "             {:8} & {:8} & {:8} && {:8.2f} & {:8.2f} && {:8.2f} & {:8.2f} && {:8.2f} & {:8.2f} && {:8.2f} & {:8.2f} \\\\".format(
                name.strip(".pkl"),
                results["original"]["size_psd_variable"],
                results["original"]["no_constraints"],
                results["original"]["objective"],
                results["original"]["computation_time"],
                results[projector_type][projection]["variable_reduction"]["objective"],
                results[projector_type][projection]["variable_reduction"]["computation_time"],
                results[projector_type][projection]["constraint_aggregation"]["objective"],
                results[projector_type][projection]["constraint_aggregation"]["computation_time"],
                results[projector_type][projection]["combined_projection"]["objective"],
                results[projector_type][projection]["combined_projection"]["computation_time"],
            )
        )

    footer = r"""
        \bottomrule
        \end{tabular}
        \end{adjustbox}
        \label{tab: unit sphere}
    \end{table}
    """
    print(footer)

def vertical_unit_sphere(name):
    """
    """

    header = r"""
        \begin{table}[!htbp]
        \centering
        \captionof{table}{Comparison of different projectors and projections for the optimization of a quartic form with 10 variables over the unit sphere}
        \begin{adjustbox}{width=\textwidth}
        \begin{tabular}{llrrrrrrrrrrrrrrrr} 
            \toprule
            Instance & Type & Projection & $X$  & $m$ & Value & Time(s) & APM   \\
            """
    print(header)

    directory = "results/unit_sphere"
    alphabetical_dir = sorted(
        [file for file in os.listdir(directory) if file.endswith(".pkl")],
        key=lambda x: int("".join([i for i in x if i.isdigit()])),
    )

    alphabetical_dir = [name]

    for name in alphabetical_dir[:1]:
        print(r"\midrule")
        file_path = os.path.join(directory, name)
        with open(file_path, "rb") as file:
            results = pickle.load(file)

        original_value = results["original"]["objective"]
        original_time = results["original"]["computation_time"]

        print(
            "             {:8} & {:8} & {:8} & {:8} & {:8} & {:8.2f} & {:8.2f} & - \\\\".format(
                name.strip(".pkl"),
                "original",
                "-",
                results["original"]["size_psd_variable"],
                results["original"]["no_constraints"],
                results["original"]["objective"],
                results["original"]["computation_time"],
            )
        )
        print(r"\cmidrule{2-8}")

        # Pick projector type from config
        matrix_size = results["original"]["size_psd_variable"]
        pick_key = [key for key in config["densities"] if matrix_size in range(int(key.split(",")[0]), int(key.split(",")[1]))]
        type_variable = config["densities"][pick_key[0]][0]
        no_constraints = results["original"]["no_constraints"]
        pick_key = [key for key in config["densities"] if no_constraints in range(int(key.split(",")[0]), int(key.split(",")[1]))]
        type_constraints = config["densities"][pick_key[0]][0]
        projections = list(results[(type_variable, type_constraints)].keys())

        for projection_type in ["variable_reduction", "constraint_aggregation", "combined_projection"]:
            print(
                    "   &  {:8} & {:8} & {:8} & {:8} & {:8.2f} & {:8.2f} & -  \\\\".format(
                        projection_type.replace("_", " "),
                        projections[0],
                        results[(type_variable, type_constraints)][projections[0]][projection_type]["size_psd_variable"],
                        results[(type_variable, type_constraints)][projections[0]][projection_type]["no_constraints"],
                        results[(type_variable, type_constraints)][projections[0]][projection_type]["objective"],
                        results[(type_variable, type_constraints)][projections[0]][projection_type]["computation_time"],
                    )
                )
            for projection in projections[1:]:
                print(
                    "   &  & {:8} & {:8} & {:8} & {:8.2f} & {:8.2f} & -  \\\\".format(
                        projection,
                        results[(type_variable, type_constraints)][projection][projection_type]["size_psd_variable"],
                        results[(type_variable, type_constraints)][projection][projection_type]["no_constraints"],
                        results[(type_variable, type_constraints)][projection][projection_type]["objective"],
                        results[(type_variable, type_constraints)][projection][projection_type]["computation_time"],
                    )
                )

            if projection_type != "combined_projection":
                print(r"\cmidrule{2-8}")

    footer = r"""
        \bottomrule
        \end{tabular}
        \end{adjustbox}
        \label{tab: unit sphere}
    \end{table}
    """

    print(footer)

def vertical_stable_set(graph_class, name):
    """
    Print table for a single graph comparing all projections
    """
    header = r"""
        \begin{table}[!htbp]
        \centering
        \captionof{table}{Comparison of different projectors and projections for the graph X for stable set problem}
        \begin{adjustbox}{width=\textwidth}
        \begin{tabular}{llrrrrrrrrrrrrrrrr} 
            \toprule
            Instance & Type & Projection & $X$  & $m$ & Value & Time(s) & APM   \\
            """
    print(header)

    directory = "results/new_stable_set/" + graph_class
    
    alphabetical_dir = [name]

    for name in alphabetical_dir[:1]:
        print(r"\midrule")
        file_path = os.path.join(directory, name)
        with open(file_path, "rb") as file:
            results = pickle.load(file)

        original_value = results["original"]["objective"]
        original_time = results["original"]["computation_time"]

        print(
            "             {:8} & {:8} & {:8} & {:8} & {:8} & {:8.2f} & {:8.2f} & - \\\\".format(
                name.strip(".pkl"),
                "original",
                "-",
                results["original"]["size_psd_variable"],
                results["original"]["no_constraints"],
                results["original"]["objective"],
                results["original"]["computation_time"],
            )
        )
        print(r"\cmidrule{2-8}")

        # Pick projector type from config
        matrix_size = results["original"]["size_psd_variable"]
        pick_key = [key for key in config["densities"] if matrix_size in range(int(key.split(",")[0]), int(key.split(",")[1]))]
        type_variable = config["densities"][pick_key[0]][0]
        no_constraints = results["original"]["no_constraints"]
        pick_key = [key for key in config["densities"] if no_constraints in range(int(key.split(",")[0]), int(key.split(",")[1]))]
        type_constraints = config["densities"][pick_key[0]][0]
        projections = list(results[(type_variable, type_constraints)].keys())

        for projection_type in ["variable_reduction", "constraint_aggregation", "combined_projection"]:
            print(
                    "   &  {:8} & {:8} & {:8} & {:8} & {:8.2f} & {:8.2f} & -  \\\\".format(
                        projection_type.replace("_", " "),
                        projections[0],
                        results[(type_variable, type_constraints)][projections[0]][projection_type]["size_psd_variable"],
                        results[(type_variable, type_constraints)][projections[0]][projection_type]["no_constraints"],
                        results[(type_variable, type_constraints)][projections[0]][projection_type]["objective"],
                        results[(type_variable, type_constraints)][projections[0]][projection_type]["computation_time"],
                    )
                )
            for projection in projections[1:]:
                print(
                    "   &  & {:8} & {:8} & {:8} & {:8.2f} & {:8.2f} & -  \\\\".format(
                        projection,
                        results[(type_variable, type_constraints)][projection][projection_type]["size_psd_variable"],
                        results[(type_variable, type_constraints)][projection][projection_type]["no_constraints"],
                        results[(type_variable, type_constraints)][projection][projection_type]["objective"],
                        results[(type_variable, type_constraints)][projection][projection_type]["computation_time"],
                    )
                )

            if projection_type != "combined_projection":
                print(r"\cmidrule{2-8}")

    footer = r"""
        \bottomrule
        \end{tabular}
        \end{adjustbox}
        \label{tab: unit sphere}
    \end{table}
    """

    print(footer)

def comparison_stable_set(projection):
    """
    Print table comparing different graphs for a specific projection
    """

    header = r"""
    \begin{table}[!htbp]
    \centering"""

    print(header)

    print(r"""\captionof{table}{Optimization complement graphs of stable set for """, int(projection * 100), "\% projection}")
    
    header = r"""
    \begin{adjustbox}{width=\textwidth}
    \begin{tabular}{lrrrrrrrrrrrrrrrrr} 
        \toprule
        & & & && \multicolumn{2}{c}{original} && \multicolumn{2}{c}{VR} && \multicolumn{2}{c}{CA} &&  \multicolumn{2}{c}{VR + CA} \\
        \cmidrule{6-7} \cmidrule{9-10} \cmidrule{12-13} \cmidrule{15-16} 
        \rule{0pt}{10pt} % Adding space of 10pt between lines and text below
        instance & $X$  & $e$ & $m$ && Value & Time && Value & Time && Value & Time && Value & Time \\
        \midrule
    """
    print(header)

    directory = "results/new_stable_set"
    graphs = ["petersen", "cordones", "helm", "jahangir"]

    for graph in graphs:
        dir = os.path.join(directory, graph)

        alphabetical_dir = sorted(
            [file for file in os.listdir(dir) if file.endswith(".pkl")],
            key=lambda x: int("".join([i for i in x if i.isdigit()])),
        )
        for name in alphabetical_dir:
            file_path = os.path.join(dir, name)
            with open(file_path, "rb") as file:
                results = pickle.load(file)

            # Pick projector type from config
            matrix_size = results["original"]["size_psd_variable"]
            pick_key = [key for key in config["densities"] if matrix_size in range(int(key.split(",")[0]), int(key.split(",")[1]))]
            type_variable = config["densities"][pick_key[0]][0]
            no_constraints = results["original"]["no_constraints"]
            pick_key = [key for key in config["densities"] if no_constraints in range(int(key.split(",")[0]), int(key.split(",")[1]))]
            type_constraints = config["densities"][pick_key[0]][0]

            original_value = results["original"]["objective"]
            original_time = results["original"]["computation_time"]

            print(
                "             {:8} & {:8} & {:8} & {:8} && {:8.2f} & {:8.2f} && {:8.2f} & {:8.2f} && {:8.2f} & {:8.2f} && {:8.2f} & {:8.2f}  \\\\".format(
                    "c-" + graph + "-" + name.strip("_complement.pkl").replace("_", "-"),
                    results["original"]["size_psd_variable"],
                    results["original"]["edges"],
                    results["original"]["no_constraints"],
                    results["original"]["objective"],
                    results["original"]["computation_time"],
                    results[(type_variable, type_constraints)][projection]["variable_reduction"]["objective"],
                    results[(type_variable, type_constraints)][projection]["variable_reduction"]["computation_time"],
                    results[(type_variable, type_constraints)][projection]["constraint_aggregation"]["objective"],
                    results[(type_variable, type_constraints)][projection]["constraint_aggregation"]["computation_time"],
                    results[(type_variable, type_constraints)][projection]["combined_projection"]["objective"],
                    results[(type_variable, type_constraints)][projection]["combined_projection"]["computation_time"],
                )
            )

    footer = r"""
        \bottomrule
        \end{tabular}
        \end{adjustbox}
        \label{tab: unit sphere}
    \end{table}
    """

    print(footer)

def vertical_first_stable_set(graph_class, name):
    """
    Print table for a single graph comparing all projections
    """
    header = r"""
        \begin{table}[!htbp]
        \centering
        \captionof{table}{Comparison of different projectors and projections for the graph X for stable set problem}
        \begin{adjustbox}{width=\textwidth}
        \begin{tabular}{llrrrrrrrrrrrrrrrr} 
            \toprule
            Instance & Type & Projection & $X$  & $m$ & Value & Time(s) & APM   \\
            """
    print(header)

    directory = "results/first_stable_set/" + graph_class
    
    alphabetical_dir = [name]

    for name in alphabetical_dir[:1]:
        print(r"\midrule")
        file_path = os.path.join(directory, name)
        with open(file_path, "rb") as file:
            results = pickle.load(file)

        original_value = results["original"]["objective"]
        original_time = results["original"]["computation_time"]

        print(
            "             {:8} & {:8} & {:8} & {:8} & {:8} & {:8.2f} & {:8.2f} & - \\\\".format(
                name.strip(".pkl"),
                "original",
                "-",
                results["original"]["size_psd_variable"],
                results["original"]["no_constraints"],
                results["original"]["objective"],
                results["original"]["computation_time"],
            )
        )
        print(r"\cmidrule{2-8}")

        # Pick projector type from config
        matrix_size = results["original"]["size_psd_variable"]
        pick_key = [key for key in config["densities"] if matrix_size in range(int(key.split(",")[0]), int(key.split(",")[1]))]
        type_variable = config["densities"][pick_key[0]][0]
        no_constraints = results["original"]["no_constraints"]
        pick_key = [key for key in config["densities"] if no_constraints in range(int(key.split(",")[0]), int(key.split(",")[1]))]
        type_constraints = config["densities"][pick_key[0]][0]
        projections = list(results[(type_variable, type_constraints)].keys())

        for projection_type in ["variable_reduction", "constraint_aggregation", "combined_projection"]:
            print(
                    "   &  {:8} & {:8} & {:8} & {:8} & {:8.2f} & {:8.2f} & -  \\\\".format(
                        projection_type.replace("_", " "),
                        projections[0],
                        results[(type_variable, type_constraints)][projections[0]][projection_type]["size_psd_variable"],
                        results[(type_variable, type_constraints)][projections[0]][projection_type]["no_constraints"],
                        results[(type_variable, type_constraints)][projections[0]][projection_type]["objective"],
                        results[(type_variable, type_constraints)][projections[0]][projection_type]["computation_time"],
                    )
                )
            for projection in projections[1:]:
                print(
                    "   &  & {:8} & {:8} & {:8} & {:8.2f} & {:8.2f} & -  \\\\".format(
                        projection,
                        results[(type_variable, type_constraints)][projection][projection_type]["size_psd_variable"],
                        results[(type_variable, type_constraints)][projection][projection_type]["no_constraints"],
                        results[(type_variable, type_constraints)][projection][projection_type]["objective"],
                        results[(type_variable, type_constraints)][projection][projection_type]["computation_time"],
                    )
                )

            if projection_type != "combined_projection":
                print(r"\cmidrule{2-8}")

    footer = r"""
        \bottomrule
        \end{tabular}
        \end{adjustbox}
        \label{tab: unit sphere}
    \end{table}
    """

    print(footer)


with open("config.yml", "r") as file:
    config = yaml.safe_load(file)


if __name__ == "__main__":
    # maxcut_to_latex("results/maxcut", config, "0.05_density", [0.1, 0.1])
    # maxcut_to_latex_single("results/maxcut", config, "0.04_density", 0.1)
    # maxcut_to_latex_single_simplified("results/maxcut", config, "0.04_density", 0.1)
    # stable_set_to_latex("results/stable_set")
    # maxsat_to_latex("results/maxsat", "sparse", [0.1, 0.2])
    # maxsat_to_latex_simplified("results/maxsat", [0.1, 0.2])
    # sparsity_test_to_latex("results/maxcut")
    # sat_to_latex_simplified(config, [0.2, 0.5])
    qcqp_to_latex("results/qcqp")
    # unit_sphere_to_latex("0.2_density", 0.9)
    # vertical_unit_sphere("form-4-10-1.pkl")
    # vertical_stable_set("petersen", "20_3_complement.pkl")
    # comparison_stable_set(0.7)
    # vertical_first_stable_set("petersen", "30_2.pkl")
