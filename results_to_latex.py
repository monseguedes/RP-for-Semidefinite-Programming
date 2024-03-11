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
        [file for file in os.listdir(directory) if file.endswith(".pkl")],
        key=lambda x: int("".join([i for i in x if i.isdigit()])),
    )

    for name in alphabetical_dir:
        file_path = os.path.join(directory, name)
        with open(file_path, "rb") as file:
            results = pickle.load(file)
        if results["original"]["size_psd_variable"] >= config["maxcut"]["results"]["min_vertices"] and results["original"]["size_psd_variable"] <= config["maxcut"]["results"]["max_vertices"]:
            first_ratio = results[projector_type][percentage[0]]["objective"] / results["original"]["objective"] * 100
            second_ratio = (
                results[projector_type][percentage[1]]["objective"] / results["original"]["objective"] * 100
            )

            if first_ratio > 10 and second_ratio > 10:
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


def stable_set_to_latex(directory, projector_type="sparse"):
    """
    Convert the results of the stable set problem to a LaTeX table.
    """

    # Create the table header
    table_header = r"""
    \begin{table}[!htbp]
        \centering
        \captionof{table}{Computational results Stable Set Problem}
        \begin{tabular}{lrrrrrrrrrrr} 
            \toprule
            & & & & \multicolumn{3}{c}{original 2\textsuperscript{nd} level} && \multicolumn{3}{c}{projected 2\textsuperscript{nd} level} \\
            \cmidrule{5-7} \cmidrule{9-11}
            \rule{0pt}{10pt} % Adding space of 10pt between lines and text below
            Graph & n & m & 1\textsuperscript{st} level  & Size & Value & CPU Time && Size & Value & CPU Time \\
            \midrule
    """
    print(table_header)

    petersen_dir = os.path.join(directory, "petersen")
    cordones_dir = os.path.join(directory, "cordones")

    alphabetical_dir = sorted(
        [file for file in os.listdir(petersen_dir) if file.endswith(".pkl")],
        key=lambda x: int("".join([i for i in x if i.isdigit()])),
    )
    for name in alphabetical_dir:
        file_path = os.path.join(petersen_dir, name)
        with open(file_path, "rb") as file:
            results = pickle.load(file)
        if "complement" in name:
            ordered_keys = [0.2]
        else:
            ordered_keys = [0.6]
        
        for key in ordered_keys:
            # if (
            #     round(results[key]["objective"], 2)
            #     == round(results["L2"]["objective"], 2)
            # ):  
            print(
            "             {:8} & {:8} & {:8} & {:8.2f} & {:8} & {:8.2f} & {:8.2f} & & {:8} & {:8.2f} & {:8.2f} \\\\".format(
                "petersen-" + name.strip(".pkl").replace("_", "-"),
                results["L1"]["size_psd_variable"] - 1,
                len(results["L2"]["edges"]),
                results["L1"]["objective"],
                results["L2"]["size_psd_variable"],
                results["L2"]["objective"],
                results["L2"]["computation_time"],
                results[key]["size_psd_variable"],
                results[key]["objective"],
                results[key]["computation_time"],
            )
            )


    # Cordones          
    alphabetical_dir = sorted(
        [file for file in os.listdir(cordones_dir) if file.endswith(".pkl")],
        key=lambda x: int("".join([i for i in x if i.isdigit()])),
    )
    for name in alphabetical_dir:
        file_path = os.path.join(cordones_dir, name)
        with open(file_path, "rb") as file:
            results = pickle.load(file)
        if "complement" in name:
            ordered_keys = [0.2]
        else:
            ordered_keys = [0.6]

        for key in ordered_keys:
            # if (
            #     round(results[key]["objective"], 2)
            #     == round(results["L2"]["objective"], 2)
            # ):  
            print(
            "             {:8} & {:8} & {:8} & {:8.2f} & {:8} & {:8.2f} & {:8.2f} & & {:8} & {:8.2f} & {:8.2f} \\\\".format(
                "cordones-" + name.strip(".pkl").replace("_", "-"),
                results["L1"]["size_psd_variable"] - 1,
                len(results["L2"]["edges"]),
                results["L1"]["objective"],
                results["L2"]["size_psd_variable"],
                results["L2"]["objective"],
                results["L2"]["computation_time"],
                results[key]["size_psd_variable"],
                results[key]["objective"],
                results[key]["computation_time"],
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
    alphabetical_dir = [file for file in os.listdir(directory) if file.endswith(".pkl")]

    for name in alphabetical_dir:
        file_path = os.path.join(directory, name)
        with open(file_path, "rb") as file:
            results = pickle.load(file)
        name = name.strip(".pkl").replace("_", "-")
        first_quality =  results[projector_type][percentage[0]]["objective"] / results["original"]["objective"] * 100
        second_quality = results[projector_type][percentage[1]]["objective"] / results["original"]["objective"] * 100        
        print(
            "             {:8} & {:8} & {:8} & {:8.2f} & {:8.2f} & {:8.2f} & & {:8.2f} & {:8.2f} & {:8.2f} & {:8.2f} & & {:8.2f} & {:8.2f} & {:8.2f} & {:8.2f}  \\\\".format(
                "f-" + name,
                name.split("-")[0],
                results["original"]["C"],
                int(results["original"]["C"]) * float(name.split("-")[0]),
                results["original"]["objective"],
                results["original"]["computation_time"],
                results[projector_type][percentage[0]]["size_psd_variable"],
                results[projector_type][percentage[0]]["objective"],
                results[projector_type][percentage[0]]["computation_time"],
                first_quality,
                results[projector_type][percentage[1]]["size_psd_variable"],
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

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

# maxcut_to_latex("results/maxcut", config, "sparse", [0.1, 0.2])
maxcut_to_latex("results/maxcut", config, "0.05_density", [0.1, 0.2])
# maxcut_to_latex("results/maxcut", config, "sparse", [0.1, 0.2])
# stable_set_to_latex("results/stable_set")
# maxsat_to_latex("results/maxsat", "sparse", [0.1, 0.2])
# maxsat_to_latex("results/maxsat", "0.2_density", [0.1, 0.2])
