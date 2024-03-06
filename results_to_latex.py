"""
Convert computational results to LaTeX tables.

For maxcut, we have the following table:

\begin{table}[!htbp]
    \centering
    \captionof{table}{Computational results \textsc{maxcut}}
    \begin{tabular}{lrrrrrrrrrrrrrrr} 
        \toprule
        & & & \multicolumn{2}{c}{original} && \multicolumn{4}{c}{10\% projection} && \multicolumn{4}{c}{20\% projection} \\
        \cmidrule{4-5} \cmidrule{7-10} \cmidrule{12-15}
        \rule{0pt}{10pt} % Adding space of 10pt between lines and text below
        Instance & n & m & Value & Time && Size & Value & Time & Ratio && Size & Value & Time & Ratio \\
        \midrule
        g05\_100 & 100 &  &  &  &  & & & & & \\
        G1 & 800 &  &  &  &  & & & & & \\
        G2 & 800 &  &  &  &  & & & & & \\
        G3 & 800 &  &  &  &  & & & & & \\
        \bottomrule
    \end{tabular}
    \label{tab:my_label}
\end{table}

For stable set, we have the following table:

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
         gpetersen-5-2 & 10 &  &  &  &  & & & & & \\
         cordones-6 & 20 &  &  &  &  & & & & & \\
         c-gpetersen-5-2 & 10 &  &  &  &  & & & & & \\
         c-cordones-6 & 20 &  &  &  &  & & & & & \\
         \bottomrule
    \end{tabular}
    \label{tab:my_label}
\end{table}


"""

import os
import pickle
import yaml


def maxcut_to_latex(directory, config, projector_type="sparse"):
    """
    Convert the results of the maxcut problem to a LaTeX table.
    """

    # Create the table header
    table_header = r"""
    \begin{table}[!htbp]
        \centering
        \captionof{table}{Computational results \textsc{maxcut}}
        \begin{tabular}{lrrrrrrrrrrrrrrr} 
            \toprule
            & & & \multicolumn{2}{c}{original} && \multicolumn{4}{c}{10\% projection} && \multicolumn{4}{c}{20\% projection} \\
            \cmidrule{4-5} \cmidrule{7-10} \cmidrule{12-15}
            \rule{0pt}{10pt} % Adding space of 10pt between lines and text below
            Instance & n & m & Value & Time && Size & Value & Time & Ratio && Size & Value & Time & Ratio \\
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
        if results["original"]["size_psd_variable"] >= config["maxcut"]["results"]["min_vertices"] and results[projector_type][0.1]["objective"] > 0:
            ten_ratio = results[projector_type][0.1]["objective"] / results["original"]["objective"] * 100
            twenty_ratio = (
                results[projector_type][0.2]["objective"] / results["original"]["objective"] * 100
            )

            name = name.strip(".pkl")
            print(
                "             {:8} & {:8} & {:8} & {:8.2f} & {:8.2f} \
                & & {:8} & {:8.2f} & {:8.2f} & {:8.2f} \
                & & {:8} & {:8.2f} & {:8.2f} & {:8.2f} \\\\".format(
                    name,
                    results["original"]["size_psd_variable"],
                    # results["original"]["edges"],
                    "-", 
                    results["original"]["objective"],
                    results["original"]["computation_time"],
                    results[projector_type][0.1]["size_psd_variable"],
                    results[projector_type][0.1]["objective"],
                    results[projector_type][0.1]["computation_time"],
                    ten_ratio,
                    results[projector_type][0.2]["size_psd_variable"],
                    results[projector_type][0.2]["objective"],
                    results[projector_type][0.2]["computation_time"],
                    twenty_ratio,
                )
            )
    table_footer = r"""
            \bottomrule
        \end{tabular}
        \label{tab:my_label}
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
        file_path = os.path.join(directory, name)
        with open(file_path, "rb") as file:
            results = pickle.load(file)
        ordered_keys = sorted(results['sparse'].keys())
        for key in ordered_keys:
            if (
                round(results['sparse'][key]["objective"], 2)
                != round(results["original"]["objective"], 2)
            ):
                print(
                    "             {:8} & {:8} & - & {:8.2f} & {:8} & {:8.2f} & {:8.2f} & & {:8} & {:8.2f} & {:8.2f} \\\\".format(
                        name,
                        results["original"]["size_psd_variable"],
                        results["original"]["objective"],
                        results['sparse'][key]["size_psd_variable"],
                        results['sparse'][key]["objective"],
                        results['sparse'][key]["computation_time"],
                        results['sparse'][key]["size_psd_variable"],
                        results['sparse'][key]["objective"],
                        results['sparse'][key]["computation_time"],
                    )
                )
                break

    for name in [file for file in os.listdir(cordones_dir) if file.endswith(".pkl")]:
        file_path = os.path.join(directory, name)
        with open(file_path, "rb") as file:
            results = pickle.load(file)

        ordered_keys = sorted(results['sparse'].keys())
        for key in ordered_keys:
            if (
                results['sparse'][key]["objective"]
                != results["original"]["objective"]
            ):
                break
            print(
                "             {:8} & {:8} & - & {:8.2f} & {:8} & {:8.2f} & {:8.2f} & & {:8} & {:8.2f} & {:8.2f} \\\\".format(
                    name,
                    results["original"]["size_psd_variable"],
                    results["original"]["objective"],
                    results['sparse'][key]["size_psd_variable"],
                    results['sparse'][key]["objective"],
                    results['sparse'][key]["computation_time"],
                    results['sparse'][key]["size_psd_variable"],
                    results['sparse'][key]["objective"],
                    results['sparse'][key]["computation_time"],
                )
            )

    table_footer = r"""
            \bottomrule
        \end{tabular}
        \label{tab:my_label}
    \end{table}
    """
    print(table_footer)

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

maxcut_to_latex("results/maxcut", config, "sparse")
# stable_set_to_latex("results/stable_set")
