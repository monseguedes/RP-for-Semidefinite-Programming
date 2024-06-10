""" File to fix table"""

table = r"""
        \begin{table}
       \captionof{table}{Computational time (s) and relative quality (\%) with respect to the original relaxation of \textsc{maxcut} of G-set graphs of 800 nodes for 10\% and 20\% projections \todo{change to relative time}} 
        \begin{tabular}{lrrrrrrrrrrrrrrr} 
            \toprule
            & & && \multicolumn{2}{c}{10\% projection} && \multicolumn{2}{c}{20\% projection} \\
            \cmidrule{5-6} \cmidrule{8-9}
            \rule{0pt}{10pt} % Adding space of 10pt between lines and text below
            Instance  & $m$  & Time original && Time & Quality && Time & Quality \\
            \midrule
             G1 & 19176 & 4.85 && 2.56 & 82.79 && 12.70 & 86.99 \\
             G2 & 19176 & 4.97 && 2.30 & 82.61 && 11.08 & 86.63 \\
             G3 & 19176 & 5.45 && 2.32 & 82.58 && 14.15 & 86.65 \\
             G4 & 19176 & 5.26 && 2.61 & 82.56 && 14.40 & 86.44 \\
             G5 & 19176 & 5.10 && 2.43 & 82.51 && 10.22 & 86.67 \\
             \hline
             G6 & 19176 & 4.86 && 2.56 & 18.52 && 12.41 & 37.09 \\
             G7 & 19176 & 4.83 && 2.69 & 14.33 && 11.53 & 34.08 \\
             G8 & 19176 & 5.08 && 2.85 & 13.16 && 19.55 &    33.17 \\
             G9 & 19176 & 5.07 && 3.15 & 14.96 && 16.07 &    34.67 \\
             G10 & 19176 & 6.12 && 2.78 & 14.44 &&17.92 &    34.33 \\
             \hline
             G11 & 1600 & 5.18 && 2.69 & 22.17 && 12.62 &    44.71 \\
             G12 & 1600 & 4.79  && 2.70 &  19.97  &&15.04 &    42.88 \\
             G13 & 1600 & 5.11 && 2.45 & 20.74  && 16.34 &    42.62 \\
             \hline
             G14 & 4694 & 5.09 && 2.34 &  79.77 && 13.37 &    85.80 \\
             G15 & 4661 & 5.66 && 2.48 &  79.53  && 10.83 &    85.61 \\
             G16 & 4672 & 5.03 && 2.38 &    79.98 && 11.73 &    85.94 \\
             G17 & 4667 & 6.58 && 2.59 &    79.74 && 13.01 &    85.89 \\
             \hline
             G18 & 4694 & 4.59 && 2.49 &  20.80 && 11.14 &    40.18 \\
             G19 & 4661 & 5.93 && 2.76 &    14.44  && 13.72 &    35.33 \\
             G20 & 4672 & 4.77 && 2.48 &    15.59 && 11.26 &    37.68 \\
             G21 & 4667 & 5.39 && 2.88 &    14.63 &&  11.65 &    36.51 \\
            \bottomrule
        \end{tabular}
        \label{tab: G-800}
    \end{table}
"""

data = r"""
            G1 & 19176 & 4.85 && 2.56 & 82.79 && 12.70 & 86.99 \\
             G2 & 19176 & 4.97 && 2.30 & 82.61 && 11.08 & 86.63 \\
             G3 & 19176 & 5.45 && 2.32 & 82.58 && 14.15 & 86.65 \\
             G4 & 19176 & 5.26 && 2.61 & 82.56 && 14.40 & 86.44 \\
             G5 & 19176 & 5.10 && 2.43 & 82.51 && 10.22 & 86.67 \\
             G6 & 19176 & 4.86 && 2.56 & 18.52 && 12.41 & 37.09 \\
             G7 & 19176 & 4.83 && 2.69 & 14.33 && 11.53 & 34.08 \\
             G8 & 19176 & 5.08 && 2.85 & 13.16 && 19.55 &    33.17 \\
             G9 & 19176 & 5.07 && 3.15 & 14.96 && 16.07 &    34.67 \\
             G10 & 19176 & 6.12 && 2.78 & 14.44 &&17.92 &    34.33 \\
             G11 & 1600 & 5.18 && 2.69 & 22.17 && 12.62 &    44.71 \\
             G12 & 1600 & 4.79  && 2.70 &  19.97  &&15.04 &    42.88 \\
             G13 & 1600 & 5.11 && 2.45 & 20.74  && 16.34 &    42.62 \\
             G14 & 4694 & 5.09 && 2.34 &  79.77 && 13.37 &    85.80 \\
             G15 & 4661 & 5.66 && 2.48 &  79.53  && 10.83 &    85.61 \\
             G16 & 4672 & 5.03 && 2.38 &    79.98 && 11.73 &    85.94 \\
             G17 & 4667 & 6.58 && 2.59 &    79.74 && 13.01 &    85.89 \\
             G18 & 4694 & 4.59 && 2.49 &  20.80 && 11.14 &    40.18 \\
             G19 & 4661 & 5.93 && 2.76 &    14.44  && 13.72 &    35.33 \\
             G20 & 4672 & 4.77 && 2.48 &    15.59 && 11.26 &    37.68 \\
             G21 & 4667 & 5.39 && 2.88 &    14.63 &&  11.65 &    36.51 \\
"""

# New header
table_header = r"""
    \begin{table}[!htbp]
        \captionof{table}{Relative time (\%) and relative quality (\%) with respect to the original relaxation of \textsc{maxcut} of G-graphs using a } 
        \begin{tabular}{lrrrrrrrrrrrrrrr} 
            \toprule
            & && \multicolumn{2}{c}{10\% projection} && \multicolumn{2}{c}{20\% projection} \\
            \cmidrule{4-5} \cmidrule{7-8}
            \rule{0pt}{10pt} % Adding space of 10pt between lines and text below
            Instance  & $m$ && Time & Quality && Time & Quality \\
            \midrule
    """
print(table_header)

# Remove third column from data
lines = [line.removeprefix("\n") for line in data.split("\\") if line != ""][:-1]
for latex_line in lines:
    name = latex_line.split("&")[0].strip()
    m = latex_line.split("&")[1].strip()
    time_ratio_10 = (
                float(latex_line.split("&&")[1].split("&")[0])
                / float(latex_line.split("&")[2])
                * 100
            )
    quality_ratio_10 = latex_line.split("&&")[1].split("&")[1]
    time_ratio_20 = (
        float(latex_line.split("&&")[2].split("&")[0])
        / float(latex_line.split("&")[2])
        * 100
    )
    quality_ratio_20 = latex_line.split("&&")[2].split("&")[1]

    print(f"            {name} & {m} && {time_ratio_10:.2f} & {quality_ratio_10} && {time_ratio_20:.2f} & {quality_ratio_20} \\\\")

table_footer = r"""
            \bottomrule
        \end{tabular}
        \label{tab: G-800}
    \end{table}
"""
print(table_footer)