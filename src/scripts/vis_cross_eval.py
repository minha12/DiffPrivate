import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

import matplotlib as mpl
import os

# Use TrueType (Type 42) fonts to ensure embedding
mpl.rcParams['pdf.fonttype'] = 42  # TrueType fonts
mpl.rcParams['ps.fonttype'] = 42  # TrueType fonts

# Create the directory if it doesn't exist
os.makedirs("./visualized", exist_ok=True)

# Load the CSV file
data = pd.read_csv("./experiments_cross/logs/summary_success_rates.txt", sep=",")

# Convert success rate columns to numeric, replacing non-numeric values with NaN
success_rate_columns = ["IRSE50", "IR152", "FaceNet", "IR101", "MobileFace"]
for column in success_rate_columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")

# Get unique attackers
attackers = data["Attacker"].unique()

# Define the models to be included in plots
models = ["IRSE50", "IR152", "FaceNet", "IR101", "MobileFace"]

# Define a color scheme (shades of blue)
colors = [
    "skyblue",
    "lightgreen",
    "salmon",
    "lightpink",
    "wheat",
    "violet",
    "lightskyblue",
]

# Define a color mapping for each model
color_mapping = {
    "IRSE50": "skyblue",
    "IR152": "lightgreen",
    "FaceNet": "salmon",
    "IR101": "lightpink",
    "MobileFace": "wheat",
    # Add more if you have more models
}

# Define hatch patterns if you want to keep using them
hatch_patterns = [
    "+",
    "x",
    "o",
    ".",
]  # This can remain the same or be adjusted as needed


hatch_patterns = ["+", "x", "o", "."]
bar_width = 0.3  # Adjust this value as needed
plt.rcParams.update({"font.size": plt.rcParams["font.size"] * 1.5})
# Iterate over each attacker, creating plots
# Define a hatch mapping for each model
hatch_mapping = {
    "IRSE50": "+",
    "IR152": "x",
    "FaceNet": "o",
    "IR101": ".",
    "MobileFace": "*",  # Example: added a new pattern
    # Add more if you have more models
}

# Adjust the plotting section to use the hatch mapping
for attacker in attackers:
    fig, axs = plt.subplots(1, 4, figsize=(8, 4), sharey=True)
    # fig.suptitle(f"Pri: {attacker}", fontsize=16)

    # Set the y-axis label only for the leftmost subplot
    axs[0].set_ylabel("PPR")

    attacker_data = data[data["Attacker"] == attacker]
    for i, (index, row) in enumerate(attacker_data.iterrows()):
        plot_models = [
            model for model in models if model not in [attacker, row["Victim"]]
        ]
        success_rates = row[plot_models].values

        for j, model in enumerate(plot_models):
            bar = axs[i].bar(
                model,
                success_rates[j],
                color=color_mapping[model],  # Use specific color for each model
                hatch=hatch_mapping[model],  # Use specific hatch pattern for each model
                width=bar_width,  # Optional: adjust bar width as previously discussed
            )
            height = success_rates[j]

        axs[i].set_title(f'Val: {row["Victim"]}')
        axs[i].set_ylim(0, 1)

        axs[i].tick_params(axis="x", labelrotation=45)

    for ax in axs:
        ax.yaxis.set_major_formatter(PercentFormatter(1))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    # Save the plot
    fig.savefig(f"./visualized/{attacker}.pdf", bbox_inches="tight")
