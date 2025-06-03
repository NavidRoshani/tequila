
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random

#sys.path.append('../quanti-gin')

from quanti_gin.shared import generate_min_local_distance_edges, generate_min_global_distance_edges, read_data_file
# from quanti_gin.shared import generate_min_local_distance_edges, read_data_file

data = read_data_file(Path('results2.csv'))

grouped_methods = data.df.groupby("method")
method_names = grouped_methods.groups.keys()

# boxplot of custom_energy and fci, and SPA energy
fig, axes = plt.subplots(1, len(grouped_methods), sharey=True, figsize=(10, 5), layout='constrained')
fig.suptitle("Method energies")

fig.set_figheight(fig.get_figheight())
for i, (axis, (method_name, method)) in enumerate(zip(axes, grouped_methods)):
    axis.boxplot(method.optimized_energy)
    axis.set_title(method_name)

plt.show()

print(method_names)
combinations = []
for method_a in method_names:
    combinations.append([(method_a, method_b) for method_b in method_names if method_a != method_b])
print(combinations)
for comb_list in combinations:
    fig, axes = plt.subplots(1, len(method_names ) -1, sharey=True, figsize=(10, 5), layout='constrained')
    fig.suptitle(f"{comb_list[0][0]}")
    for plot, (a, b) in zip(axes, comb_list):
        energy_a = np.array(grouped_methods.get_group(a).optimized_energy)
        energy_b = np.array(grouped_methods.get_group(b).optimized_energy)
        error = np.abs(energy_a - energy_b)

        if a == "main.run_optimization":
            labelA= "custom"
        elif a == "quanti_gin.data_generator.run_spa_optimization":
            labelA = "spa"
        elif a == "quanti_gin.data_generator.run_fci_optimization":
            labelA = "fci"

        if b == "main.run_optimization":
            labelB = "custom"
        elif b == "quanti_gin.data_generator.run_spa_optimization":
            labelB = "spa"
        elif b == "quanti_gin.data_generator.run_fci_optimization":
            labelB = "fci"

        print(f"{labelA} {labelB} {a} {b}")
        plot.boxplot(abs(energy_a - energy_b), tick_labels=[f"{labelA} vs. {labelB}"])


plt.show()