import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Load your DataFrame
df = pd.read_csv('tag_diff.csv')

# Set the first column as the index if it represents the metrics
df = df.set_index(df.columns[0])

# Extracting unique thresholds and chunk sizes from column names
thresholds = sorted({float(col.split('_')[0]) for col in df.columns}, reverse=True)
chunk_sizes = sorted({int(col.split('_')[1]) for col in df.columns})

# Specify the metrics for which you want to generate heatmaps
selected_metrics = ['rouge-l.r', 'rouge-l.p', 'rouge-l.f']  # Replace with your actual metric names
type_map = {
    'r': 'Recall',
    'p': 'Precision',
    'f': 'F1 Score'
}

def metric_to_title(metric):
    rouge_type, subtype = metric.split('.')
    return f'{rouge_type.upper()} {type_map[subtype]}'

# Filter the DataFrame to include only the selected metrics
df_filtered = df.loc[selected_metrics]

# Calculate global min and max across the selected metrics for consistent color scaling
abs_global_max = max(abs(df_filtered.min().min()), abs(df_filtered.max().max())) * 100

# Create a custom colormap
colors = ["red", "white", "green"]  # Red for negative, white for zero, green for positive
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

# Plot heatmaps for each specified metric
plt.figure(figsize=(4, len(selected_metrics) * 3))  # Adjust size as needed

for i, metric in enumerate(selected_metrics):
    plt.subplot(len(selected_metrics), 1, i + 1)

    # Reshape and convert data to percentages for the current metric
    heatmap_data = (df.loc[metric].values * 100).reshape(len(thresholds), len(chunk_sizes))
    heatmap_data = np.flipud(heatmap_data) 
    annot_array = np.array([["{:.1f}%".format(value) for value in row] for row in heatmap_data])

    sns.heatmap(heatmap_data, annot=annot_array, fmt="", cmap=cmap, vmin=-abs_global_max, vmax=abs_global_max,
                xticklabels=chunk_sizes, yticklabels=thresholds)
    plt.title(f'Absolute %diff in {metric_to_title(metric)}')
    plt.xlabel('Chunk Size')
    plt.ylabel('Threshold')

plt.tight_layout()
plt.savefig('rouge-l')
plt.show()
