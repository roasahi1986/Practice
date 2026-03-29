import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from pyspark.sql import SparkSession, functions as F
import random

spark = SparkSession.builder.getOrCreate()

# COMMOND -------------------------------------------------------------
# Configuration
INPUT_PATH = "s3://exchange-dev/luxu/similarity/rtb_conn_cosine"
MAX_RTBS_TO_PLOT = 100 # Limit to avoid OOM and browser lag

# 1. Load Data
print(f"Loading similarity data from {INPUT_PATH}...")

# Use Spark to read Parquet directory
df_spark = spark.read.parquet(INPUT_PATH)

# SAMPLING LOGIC
print("Sampling RTB IDs for visualization...")
# Get list of all unique RTB IDs
# We only need to check one column if the matrix is symmetric
unique_rtbs_rows = df_spark.select("rtb_id_A").distinct().collect()
unique_rtbs = [row.rtb_id_A for row in unique_rtbs_rows]
total_rtbs = len(unique_rtbs)
print(f"Total Unique RTB IDs available: {total_rtbs}")

if total_rtbs > MAX_RTBS_TO_PLOT:
    sampled_rtbs = random.sample(unique_rtbs, MAX_RTBS_TO_PLOT)
    print(f"⚠️ Data too large. Randomly sampled {len(sampled_rtbs)} RTB IDs.")
else:
    sampled_rtbs = unique_rtbs
    print("Using all RTB IDs.")

# Filter Spark DataFrame BEFORE collecting to Pandas
# Use isin for efficient filtering
df_filtered_spark = df_spark.filter(
    F.col("rtb_id_A").isin(sampled_rtbs) & 
    F.col("rtb_id_B").isin(sampled_rtbs)
)

# Convert to Pandas
df = df_filtered_spark.toPandas()
print(f"Loaded {len(df)} similarity pairs after sampling.")

# 2. Build Full Matrix
print("Building similarity matrix...")

# Get unique IDs from both columns to ensure we cover everyone
rtb_ids = sorted(list(set(df['rtb_id_A'].unique()) | set(df['rtb_id_B'].unique())))
n = len(rtb_ids)
id_to_idx = {rid: i for i, rid in enumerate(rtb_ids)}

matrix = np.zeros((n, n))

# Fill matrix
# Data contains (A, B, sim). Assuming symmetric pairs are already present or we fill them.
# The previous script generated symmetric pairs + diagonals.
# Just in case, we enforce symmetry here.

for _, row in df.iterrows():
    i = id_to_idx[row['rtb_id_A']]
    j = id_to_idx[row['rtb_id_B']]
    sim = row['cosine_similarity']
    matrix[i, j] = sim
    matrix[j, i] = sim

# Ensure diagonal is 1.0 (sometimes float precision issues)
np.fill_diagonal(matrix, 1.0)

print(f"Matrix shape: {matrix.shape}")

# 3. Hierarchical Clustering for Sorting (The 'Professional' Way)
print("Performing hierarchical clustering to reorder matrix...")

# Convert similarity to distance (1 - similarity) for clustering
# Clip to [0, 1] to handle potential float errors
distance_matrix = 1.0 - matrix
distance_matrix = np.clip(distance_matrix, 0, 1)
np.fill_diagonal(distance_matrix, 0.0)

# Condensed distance matrix for linkage
condensed_dist = squareform(distance_matrix)

# Compute linkage (Ward's method minimizes variance within clusters)
Z = linkage(condensed_dist, method='ward')

# Get sorted indices
optimal_ordering_indices = leaves_list(Z)

# Reorder matrix and labels
sorted_rtb_ids = [rtb_ids[i] for i in optimal_ordering_indices]
sorted_matrix = matrix[optimal_ordering_indices, :][:, optimal_ordering_indices]

# 4. Plotly Heatmap
print("Generating Heatmap...")

# Create Hover Text matrix
hover_text = []
for i in range(n):
    hover_row = []
    for j in range(n):
        hover_row.append(
            f"Row: {sorted_rtb_ids[i]}<br>"
            f"Col: {sorted_rtb_ids[j]}<br>"
            f"Cosine Sim: {sorted_matrix[i][j]:.4f}"
        )
    hover_text.append(hover_row)

fig = go.Figure(data=go.Heatmap(
    z=sorted_matrix,
    x=sorted_rtb_ids,
    y=sorted_rtb_ids,
    text=hover_text,
    hoverinfo='text',
    colorscale='Viridis', 
    zmin=0.0,
    zmax=1.0,
    colorbar=dict(title='Cosine Similarity')
))

fig.update_layout(
    title=f"RTB Strategy Similarity Heatmap (Clustered - {n} IDs)<br><sup>Based on Feature Preference Vectors</sup>",
    width=1200,
    height=1200,
    xaxis=dict(
        tickmode='array',
        tickvals=list(range(len(sorted_rtb_ids))),
        ticktext=sorted_rtb_ids,
        tickfont=dict(size=8),
        title="RTB ID"
    ),
    yaxis=dict(
        tickmode='array',
        tickvals=list(range(len(sorted_rtb_ids))),
        ticktext=sorted_rtb_ids,
        tickfont=dict(size=8),
        autorange='reversed',
        title="RTB ID"
    )
)

fig.show()
print("✅ Heatmap generated.")

