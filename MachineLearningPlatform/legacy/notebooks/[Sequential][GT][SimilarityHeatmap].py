import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# 1. Load Data
INPUT_FILE = "/Users/xlu/Workspace/code/tools/data/rtb_connection_similarity.csv"
print(f"Loading similarity data from {INPUT_FILE}...")

# Assuming CSV has columns like: rtb_id_A, rtb_id_B, similarity
# Adjust column names if necessary based on CSV content.
# Spark usually outputs weird directory structures or parts.
# If it's a single CSV file, read it directly.
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows.")
    # Standardize column names if needed
    # Expecting: rtb_id_A, rtb_id_B, similarity
    # If spark saved it, it might have differnet names.
    # Let's inspect first few columns or assume standard
    if 'rtb_id_A' not in df.columns:
        # Fallback for common Spark export names if needed, or assume first 3 cols
        df.columns = ['rtb_id_A', 'rtb_id_B', 'similarity']
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# 2. Build Full Matrix
print("Building similarity matrix...")
# Get unique IDs
rtb_ids = sorted(list(set(df['rtb_id_A'].unique()) | set(df['rtb_id_B'].unique())))
n = len(rtb_ids)
id_to_idx = {rid: i for i, rid in enumerate(rtb_ids)}

matrix = np.zeros((n, n))

# Fill matrix
for _, row in df.iterrows():
    i = id_to_idx[row['rtb_id_A']]
    j = id_to_idx[row['rtb_id_B']]
    sim = row['similarity']
    matrix[i, j] = sim
    matrix[j, i] = sim # Ensure symmetry

# Fill diagonal with 1.0 (Self-similarity)
np.fill_diagonal(matrix, 1.0)

print(f"Matrix shape: {matrix.shape}")

# 3. Hierarchical Clustering for Sorting (The 'Professional' Way)
# We convert similarity to distance (1 - similarity)
print("Performing hierarchical clustering to reorder matrix...")
distance_matrix = 1.0 - matrix
# Ensure distance matrix is symmetric and has zeros on diagonal (floating point errors)
np.fill_diagonal(distance_matrix, 0.0)
distance_matrix = np.clip(distance_matrix, 0, 1)

# Condensed distance matrix for linkage
condensed_dist = squareform(distance_matrix)

# Compute linkage (Ward's method is usually good for dense-ish clusters)
# Optimal leaf ordering can be slow for N>1000, but fine for N~500
Z = linkage(condensed_dist, method='ward')

# Get sorted indices
optimal_ordering_indices = leaves_list(Z)

# Reorder matrix and labels
sorted_rtb_ids = [rtb_ids[i] for i in optimal_ordering_indices]
sorted_matrix = matrix[optimal_ordering_indices, :][:, optimal_ordering_indices]

# 4. Plotly Heatmap
print("Generating Heatmap...")

# Create Hover Text
hover_text = []
for i in range(n):
    hover_row = []
    for j in range(n):
        hover_row.append(
            f"Row: {sorted_rtb_ids[i]}<br>"
            f"Col: {sorted_rtb_ids[j]}<br>"
            f"Similarity: {sorted_matrix[i][j]:.4f}"
        )
    hover_text.append(hover_row)

fig = go.Figure(data=go.Heatmap(
    z=sorted_matrix,
    x=sorted_rtb_ids,
    y=sorted_rtb_ids,
    text=hover_text,
    hoverinfo='text',
    colorscale='Viridis', # 'Magma', 'Plasma', 'Viridis' are good professional scales
    zmin=0.0,
    zmax=1.0,
))

fig.update_layout(
    title=f"RTB Connection Similarity Heatmap (Clustered Ordering - {n} IDs)",
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
        autorange='reversed', # Heatmap convention: (0,0) at top-left
        title="RTB ID"
    )
)

fig.show()
print("✅ Heatmap generated.")

