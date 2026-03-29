import pandas as pd
import plotly.graph_objects as go
from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.getOrCreate()

# Databricks display function - fallback to show() for local execution
try:
    from databricks.sdk.runtime import display
except ImportError:
    display = lambda df: df.show()

# 1. Configuration
CAPS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
GT_BASE_PATH = "s3://exchange-dev/luxu/cap_navigation_gt"
MODEL_BASE_PATH = "s3://exchange-dev/luxu/cap_navigation"
OVERFIT_BASE_PATH = "s3://exchange-dev/luxu/cap_navigation_overfit"
REG_BASE_PATH = "s3://exchange-dev/luxu/cap_navigation_reg"

# 2. Data Loading Function
def load_and_process_data(base_path, source_name):
    all_data = []
    for cap in CAPS:
        try:
            if cap == 1.0:
                # Handle baseline (cap=1.0)
                # We need to infer rtb_account_ids. We can read from cap=1.1 to get IDs
                # and fill 0s, or just skip and let outer join handle it if we read others.
                # Better strategy: Read cap=1.1 to get schema and IDs, then set metrics to 0.
                # Assuming 1.1 exists.
                temp_path = f"{base_path}/1.1"
                df = spark.read.parquet(temp_path).select("rtb_account_id").distinct()
                df = df.withColumn("unr_uplift", F.lit(0.0))
            else:
                path = f"{base_path}/{cap}"
                # Check if path exists logic usually requires hadoop fs check, 
                # but spark.read might fail if not exists.
                # We assume paths exist as per user description.
                df = spark.read.parquet(path).select("rtb_account_id", "unr_uplift")
            
            # Convert to Pandas for plotting (data size is small: accounts * caps)
            pdf = df.toPandas()
            pdf['cap'] = cap
            pdf['source'] = source_name
            all_data.append(pdf)
            print(f"✅ Loaded {source_name} data for cap={cap}")
        except Exception as e:
            print(f"⚠️ Failed to load {source_name} data for cap={cap}: {e}")
            
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)

# 3. Load Data
print("Loading Ground Truth Data...")
gt_df = load_and_process_data(GT_BASE_PATH, "Ground Truth")

print("Loading Model Evaluation Data...")
model_df = load_and_process_data(MODEL_BASE_PATH, "Model")

print("Loading Overfit Evaluation Data...")
overfit_df = load_and_process_data(OVERFIT_BASE_PATH, "Overfit Model")

print("Loading Regularized Evaluation Data...")
reg_df = load_and_process_data(REG_BASE_PATH, "Regularized Model")

# 4. Consolidate
full_df = pd.concat([gt_df, model_df, overfit_df, reg_df], ignore_index=True)

# 5. Plotting
# Get unique accounts
accounts = full_df['rtb_account_id'].unique()
print(f"Generating plots for {len(accounts)} RTB Accounts...")

# We will generate one plot per account. 
# If there are too many, we might want to limit or aggregate. 
# For now, let's display Top 20 by variance or just first 20 to avoid browser crash.
# Or better, use a dropdown in Plotly (might be heavy).
# Let's display separate simple plots for a few accounts as examples.
accounts = [
  "5fffe796e4fa2100156e364a",
  "645a4ac71263430014fc0a33",
  "602442ef9f0893001526406b",
  "5caf77f1e04ca66a2d4bcd7c",
  "5fbc06efdcdfdb0015dda411",
  "618a29d9ef3f360017e9acca",
  "5e89fcef428f870016d9e994",
  "65b7e5c3d5f847001169024b",
  "6135c233b2f85d001747c3d2",
  "5e56893e6c71d40018de3ad1",
  "61e94b2bc6dbdc0017ea3870",
  "62d4edd34b2125001b0a042a",
  "65660afecebd4a0011dcc2ac",
  "6526e0feb1392d001124df7d",
  "5f06518e8421a500152881bb",
  "5e150c8b5fd7df7b642140df",
  "63d743303639240014ba0dd0",
  "61c23cfa1958de001706a194",
  "60770b778d0e4a0015f20ea9",
  "61936e419aca75001740e38a",
  "561e8d936b8d90f61a000d2f",
  "615ff7aacf8ca30017d834ea",
  "5fd73c61617f68000fb6584f",
  "60c86e9b217f2400175cfe6b",
  "60c9b375fb988e0017d6e500",
  "685d0956892f9d0011d79cb7",
  "64c165a37c1dd00011cb65f1",
  "644fff26ff6cc5001440892f",
  "624a723e1bad4b0011a0cfd7",
  "63d743e53639240014ba0ddd",
  "6391345e690bb000176c1058",
  "5f3dc36a8421a5001528b784",
  "60ebaee89459b10017db3907",
  "61dba28ceabd16001728ae11",
  "688a2196206ad6dea34bcf23",
  "66cde67e3db5f70011e66c74",
  "5f8f4996f2a95d39439fccf1",
  "6384f4ef2c8690001430be0d",
  "67ef127dc1727d00118d86e2",
  "62451df8df1c7b0011064d0d",
  "68ec715094a2ed8b65447e7a",
  "62563a74f5c7a20011b32398",
  "631c290e3f7bdc001bdfb779",
  "67cffc9a23012b0011aed182",
  "6177a5dce899b2001727f1e8",
  "649482d5cec8d700113d2881",
  "63217c5d436480001cd32ae4",
  "68372d339886e6001161c9c1",
  "617f985e4f29ac001736e2f4",
  "68f8632d386a23dd1d615bf1",
  "682accb99886e60011615293",
  "6168e52a9008de00174bc182",
  "5c4f514dbfa483080c46675f",
  "5fdbcb6754719a0015122725",
  "60d191906f59f30017a17639",
  "671bd87469f356001105653b",
  "5e20bbdf55c1c23edbbca5fa",
  "66ed7eacad09bd00115d1340",
  "6242aaf6f98c6b001865107f",
  "634f8edc0617c5001b20f9a7",
  "6323ac240ece9f001bef1aa4",
  "64371dd35122b200141e0019",
  "64f77eca27f7020011a853b1",
  "671a9d1a69f3560011055966",
  "6483b361b1f1e40011f762d6",
  "5bede32c00cd071fc5770fcf",
  "62f47a393a874f001b0a20c6",
  "5b9827f6abdbc240d1d45071",
  "6163ee8d2e2c3200179604b9",
  "6358477435d47b0014f36d9b",
  "60f86861885bd3001740f2fe",
  "68a8024afa61a7bb4aac8df6",
  "60d2bd9281c6120017f178fb",
  "638f936a7097c0001a936119",
  "64f8922427f7020011a867d1",
  "67f579a965660600111030c2",
  "59791c2469736e2d62002c8f",
  "62cf10126b748c001b9ca6de",
  "690abe83eb3bf4c11bf9175c",
  "64d1e8f87c1dd00011cc3de2",
  "5a8371eaeadf6c6135002396",
  "5bff10ad6b66b21fde43b6d7",
  "6761ddf61d62100011606978",
  "60d6158081c6120017f17b15",
  "67926a1377bd1c00117a7b54",
  "655e228e5942160011cb6f5e",
  "63212ea1436480001cd3285e",
  "602442a043209100157f4392",
  "64c83e58bedba60011a414cf",
  "682648ce4c15ca0011ac65c5",
  "5a0b03dc33675100118bafad",
  "5cd92b2661a35300113a8487",
  "5df14535bd53a72fd32480e6",
  "6716f3d175b32c0011dcb46d"
]

for account in accounts:
    if not account in full_df['rtb_account_id'].unique():
        continue

    account_data = full_df[full_df['rtb_account_id'] == account]
    
    gt_data = account_data[account_data['source'] == "Ground Truth"].sort_values('cap')
    model_data = account_data[account_data['source'] == "Model"].sort_values('cap')
    overfit_data = account_data[account_data['source'] == "Overfit Model"].sort_values('cap')
    reg_data = account_data[account_data['source'] == "Regularized Model"].sort_values('cap')
    
    fig = go.Figure()
    
    # GT Line
    fig.add_trace(go.Scatter(
        x=gt_data['cap'], 
        y=gt_data['unr_uplift'],
        mode='lines+markers',
        name='Ground Truth',
        line=dict(color='blue')
    ))
    
    # Model Line
    fig.add_trace(go.Scatter(
        x=model_data['cap'], 
        y=model_data['unr_uplift'],
        mode='lines+markers',
        name='Model',
        line=dict(color='red', dash='dash')
    ))

    # Overfit Model Line
    fig.add_trace(go.Scatter(
        x=overfit_data['cap'], 
        y=overfit_data['unr_uplift'],
        mode='lines+markers',
        name='Overfit Model',
        line=dict(color='green', dash='dot')
    ))

    # Regularized Model Line
    fig.add_trace(go.Scatter(
        x=reg_data['cap'], 
        y=reg_data['unr_uplift'],
        mode='lines+markers',
        name='Regularized Model',
        line=dict(color='purple', dash='dashdot')
    ))
    
    fig.update_layout(
        title=f"NR Uplift vs Cap - Account: {account}",
        xaxis_title="QPS Cap",
        yaxis_title="NR Uplift",
        template="plotly_white",
        width=800,
        height=500
    )
    
    fig.show()

print("✅ Done.")
