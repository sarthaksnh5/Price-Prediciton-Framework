# %%
import os, json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub

print("Packages imported & ready.")

# %%
KAGGLE_USERNAME = ""                 # can be blank if key-only tokens work
KAGGLE_KEY      = ""

kcfg_dir = os.path.abspath("./kaggle")        # local, user-space directory
os.makedirs(kcfg_dir, exist_ok=True)

with open(os.path.join(kcfg_dir, "kaggle.json"), "w") as f:
    json.dump({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}, f)

os.environ["KAGGLE_CONFIG_DIR"] = kcfg_dir    # tells the CLI where to look
print("kaggle.json written →", kcfg_dir)

# %%
import shutil

path = kagglehub.dataset_download("thedevastator/unlock-profits-with-e-commerce-sales-data")

print("Download finished. Contents of ./data:")
data_dir = os.path.join(os.getcwd(), "data")
print(path)

# Move the entire downloaded folder to ./data (rename/move as 'data')
if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
shutil.move(path, data_dir)

print("Folder moved and renamed to ./data")
print(os.listdir(data_dir))


# %%
df = pd.read_csv(os.path.join(data_dir, "Amazon Sale Report.csv"))

# %%
df.columns

# %%
# Drop unnamed or irrelevant columns
df.drop(columns=[
    'Unnamed: 22', 'promotion-ids', 'Courier Status', 'ship-city',
    'ship-postal-code', 'ship-country'
], inplace=True, errors='ignore')

# %%
# Normalize column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')

# %%
# Filter to only completed sales
shipped_df = df[(df['status'].str.lower() == 'shipped') | (df['status'].str.lower() == 'shipped - delivered to buyer')]

# %%
shipped_df

# %%
shipped_df['sku'].unique()

# %%
# Drop rows with missing or invalid sales
shipped_df.dropna(subset=['amount', 'qty', 'sku'], inplace=True)

# Remove rows with non-positive amount or quantity
shipped_df = shipped_df[(shipped_df['amount'] > 0) & (shipped_df['qty'] > 0)]

# %%
shipped_df['date'] = pd.to_datetime(shipped_df['date'], errors='coerce')
shipped_df['qty'] = pd.to_numeric(shipped_df['qty'], errors='coerce')
shipped_df['amount'] = pd.to_numeric(shipped_df['amount'], errors='coerce')

# Drop rows with invalid dates
shipped_df = shipped_df.dropna(subset=['date'])

# %%
shipped_df['unit_price'] = shipped_df['amount'] / shipped_df['qty']

# %%
shipped_df['revenue'] = shipped_df['amount']  # already available, but explicit naming helps clarity

# %%
shipped_df['year'] = shipped_df['date'].dt.year
shipped_df['month'] = shipped_df['date'].dt.month
shipped_df['day'] = shipped_df['date'].dt.day
shipped_df['weekday'] = shipped_df['date'].dt.day_name()
shipped_df['weekofyear'] = shipped_df['date'].dt.isocalendar().week


# %%
shipped_df['sales_channel'] = shipped_df['sales_channel'].str.strip().str.lower()
shipped_df['fulfilled_by'] = shipped_df['fulfilled_by'].str.strip().str.lower()

# %%
sku_agg = shipped_df.groupby('sku').agg({
    'unit_price': 'mean',
    'qty': 'sum',
    'revenue': 'sum',
    'style': 'first',
    'category': 'first'
}).reset_index()

# Rename for clarity
sku_agg.columns = ['sku', 'avg_price', 'total_qty', 'total_revenue', 'style', 'category']

# %%
sku_agg

# %% [markdown]
# ### Preliminary EDA Highlights (based on features)
# 
# We’ll analyze:
# * Price distribution (identify pricing tiers or outliers)
# * Demand vs price (scatter plots or binning)
# * Top categories/SKUs by revenue
# * Seasonality using time-based aggregates

# %%
# Price Distribution
sns.histplot(shipped_df['unit_price'], bins=50, kde=True)
plt.title("Unit Price Distribution")
plt.show()

# %% [markdown]
# ### Key Observations:
# 
# 1. **Right-skewed distribution**:
# 
#    * The unit prices are **concentrated between ₹300 and ₹1000**.
#    * This tells us that **most products are low to mid-range priced**.
#    * A long tail exists where a few products are priced as high as ₹2500.
# 
# 2. **Peak (mode) around ₹400–₹600**:
# 
#    * The **most frequent unit price range** is roughly in the ₹400–₹600 band.
#    * This is the pricing "sweet spot" where your product volume is highest.
# 
# 3. **Multiple peaks (multimodal)**:
# 
#    * There are **several smaller spikes** after ₹800, around ₹1000 and ₹1200.
#    * This may indicate **distinct product segments** or categories with different pricing strategies (e.g., basic, premium SKUs).
# 
# 4. **KDE line (blue curve)**:
# 
#    * This smoothed line (Kernel Density Estimate) helps visualize the **underlying probability density**.
#    * It confirms the **non-normal**, skewed nature of your price distribution.

# %%
shipped_df['sku'].value_counts()

# %%
sku="JNE3797-KR-L"
sku_data = shipped_df[shipped_df['sku'] == sku]

price_demand = sku_data.groupby('unit_price')['qty'].sum().reset_index()

# plot demand curve
sns.lineplot(x='unit_price', y='qty', data=price_demand, marker='o')
plt.title(f"Demand Curve for SKU: {sku}")
plt.xlabel("Unit Price")
plt.ylabel("Total Quantity Sold")
plt.grid()
plt.show()

# %% [markdown]
# ### 🔍 Key Observations:
# 
# 1. **Demand increases up to a certain price (\~₹735)**:
# 
#    * There is a **sharp rise in quantity sold as the price moves from ₹720 to ₹735**.
#    * This suggests the product may have been underpriced earlier, and customers responded more favorably at slightly higher prices (could be perceived value, offer period, or availability).
# 
# 2. **Demand drops after ₹735**:
# 
#    * After ₹735, total quantity sold **declines steadily**, showing typical **price elasticity**: as the price increases, fewer units are sold.
#    * This is the expected economic behavior.
# 
# 3. **Non-linear relationship**:
# 
#    * The relationship between price and demand is **not purely linear**.
#    * This implies that we might need to **fit a non-linear or log-log regression model** to capture the elasticity accurately.
# 
# ---
# 
# ### Business Insight:
# 
# * The **peak demand occurs at ₹735**, which could be close to the **optimal price point** for maximizing volume sales of this SKU.
# * Raising prices beyond this leads to **reduced demand**, so if your margin per unit isn’t high, revenue or profit may decline.
# * Lower prices (₹715–₹725) didn’t help sell significantly more units either, meaning **aggressive discounting may not be necessary**.
# 

# %%
# Group by unit price and aggregate total quantity sold
sku_price_demand = shipped_df.groupby(['sku', 'unit_price'])['qty'].sum().reset_index()

# %%
sku_price_demand['log_price'] = np.log(sku_price_demand['unit_price'])
sku_price_demand['log_qty'] = np.log(sku_price_demand['qty'])

# %%
from sklearn.linear_model import LinearRegression

elasticity_results = []

for sku_id, group in sku_price_demand.groupby('sku'):
    if group.shape[0] >= 3:  # minimum points for regression
        X = group[['log_price']]
        y = group['log_qty']
        model = LinearRegression()
        model.fit(X, y)
        elasticity = model.coef_[0]
        r2 = model.score(X, y)
        elasticity_results.append({'sku': sku_id, 'elasticity': elasticity, 'r2': r2})


# %%
elasticity_df = pd.DataFrame(elasticity_results)
elasticity_df.sort_values(by='elasticity', inplace=True)

# %%
elasticity_df['elasticity_range'] = pd.cut(
    elasticity_df['elasticity'],
    bins=[-np.inf, -1, 0, np.inf],
    labels=['Elastic', 'Inelastic', 'Positive Elasticity']
)

# %%
elasticity_df['elasticity_range'].value_counts()

# %% [markdown]
# | Elasticity Range                 | Interpretation                                                        | Example SKU                  |
# | -------------------------------- | --------------------------------------------------------------------- | ---------------------------- |
# | **Elastic** (< -1)               | Demand drops sharply with price ↑                                     | SAR062, J012-TP-XL           |
# | **Inelastic** (between 0 and -1) | Demand mildly sensitive                                               | MEN5027-KR-XXL,MEN5027-KR-XXL |
# | **Positive Elasticity**          | Counterintuitive – possible data error, bundling, or premium behavior | SET257-KR-PP-M (175.8)       |
# 
# ---
# 
# | Elasticity Range        | Count | % Share     | Interpretation                                                                                                |
# | ----------------------- | ----- | ----------- | ------------------------------------------------------------------------------------------------------------- |
# | **Elastic**             | 1306  | **\~45%**   | Majority of SKUs — lowering price will likely increase total revenue. Important for discounts or price drops. |
# | **Positive Elasticity** | 1237  | **\~42.6%** | Large chunk with unexpected behavior — needs inspection. Could be premium SKUs or data anomalies.             |
# | **Inelastic**           | 211   | **\~7.3%**  | Small subset — price changes have limited impact. Good candidates for price increase.                         |
# 

# %%
shipped_df['category'].value_counts()

# %%
# 1. Group by category and unit_price, summing quantities
cat_price_demand = shipped_df.groupby(['category', 'unit_price'])['qty'].sum().reset_index()

# 2. Filter out zero or negative values
cat_price_demand = cat_price_demand[(cat_price_demand['unit_price'] > 0) & (cat_price_demand['qty'] > 0)]

# %%
cat_price_demand['log_price'] = np.log(cat_price_demand['unit_price'])
cat_price_demand['log_qty']   = np.log(cat_price_demand['qty'])


# %%
results = []

for cat, group in cat_price_demand.groupby('category'):
    # Need at least 3 unique price points to fit a model
    if group['unit_price'].nunique() < 3:
        continue

    X = group[['log_price']]
    y = group['log_qty']

    model = LinearRegression()
    model.fit(X, y)

    elasticity = model.coef_[0]
    r2          = model.score(X, y)
    n_points    = group.shape[0]

    results.append({
        'category':       cat,
        'elasticity':     elasticity,
        'r_squared':      r2,
        'n_price_points': n_points
    })

category_elasticity_df = pd.DataFrame(results).sort_values('elasticity')

# %%
category_elasticity_df

# %%
category_elasticity_df['elasticity_range'] = pd.cut(
    category_elasticity_df['elasticity'],
    bins=[-np.inf, -1, 0, np.inf],
    labels=['Elastic', 'Inelastic', 'Positive Elasticity']
)

# %%
category_elasticity_df.value_counts('elasticity_range')

# %% [markdown]
# ## What This Tells Us
# 
# 1. **No “Elastic” Categories (< –1)**
# 
#    * At the **category level**, none of your product groups behaves with high price sensitivity.
#    * **Implication**: Broad price cuts are unlikely to drive large volume gains in any category.
# 
# 2. **Predominantly Inelastic Demand (5/8 Categories)**
# 
#    * Categories like **Saree, Kurta, Bottom, Set, Ethnic Dress** have elasticities between –1 and 0.
#    * **Interpretation**: Customers are relatively **insensitive** to small price changes here.
#    * **Strategy**: You could **raise prices** modestly in these categories to improve margins without sacrificing too much volume.
# 
# 3. **Counter‑Intuitive “Positive Elasticity” (3/8 Categories)**
# 
#    * **Top, Blouse, Western Dress** show a **positive** relationship: price ↑ → demand ↑.
#    * **Possible Explanations**:
# 
#      * **Premium signaling**: Higher price suggests higher quality, attracting more buyers.
#      * **Data artifacts**: Bundles, promotions, or inventory lulls at lower price points distorting the curve.
#    * **Action**: Investigate further—check for:
# 
#      * Promotional spikes at specific prices
#      * Low-stock periods that force higher-price sales
#      * Data anomalies (returns, refunds mis‑recorded)
# 
# 4. **Weak Model Fit (Low R² Values)**
# 
#    * Except for Saree (R² ≈ 0.05), all R² values are very low (< 0.05).
#    * **Meaning**: Price alone explains only a tiny fraction of demand variation at the category level.
#    * **Next Steps**: Incorporate additional features (seasonality, promotions, channel mix) or switch to **non‑linear**/segmented models.
# 
# ---
# 
# ## Strategic Takeaways
# 
# | Focus Area               | Recommended Next Steps                                                                             |
# | ------------------------ | -------------------------------------------------------------------------------------------------- |
# | **Inelastic Categories** | Test **small price increases** (e.g. +5–10%) and monitor volume impact. Focus on margin expansion. |
# | **Positive Elasticity**  | Audit data and business context. Consider **premium positioning** or adjust for anomalies.         |
# | **Model Improvement**    | Add features (e.g. seasonal dummies, channel flags) or try **piecewise / polynomial** fits.        |
# 

# %% [markdown]
# Starting Price Simulation

# %%
baseline = (
    shipped_df
    .groupby('category')
    .agg(
        baseline_qty   = ('qty', 'sum'),
        baseline_rev   = ('unit_price', lambda s: (s * shipped_df.loc[s.index, 'qty']).sum())
    )
    .reset_index()
)

# %%
# Merge elasticity into baseline
baseline = baseline.merge(category_elasticity_df[['category','elasticity']], on='category')

# %%
# 3. Define price adjustment multipliers (−10% to +10% in 5% steps)
multipliers = np.linspace(0.90, 1.10, 5)

# %%

# 4. Build a scenario table
rows = []
for _, row in baseline.iterrows():
    cat, Q0, R0, E = row['category'], row['baseline_qty'], row['baseline_rev'], row['elasticity']
    for m in multipliers:
        P_mult = m
        # Forecast new quantity: Q1 = Q0 * (P_mult)^E
        Q1 = Q0 * (P_mult ** E)
        # Forecast new revenue: R1 = R0 * P_mult * (P_mult^E)
        R1 = R0 * P_mult * (P_mult ** E)
        rows.append({
            'category':          cat,
            'price_change_pct':  (m - 1)*100,
            'baseline_revenue':  R0,
            'simulated_revenue': R1,
            'rev_delta_pct':     (R1 - R0) / R0 * 100
        })

# %%

scenario_df = pd.DataFrame(rows)

# %%

# 5. Pivot for readability (categories as rows, price_change_pct as columns)
pivot = scenario_df.pivot_table(
    index='category',
    columns='price_change_pct',
    values='rev_delta_pct'
).round(1)

# %%

print("Revenue % Change vs Baseline (by Category):")
print(pivot)

# %% [markdown]
# ## Key Takeaways
# 
# 1. **Revenue Gains from Price Increases**
# 
#    * **Western Dress** sees the largest upside:
# 
#      * +5% price → +11.1% revenue
#      * +10% price → +22.9% revenue
#    * **Blouse** and **Top** also benefit significantly from moderate price hikes:
# 
#      * Blouse: +10% price → +18.3% revenue
#      * Top:   +10% price → +13.2% revenue
# 
# 2. **Limited Impact of Price Cuts**
# 
#    * Across all categories, **cutting prices reduces revenue**.
#    * Even the “elastic” categories (Blouse, Top, Western Dress) don’t recoup enough volume to offset the lower price.
#    * **Conclusion**: Across the board, **price increases** are more beneficial than discounts.
# 
# 3. **Near Unitary Category (Saree)**
# 
#    * **Saree** shows essentially **0% change** under ±10% moves.
#    * Elasticity ≈ –1 → **unit elastic**:
# 
#      * Price hikes or cuts leave revenue roughly unchanged.
# 
# 4. **Mild Sensitivity Categories**
# 
#    * **Kurta** and **Bottom**:
# 
#      * +10% price → +3.9% (Kurta) and +6.1% (Bottom) revenue gain
#    * **Ethnic Dress** and **Set**:
# 
#      * +10% price → \~+9–8% revenue gain
# 
# ---
# 
# ## Strategic Recommendations
# 
# 1. **Implement Price Increases**
# 
#    * **Western Dress**, **Blouse**, **Top**: Raise prices by +5–10% to boost revenue.
# 2. **Marginal Adjustments for Mid‑Sensitivity SKUs**
# 
#    * **Ethnic Dress**, **Bottom**, **Kurta**, **Set**: Consider modest price hikes (+5%) for incremental revenue.
# 3. **Maintain Current Pricing on Sarees**
# 
#    * Since revenue is flat, focus on other levers (bundles, promotions) rather than price.
# 4. **Avoid Across‑the‑Board Discounts**
# 
#    * Even “elastic” segments didn’t generate enough volume to justify price cuts.
# 

# %%
def most_common_or_nan(series):
    """Return the mode of the non-null values in series, or np.nan if none exist."""
    vals = series.dropna()
    return vals.mode().iat[0] if not vals.mode().empty else np.nan

sku_meta = (
    shipped_df
      .groupby('sku')
      .agg(
          sales_channel = ('sales_channel', most_common_or_nan),
          fulfilled_by  = ('fulfilled_by',  most_common_or_nan),
          b2b_flag      = ('b2b',           most_common_or_nan),
      )
      .reset_index()
)

# %%
seg_df = elasticity_df.merge(sku_meta, on='sku', how='left')
seg_df = seg_df.dropna(subset=['sales_channel','fulfilled_by','b2b_flag'])

# %%
# choose the dimensions to segment by
segment_cols = ['sales_channel', 'fulfilled_by', 'b2b_flag']

# for each dimension, get mean/median/count of elasticity
segment_stats = {}
for col in segment_cols:
    stats = (
        seg_df
        .groupby(col)['elasticity']
        .agg(['count', 'mean', 'median'])
        .rename(columns={'count':'n_skus','mean':'avg_elasticity','median':'med_elasticity'})
        .sort_values('avg_elasticity')
    )
    segment_stats[col] = stats


# %%
for col, stats in segment_stats.items():
    print(f"\n=== Segment: {col} ===")
    print(stats.round(2))


# %% [markdown]
# Since every SKU in your sample falls under the **same** metadata values, the segmentation returned only one group for each dimension:
# 
# * **Sales Channel**: all SKUs are sold via `amazon.in`
# * **Fulfillment**: all are fulfilled by `easy ship`
# * **B2B Flag**: all are retail (`False`)
# 
# ---
# 
# ## What This Means
# 
# 1. **No Variability in These Dimensions**
# 
#    * Because our dataset (or the subset you ran this on) only contains a single sales channel, fulfillment type, and B2B flag, you can’t learn anything about how elasticity varies across these segments—they’re all the same.
# 
# 2. **Your Segment‑Level Averages Mirror the Full Catalog**
# 
#    * `avg_elasticity = 1.49` and `med_elasticity = –0.77` simply restate your overall SKU elasticity distribution (skewed positive mean, negative median) for the entire set.

# %% [markdown]
# Step 5

# %%
baseline = (
    shipped_df
    .assign(revenue = lambda df: df['unit_price'] * df['qty'])
    .groupby('category')
    .agg(
        baseline_qty = ('qty', 'sum'),
        baseline_rev = ('revenue', 'sum')
    )
    .reset_index()
)

# %%
# Merge in the elasticity coefficients we computed earlier
baseline = baseline.merge(category_elasticity_df[['category','elasticity']], on='category')

# %%
# ─── 2. Define Scenario Parameters ───────────────────────────────────────────
disc_multipliers = [0.90, 0.80]     # –10% and –20% discount
peak_uplifts     = [1.10, 1.20]     # +10% and +20% baseline demand uplift

# %%
# ─── 3. Promotion Scenarios (Price Discount) ─────────────────────────────────
promo_rows = []
for _, row in baseline.iterrows():
    cat, Q0, R0, E = row['category'], row['baseline_qty'], row['baseline_rev'], row['elasticity']
    for m in disc_multipliers:
        # Price is discounted → unit_price * m
        # New quantity: Q1 = Q0 * (m ** E)
        Q1 = Q0 * (m ** E)
        # New revenue: R1 = (R0 / Q0 * m) * Q1  ==> baseline unit price * m * Q1
        R1 = (R0 / Q0) * m * Q1
        promo_rows.append({
            'category':         cat,
            'scenario':         f'{int((m-1)*100)}% discount',
            'baseline_rev':     R0,
            'sim_rev':          R1,
            'rev_change_pct':   (R1 - R0) / R0 * 100
        })

promo_df = pd.DataFrame(promo_rows)

# %%
# ─── 4. Peak‑Demand Scenarios (Volume Uplift) ─────────────────────────────────
peak_rows = []
for _, row in baseline.iterrows():
    cat, Q0, R0 = row['category'], row['baseline_qty'], row['baseline_rev']
    for u in peak_uplifts:
        # Assume price stays same, but quantity increases by uplift factor
        Q1 = Q0 * u
        R1 = R0 * u
        peak_rows.append({
            'category':       cat,
            'scenario':       f'{int((u-1)*100)}% demand uplift',
            'baseline_rev':   R0,
            'sim_rev':        R1,
            'rev_change_pct': (R1 - R0) / R0 * 100
        })

# %%
peak_df = pd.DataFrame(peak_rows)

# %%
evaluation_df = pd.concat([promo_df, peak_df], ignore_index=True)

# %%
pivot = evaluation_df.pivot_table(
    index='category',
    columns='scenario',
    values='rev_change_pct'
).round(1)

# %%
pivot

# %%
# pivot from your last step: pivot (categories × scenarios) of rev_change_pct
plt.figure(figsize=(8, 6))
sns.heatmap(
    pivot, 
    annot=True, 
    fmt=".1f", 
    cmap="RdYlGn", 
    center=0,
    linewidths=0.5
)
plt.title("Revenue % Change vs Baseline (Categories × Scenarios)")
plt.xlabel("Scenario")
plt.ylabel("Category")
plt.tight_layout()
plt.show()

# %% [markdown]
# 


