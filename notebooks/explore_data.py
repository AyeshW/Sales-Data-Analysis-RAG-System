import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

os.makedirs('plots', exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.titlesize': 14,
                     'axes.labelsize': 12, 'figure.dpi': 130})

BLUE = "#2E75B6"
GREEN = "#70AD47"
ORANGE = "#ED7D31"
RED = "#FF4C4C"
PURPLE = "#7030A0"
COLORS = [BLUE, GREEN, ORANGE, RED, PURPLE, "#FFC000", "#00B0F0", "#92D050"]


df = pd.read_csv('../data/Sample - Superstore.csv', encoding='latin-1')
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])
df['Order Year'] = df['Order Date'].dt.year
df['Order Month'] = df['Order Date'].dt.month
df['Order Month Name'] = df['Order Date'].dt.strftime('%b')
df['Order Quarter'] = df['Order Date'].dt.quarter
for col in ['Category', 'Sub-Category', 'Region', 'Segment', 'Ship Mode']:
    df[col] = df[col].str.strip().str.title()
df['Profit Margin'] = df.apply(
    lambda r: round(r['Profit'] / r['Sales'], 4) if r['Sales'] != 0 else 0, axis=1
)
df['Season'] = df['Order Month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall',  10: 'Fall', 11: 'Fall'
})


print("\n" + "="*55)
print("  DATASET OVERVIEW")
print("="*55)
print(f"  Total rows       : {len(df):,}")
print(f"  Unique orders    : {df['Order ID'].nunique():,}")
print(f"  Unique customers : {df['Customer ID'].nunique():,}")
print(f"  Unique products  : {df['Product ID'].nunique():,}")
print(
    f"  Date range       : {df['Order Date'].min().date()} → {df['Order Date'].max().date()}")
print(f"  Missing values   : {df.isnull().sum().sum()}")
print(f"  Duplicate rows   : {df.duplicated().sum()}")
print(f"\n  Categories  : {list(df['Category'].unique())}")
print(f"  Regions     : {list(df['Region'].unique())}")
print(f"  Segments    : {list(df['Segment'].unique())}")
print(f"  Ship Modes  : {list(df['Ship Mode'].unique())}")
print(
    f"  Sub-categories ({df['Sub-Category'].nunique()}): {list(df['Sub-Category'].unique())}")
print("\n  Numerical summary:")
print(df[['Sales', 'Profit', 'Quantity', 'Discount',
      'Profit Margin']].describe().round(2).to_string())
print("="*55)

print("\n Plot 1: Data type overview...")

dtypes_info = {
    'Numerical':    ['Sales', 'Profit', 'Quantity', 'Discount', 'Profit Margin'],
    'Date':         ['Order Date', 'Ship Date'],
    'Categorical':  ['Category', 'Sub-Category', 'Region', 'Segment', 'Ship Mode'],
    'Identifier':   ['Order ID', 'Customer ID', 'Product ID', 'Row ID'],
    'Text':         ['Customer Name', 'Product Name', 'City', 'State', 'Country'],
}

fig, ax = plt.subplots(figsize=(11, 5))
ax.axis('off')

col_colors = [BLUE, "#2196F3", GREEN, ORANGE, RED]
y_start = 0.92
row_h = 0.13

for i, (dtype, cols) in enumerate(dtypes_info.items()):
    color = col_colors[i]
    ax.add_patch(plt.Rectangle((0.01, y_start - i*row_h - 0.09),
                               0.18, 0.10, color=color, transform=ax.transAxes, zorder=2))
    ax.text(0.10, y_start - i*row_h - 0.04, dtype,
            transform=ax.transAxes, ha='center', va='center',
            fontsize=11, fontweight='bold', color='white', zorder=3)
    col_text = ",   ".join(cols)
    ax.text(0.22, y_start - i*row_h - 0.04, col_text,
            transform=ax.transAxes, ha='left', va='center',
            fontsize=10, color='#333333')
    ax.text(0.97, y_start - i*row_h - 0.04, f"{len(cols)} cols",
            transform=ax.transAxes, ha='right', va='center',
            fontsize=9, color=color, fontweight='bold')

ax.set_title("Dataset Column Types — What Data is Available", fontsize=14,
             fontweight='bold', color='#1F4E79', pad=18)
plt.tight_layout()
plt.savefig('plots/01_data_type_overview.png', bbox_inches='tight')
plt.close()


print("Plot 2: Numerical distributions...")

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
num_cols = ['Sales', 'Profit', 'Quantity', 'Discount', 'Profit Margin']
col_colors_num = [BLUE, GREEN, ORANGE, PURPLE, RED]

for ax, col, color in zip(axes, num_cols, col_colors_num):
    data = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99))
    ax.hist(data, bins=35, color=color, edgecolor='white', alpha=0.85)
    ax.set_title(col, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    mean_val = df[col].mean()
    ax.axvline(mean_val, color='black', linestyle='--',
               linewidth=1.2, label=f'Mean: {mean_val:.1f}')
    ax.legend(fontsize=8)

fig.suptitle("Distribution of Numerical Columns", fontsize=14,
             fontweight='bold', color='#1F4E79', y=1.02)
plt.tight_layout()
plt.savefig('plots/02_numerical_distributions.png', bbox_inches='tight')
plt.close()
print("     Saved: plots/02_numerical_distributions.png")

print("  Plot 3: Categorical value counts...")

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
cat_cols = ['Category', 'Region', 'Segment', 'Ship Mode']

for ax, col, color in zip(axes, cat_cols, [BLUE, GREEN, ORANGE, PURPLE]):
    counts = df[col].value_counts()
    bars = ax.barh(counts.index, counts.values, color=color,
                   edgecolor='white', alpha=0.85)
    ax.set_title(col, fontweight='bold')
    ax.set_xlabel('Number of Rows')
    for bar, val in zip(bars, counts.values):
        ax.text(val + 30, bar.get_y() + bar.get_height()/2,
                f'{val:,}', va='center', fontsize=9)

fig.suptitle("Categorical Columns — How Many Rows Per Value",
             fontsize=14, fontweight='bold', color='#1F4E79', y=1.02)
plt.tight_layout()
plt.savefig('plots/03_categorical_counts.png', bbox_inches='tight')
plt.close()
print("     Saved: plots/03_categorical_counts.png")

print("  Plot 4: Sales and profit by category...")

cat_summary = df.groupby('Category').agg(
    Total_Sales=('Sales', 'sum'),
    Total_Profit=('Profit', 'sum')
).reset_index()

x = np.arange(len(cat_summary))
w = 0.38
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w/2, cat_summary['Total_Sales'],
            w, label='Total Sales',  color=BLUE,  alpha=0.9)
b2 = ax.bar(x + w/2, cat_summary['Total_Profit'],
            w, label='Total Profit', color=GREEN, alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(cat_summary['Category'], fontsize=11)
ax.set_ylabel('Amount ($)')
ax.set_title('Total Sales vs Profit by Category',
             fontweight='bold', color='#1F4E79')
ax.legend()
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f'${v/1000:.0f}K'))

for bar in b1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2000,
            f'${bar.get_height()/1000:.0f}K', ha='center', fontsize=9, color=BLUE)
for bar in b2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2000,
            f'${bar.get_height()/1000:.0f}K', ha='center', fontsize=9, color='#3a7d22')

plt.tight_layout()
plt.savefig('plots/04_sales_profit_by_category.png', bbox_inches='tight')
plt.close()
print("     Saved: plots/04_sales_profit_by_category.png")

print("  Plot 5: Profit by sub-category...")

subcat_profit = df.groupby('Sub-Category')['Profit'].sum().sort_values()
colors_bar = [RED if v < 0 else GREEN for v in subcat_profit.values]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(subcat_profit.index, subcat_profit.values,
               color=colors_bar, edgecolor='white', alpha=0.88)
ax.axvline(0, color='black', linewidth=0.9)
ax.set_xlabel('Total Profit ($)')
ax.set_title('Total Profit by Sub-Category\n(Red = Loss, Green = Profit)',
             fontweight='bold', color='#1F4E79')
ax.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f'${v/1000:.0f}K'))

for bar, val in zip(bars, subcat_profit.values):
    offset = 300 if val >= 0 else -300
    ax.text(val + offset, bar.get_y() + bar.get_height()/2,
            f'${val/1000:.1f}K', va='center', fontsize=8.5,
            color='#1a5c1a' if val >= 0 else '#9e1c1c')

plt.tight_layout()
plt.savefig('plots/05_profit_by_subcategory.png', bbox_inches='tight')
plt.close()
print("     Saved: plots/05_profit_by_subcategory.png")

print("  Plot 6: Monthly sales trend by year...")

monthly = df.groupby(['Order Year', 'Order Month'])[
    'Sales'].sum().reset_index()
year_colors = {2014: BLUE, 2015: GREEN, 2016: ORANGE, 2017: RED}

fig, ax = plt.subplots(figsize=(13, 5))
for year, grp in monthly.groupby('Order Year'):
    ax.plot(grp['Order Month'], grp['Sales'], marker='o', linewidth=2.2,
            markersize=5, label=str(year), color=year_colors[year])

ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.set_ylabel('Total Sales ($)')
ax.set_title('Monthly Sales Trend — 2014 to 2017',
             fontweight='bold', color='#1F4E79')
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f'${v/1000:.0f}K'))
ax.legend(title='Year')
plt.tight_layout()
plt.savefig('plots/06_monthly_sales_trend.png', bbox_inches='tight')
plt.close()
print("     Saved: plots/06_monthly_sales_trend.png")

print("  Plot 7: Yearly sales growth...")

yearly = df.groupby('Order Year').agg(
    Total_Sales=('Sales', 'sum'),
    Order_Count=('Order ID', 'nunique')
).reset_index()

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

bars = ax1.bar(yearly['Order Year'], yearly['Total_Sales'],
               color=BLUE, alpha=0.85, label='Total Sales')
ax2.plot(yearly['Order Year'], yearly['Order_Count'], color=ORANGE, marker='o',
         linewidth=2.5, markersize=8, label='Unique Orders')

ax1.set_xlabel('Year')
ax1.set_ylabel('Total Sales ($)', color=BLUE)
ax2.set_ylabel('Unique Orders', color=ORANGE)
ax1.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f'${v/1000:.0f}K'))
ax1.set_title('Yearly Sales Growth & Order Count',
              fontweight='bold', color='#1F4E79')

for bar, val in zip(bars, yearly['Total_Sales']):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5000,
             f'${val/1000:.0f}K', ha='center', fontsize=9, color=BLUE, fontweight='bold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.tight_layout()
plt.savefig('plots/07_yearly_growth.png', bbox_inches='tight')
plt.close()
print("     Saved: plots/07_yearly_growth.png")

print("  Plot 8: Sales and profit by region...")

reg = df.groupby('Region').agg(
    Total_Sales=('Sales', 'sum'),
    Total_Profit=('Profit', 'sum')
).reset_index().sort_values('Total_Sales', ascending=False)

x = np.arange(len(reg))
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - 0.2, reg['Total_Sales'],  0.38,
            label='Sales',  color=BLUE,  alpha=0.9)
b2 = ax.bar(x + 0.2, reg['Total_Profit'], 0.38,
            label='Profit', color=GREEN, alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(reg['Region'], fontsize=11)
ax.set_ylabel('Amount ($)')
ax.set_title('Total Sales vs Profit by Region',
             fontweight='bold', color='#1F4E79')
ax.legend()
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f'${v/1000:.0f}K'))
plt.tight_layout()
plt.savefig('plots/08_sales_profit_by_region.png', bbox_inches='tight')
plt.close()
print("     Saved: plots/08_sales_profit_by_region.png")

print("  Plot 9: Discount vs profit scatter...")

sample = df.sample(1500, random_state=42)
cat_color_map = {'Furniture': BLUE,
                 'Office Supplies': GREEN, 'Technology': ORANGE}

fig, ax = plt.subplots(figsize=(10, 5))
for cat, grp in sample.groupby('Category'):
    ax.scatter(grp['Discount'], grp['Profit'], alpha=0.45, s=25,
               color=cat_color_map[cat], label=cat)

ax.axhline(0, color='red', linewidth=1, linestyle='--', label='Break-even')
ax.set_xlabel('Discount Rate')
ax.set_ylabel('Profit ($)')
ax.set_title('Discount vs Profit by Category\n(shows how discounts affect profitability)',
             fontweight='bold', color='#1F4E79')
ax.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f'{v*100:.0f}%'))
ax.legend()
plt.tight_layout()
plt.savefig('plots/09_discount_vs_profit.png', bbox_inches='tight')
plt.close()
print("     Saved: plots/09_discount_vs_profit.png")

print("  Plot 10: Seasonality pattern...")

season_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May',
                'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_avg = df.groupby('Order Month')['Sales'].mean().reset_index()
monthly_avg['Month Name'] = monthly_avg['Order Month'].apply(
    lambda m: season_order[m-1])

season_color_map = {
    'Jan': '#74B9FF', 'Feb': '#74B9FF', 'Mar': '#55EFC4',
    'Apr': '#55EFC4', 'May': '#55EFC4', 'Jun': '#FDCB6E',
    'Jul': '#FDCB6E', 'Aug': '#FDCB6E', 'Sep': '#E17055',
    'Oct': '#E17055', 'Nov': '#E17055', 'Dec': '#74B9FF'
}
bar_colors = [season_color_map[m] for m in monthly_avg['Month Name']]

fig, ax = plt.subplots(figsize=(13, 5))
bars = ax.bar(monthly_avg['Month Name'], monthly_avg['Sales'],
              color=bar_colors, edgecolor='white', alpha=0.9)
ax.set_ylabel('Average Sales per Transaction ($)')
ax.set_title('Seasonality — Average Sales by Month (all years combined)\n'
             'Winter=Blue  Spring=Green  Summer=Yellow  Fall=Orange',
             fontweight='bold', color='#1F4E79')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:.0f}'))

for bar, val in zip(bars, monthly_avg['Sales']):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f'${val:.0f}', ha='center', fontsize=8, rotation=45)

plt.tight_layout()
plt.savefig('plots/10_seasonality_monthly_avg.png', bbox_inches='tight')
plt.close()
print("     Saved: plots/10_seasonality_monthly_avg.png")

print("  Plot 11: Segment breakdown...")

seg = df.groupby('Segment').agg(
    Orders=('Order ID', 'nunique'),
    Sales=('Sales', 'sum'),
    Profit=('Profit', 'sum')
).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics = ['Orders', 'Sales', 'Profit']
titles = ['Orders by Segment', 'Sales by Segment', 'Profit by Segment']

for ax, metric, title in zip(axes, metrics, titles):
    wedges, texts, autotexts = ax.pie(
        seg[metric], labels=seg['Segment'],
        autopct='%1.1f%%', colors=[BLUE, GREEN, ORANGE],
        startangle=140, pctdistance=0.75,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax.set_title(title, fontweight='bold', color='#1F4E79')

fig.suptitle('Customer Segment Breakdown', fontsize=14,
             fontweight='bold', color='#1F4E79')
plt.tight_layout()
plt.savefig('plots/11_segment_breakdown.png', bbox_inches='tight')
plt.close()
print("     Saved: plots/11_segment_breakdown.png")

print("  Plot 12: Correlation heatmap...")

num_df = df[['Sales', 'Profit', 'Quantity',
             'Discount', 'Profit Margin']].copy()
corr = num_df.corr()

fig, ax = plt.subplots(figsize=(7, 5))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, ax=ax, square=True,
            cbar_kws={"shrink": 0.8})
ax.set_title('Correlation Between Numerical Columns',
             fontweight='bold', color='#1F4E79', pad=12)
plt.tight_layout()
plt.savefig('plots/12_correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("     Saved: plots/12_correlation_heatmap.png")

print("  Plot 13: Ship mode analysis...")

df['Delivery Days'] = (df['Ship Date'] - df['Order Date']).dt.days
ship = df.groupby('Ship Mode').agg(
    Count=('Order ID', 'nunique'),
    Avg_Days=('Delivery Days', 'mean')
).reset_index().sort_values('Count', ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

bars1 = ax1.bar(ship['Ship Mode'], ship['Count'], color=[
                BLUE, GREEN, ORANGE, PURPLE], alpha=0.88)
ax1.set_ylabel('Number of Orders')
ax1.set_title('Orders by Ship Mode', fontweight='bold', color='#1F4E79')
for bar, val in zip(bars1, ship['Count']):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
             f'{val:,}', ha='center', fontsize=9)

bars2 = ax2.bar(ship['Ship Mode'], ship['Avg_Days'], color=[
                BLUE, GREEN, ORANGE, PURPLE], alpha=0.88)
ax2.set_ylabel('Average Days')
ax2.set_title('Average Delivery Days by Ship Mode',
              fontweight='bold', color='#1F4E79')
for bar, val in zip(bars2, ship['Avg_Days']):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
             f'{val:.1f}d', ha='center', fontsize=9)

plt.suptitle('Ship Mode Analysis', fontsize=14,
             fontweight='bold', color='#1F4E79')
plt.tight_layout()
plt.savefig('plots/13_ship_mode_analysis.png', bbox_inches='tight')
plt.close()
print("     Saved: plots/13_ship_mode_analysis.png")

print("  Plot 14: Quarterly heatmap...")

quarterly = df.groupby(['Order Year', 'Order Quarter'])[
    'Sales'].sum().unstack()
quarterly.columns = ['Q1 (Jan-Mar)', 'Q2 (Apr-Jun)',
                     'Q3 (Jul-Sep)', 'Q4 (Oct-Dec)']

fig, ax = plt.subplots(figsize=(9, 4))
sns.heatmap(quarterly, annot=True, fmt=".0f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Sales ($)"})
ax.set_title('Total Sales by Year × Quarter (Seasonality Heatmap)',
             fontweight='bold', color='#1F4E79', pad=12)
ax.set_ylabel('Year')
ax.set_xlabel('')

for t in ax.texts:
    val = float(t.get_text().replace(',', ''))
    t.set_text(f'${val/1000:.0f}K')

plt.tight_layout()
plt.savefig('plots/14_quarterly_heatmap.png', bbox_inches='tight')
plt.close()
print("     Saved: plots/14_quarterly_heatmap.png")

print("\n" + "="*55)
print("  ALL PLOTS SAVED TO: plots/")
print("="*55)
plots = [
    "01 — Data type overview (what columns exist)",
    "02 — Numerical column distributions (histograms)",
    "03 — Categorical value counts (bar charts)",
    "04 — Sales & profit by category",
    "05 — Profit by sub-category (shows losses in red)",
    "06 — Monthly sales trend 2014-2017 (line chart)",
    "07 — Yearly sales growth + order count",
    "08 — Sales & profit by region",
    "09 — Discount vs profit scatter (by category)",
    "10 — Seasonality: avg sales by month",
    "11 — Segment breakdown (pie charts)",
    "12 — Correlation heatmap of numerical columns",
    "13 — Ship mode: order count & delivery days",
    "14 — Quarterly sales heatmap (year x quarter)",
]
for p in plots:
    print(f"    {p}")
print("="*55)
