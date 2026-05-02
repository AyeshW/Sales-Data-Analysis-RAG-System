import pandas as pd
import numpy as np
import json
from collections import Counter

df = pd.read_csv('../data/Sample - Superstore.csv', encoding='latin-1')

print(f"Dataframe shape: {df.shape}")
print(f"Dataframe columns: {df.columns}")

df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

df['Order Year'] = df['Order Date'].dt.year
df['Order Month'] = df['Order Date'].dt.month
df['Order Month Name'] = df['Order Date'].dt.strftime('%B')
df['Order Quarter'] = df['Order Date'].dt.quarter

for col in ['Category', 'Sub-Category', 'Region', 'Segment', 'Ship Mode']:
    df[col] = df[col].str.strip().str.title()

df['Profit Margin'] = df.apply(
    lambda row: round(row['Profit'] / row['Sales'],
                      4) if row['Sales'] != 0 else 0,
    axis=1
)

df.to_csv('../data/superstore_clean.csv', index=False)
print("Preprocessing done.File saved.")


def order_to_text(order_df):

    order_df = order_df.reset_index(drop=True)
    first = order_df.iloc[0]

    order_id = first['Order ID']
    order_date = first['Order Date'].strftime('%B %d, %Y')
    customer = first['Customer Name']
    segment = first['Segment']
    ship_mode = first['Ship Mode']
    city = first['City']
    state = first['State']
    region = first['Region']

    header = (
        f"Order {order_id} was placed on {order_date} by {customer} "
        f"({segment} segment), shipped via {ship_mode} to {city}, "
        f"{state}, {region} region."
    )

    product_lines = []
    for i, row in order_df.iterrows():
        discount_str = (
            f"with a {int(row['Discount']*100)}% discount"
            if row['Discount'] > 0 else "with no discount"
        )
        profit_str = (
            f"profit of ${row['Profit']:.2f}"
            if row['Profit'] >= 0 else f"loss of ${abs(row['Profit']):.2f}"
        )
        line = (
            f"({i+1}) {row['Product Name']} — {row['Category']}/{row['Sub-Category']}, "
            f"{int(row['Quantity'])} unit(s), ${row['Sales']:.2f} {discount_str}, {profit_str}."
        )
        product_lines.append(line)

    products_text = " ".join(product_lines)

    total_sales = order_df['Sales'].sum()
    total_profit = order_df['Profit'].sum()
    total_quantity = order_df['Quantity'].sum()
    num_products = len(order_df)

    profit_summary = (
        f"total profit of ${total_profit:.2f}"
        if total_profit >= 0 else f"total loss of ${abs(total_profit):.2f}"
    )

    summary = (
        f"This order contained {num_products} product(s) totalling "
        f"{int(total_quantity)} items. Order total: ${total_sales:.2f} in sales "
        f"with {profit_summary}."
    )

    return f"{header} {products_text} {summary}"


grouped = df.groupby('Order ID', sort=False)
order_records = []
for order_id, group in grouped:
    text = order_to_text(group)
    order_records.append({'Order ID': order_id, 'text': text})

order_texts = pd.DataFrame(order_records)


print("\n Sample order descriptions:")
for i in range(3):
    print(f"\n--- Order {i+1} ---")
    print(order_texts['text'].iloc[i])


summary_docs = []

# Monthly Summary
monthly = df.groupby(['Order Year', 'Order Month', 'Order Month Name']).agg(
    total_sales=('Sales', 'sum'),
    total_profit=('Profit', 'sum'),
    num_orders=('Order ID', 'nunique'),
    avg_discount=('Discount', 'mean')
).reset_index()


monthly_cat = df.groupby(
    ['Order Year', 'Order Month', 'Category']
)['Sales'].sum().reset_index()


monthly_reg = df.groupby(
    ['Order Year', 'Order Month', 'Region']
)['Sales'].sum().reset_index()


monthly_seg = df.groupby(
    ['Order Year', 'Order Month', 'Segment']
)['Order ID'].nunique().reset_index()
monthly_seg.columns = ['Order Year', 'Order Month', 'Segment', 'seg_orders']

for _, row in monthly.iterrows():
    year = int(row['Order Year'])
    month = int(row['Order Month'])

    cat_data = monthly_cat[
        (monthly_cat['Order Year'] == year) &
        (monthly_cat['Order Month'] == month)
    ].sort_values('Sales', ascending=False).reset_index(drop=True)

    top_cat = cat_data.iloc[0]
    top_cat_pct = (top_cat['Sales'] / row['total_sales']) * 100
    other_cats = cat_data.iloc[1:]
    other_cat_str = " and ".join(
        [f"{r['Category']} ${r['Sales']:.0f}" for _, r in other_cats.iterrows()]
    )

    reg_data = monthly_reg[
        (monthly_reg['Order Year'] == year) &
        (monthly_reg['Order Month'] == month)
    ].sort_values('Sales', ascending=False).reset_index(drop=True)

    top_reg = reg_data.iloc[0]
    second_reg = reg_data.iloc[1] if len(reg_data) > 1 else None
    second_reg_str = (
        f" followed by {second_reg['Region']} with ${second_reg['Sales']:.0f}"
        if second_reg is not None else ""
    )

    seg_data = monthly_seg[
        (monthly_seg['Order Year'] == year) &
        (monthly_seg['Order Month'] == month)
    ].sort_values('seg_orders', ascending=False).reset_index(drop=True)

    top_seg = seg_data.iloc[0]
    total_seg_ord = seg_data['seg_orders'].sum()
    top_seg_pct = (top_seg['seg_orders'] / total_seg_ord) * 100

    margin = (row['total_profit'] / row['total_sales']) * \
        100 if row['total_sales'] != 0 else 0

    existing = (
        f"In {row['Order Month Name']} {year}, "
        f"the store had {int(row['num_orders'])} orders with total sales of "
        f"${row['total_sales']:.2f} and total profit of ${row['total_profit']:.2f}."
    )

    enriched = (
        f" Top performing category was {top_cat['Category']} with "
        f"${top_cat['Sales']:.0f} ({top_cat_pct:.1f}% of total). "
        f"{other_cat_str}. "
        f"Best region was {top_reg['Region']} with ${top_reg['Sales']:.0f} in sales"
        f"{second_reg_str}. "
        f"Average discount applied was {row['avg_discount']*100:.1f}%. "
        f"Total profit ${row['total_profit']:.0f} representing a {margin:.1f}% margin. "
        f"{top_seg['Segment']} segment drove "
        f"{top_seg_pct:.0f}% of orders."
    )

    text = existing + enriched

    summary_docs.append({
        'type': 'monthly_summary',
        'text': text,
        'year': year,
        'month': month
    })

# Yearly Summary
yearly_detail = df.groupby('Order Year').agg(
    total_sales=('Sales', 'sum'),
    total_profit=('Profit', 'sum'),
    num_orders=('Order ID', 'nunique'),
    avg_order_value=('Sales', 'mean')
).reset_index()

# Find best month per year
best_month_per_year = (
    df.groupby(['Order Year', 'Order Month Name'])['Sales']
    .sum()
    .reset_index()
    .sort_values('Sales', ascending=False)
    .groupby('Order Year')
    .first()
    .reset_index()
)
best_month_map = dict(zip(
    best_month_per_year['Order Year'],
    best_month_per_year['Order Month Name']
))

for _, row in yearly_detail.iterrows():
    year = int(row['Order Year'])
    best_month = best_month_map.get(year, 'N/A')
    yearly_margin = (row['total_profit'] / row['total_sales']
                     * 100) if row['total_sales'] != 0 else 0
    text = (
        f"In {year}, the store recorded total sales of ${row['total_sales']:.2f} "
        f"with a total profit of ${row['total_profit']:.2f} and an overall profit margin of "
        f"{yearly_margin:.1f}% across {int(row['num_orders'])} unique orders. "
        f"The average order value was ${row['avg_order_value']:.2f}. "
        f"The best performing month of {year} was {best_month} based on total sales."
    )
    summary_docs.append({'type': 'yearly_summary', 'text': text, 'year': year})


# Quarterly Summary
quarterly = df.groupby(['Order Year', 'Order Quarter']).agg(
    total_sales=('Sales', 'sum'),
    total_profit=('Profit', 'sum'),
    num_orders=('Order ID', 'nunique')
).reset_index()

# Find best category per quarter+year
best_cat_per_quarter = (
    df.groupby(['Order Year', 'Order Quarter', 'Category'])['Sales']
    .sum()
    .reset_index()
    .sort_values('Sales', ascending=False)
    .groupby(['Order Year', 'Order Quarter'])
    .first()
    .reset_index()
)
best_cat_map = {
    (int(r['Order Year']), int(r['Order Quarter'])): r['Category']
    for _, r in best_cat_per_quarter.iterrows()
}

quarter_names = {1: 'Q1 (Jan-Mar)', 2: 'Q2 (Apr-Jun)',
                 3: 'Q3 (Jul-Sep)', 4: 'Q4 (Oct-Dec)'}

for _, row in quarterly.iterrows():
    year = int(row['Order Year'])
    quarter = int(row['Order Quarter'])
    q_name = quarter_names[quarter]
    best_cat = best_cat_map.get((year, quarter), 'N/A')
    q_margin = (row['total_profit'] / row['total_sales']
                * 100) if row['total_sales'] != 0 else 0
    text = (
        f"In {q_name} of {year}, the store generated total sales of "
        f"${row['total_sales']:.2f} with total profit of ${row['total_profit']:.2f} "
        f"and a profit margin of {q_margin:.1f}% across {int(row['num_orders'])} orders. "
        f"The top performing product category this quarter was {best_cat}."
    )
    summary_docs.append({
        'type': 'quarterly_summary', 'text': text,
        'year': year, 'quarter': quarter
    })

# Category Summary

category = df.groupby('Category').agg(
    total_sales=('Sales', 'sum'),
    total_profit=('Profit', 'sum'),
    avg_margin=('Profit Margin', 'mean'),
    num_orders=('Order ID', 'nunique')
).reset_index()

for _, row in category.iterrows():
    text = (
        f"Category '{row['Category']}' generated total sales of ${row['total_sales']:.2f} "
        f"with total profit of ${row['total_profit']:.2f} across {int(row['num_orders'])} orders. "
        f"The average profit margin for this category is {row['avg_margin']*100:.1f}%."
    )
    summary_docs.append({'type': 'category_summary',
                        'text': text, 'category': row['Category']})

# Region Summary

regional = df.groupby('Region').agg(
    total_sales=('Sales', 'sum'),
    total_profit=('Profit', 'sum'),
    num_orders=('Order ID', 'nunique')
).reset_index()

for _, row in regional.iterrows():
    reg_margin = (row['total_profit'] / row['total_sales']
                  * 100) if row['total_sales'] != 0 else 0
    text = (
        f"The {row['Region']} region recorded total sales of ${row['total_sales']:.2f} "
        f"and total profit of ${row['total_profit']:.2f} with a profit margin of "
        f"{reg_margin:.1f}% from {int(row['num_orders'])} orders."
    )
    summary_docs.append({'type': 'regional_summary',
                        'text': text, 'region': row['Region']})

# Subcatergory Summary
subcat = df.groupby('Sub-Category').agg(
    total_sales=('Sales', 'sum'),
    total_profit=('Profit', 'sum'),
    avg_margin=('Profit Margin', 'mean'),
    num_orders=('Order ID', 'nunique')
).reset_index()

for _, row in subcat.iterrows():
    profit_str = (
        f"profit of ${row['total_profit']:.2f}"
        if row['total_profit'] >= 0 else f"loss of ${abs(row['total_profit']):.2f}"
    )
    text = (
        f"The {row['Sub-Category']} sub-category generated total sales of "
        f"${row['total_sales']:.2f} with a {profit_str} across "
        f"{int(row['num_orders'])} orders. "
        f"Its average profit margin was {row['avg_margin']*100:.1f}%."
    )
    summary_docs.append({
        'type': 'subcategory_summary', 'text': text,
        'subcategory': row['Sub-Category']
    })
# Seasonal Summary Patterns
season_map = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring',  4: 'Spring', 5: 'Spring',
    6: 'Summer',  7: 'Summer', 8: 'Summer',
    9: 'Fall',   10: 'Fall',  11: 'Fall'
}
df['Season'] = df['Order Month'].map(season_map)

seasonal = df.groupby('Season').agg(
    total_sales=('Sales', 'sum'),
    total_profit=('Profit', 'sum'),
    avg_sales=('Sales', 'mean'),
    num_orders=('Order ID', 'nunique')
).reset_index()

best_season = seasonal.loc[seasonal['total_sales'].idxmax(), 'Season']

for _, row in seasonal.iterrows():
    is_best = " This is the best performing season overall." if row[
        'Season'] == best_season else ""
    sea_margin = (row['total_profit'] / row['total_sales']
                  * 100) if row['total_sales'] != 0 else 0
    text = (
        f"During {row['Season']}, the store averaged ${row['avg_sales']:.2f} per transaction "
        f"with total sales of ${row['total_sales']:.2f} and total profit of "
        f"${row['total_profit']:.2f} representing a profit margin of {sea_margin:.1f}% "
        f"across {int(row['num_orders'])} orders.{is_best}"
    )
    summary_docs.append({'type': 'seasonality_summary',
                        'text': text, 'season': row['Season']})

# Combned seasonal pattern
season_order = ['Spring', 'Summer', 'Fall', 'Winter']
season_data = seasonal.set_index('Season')
season_lines = []
for s in season_order:
    if s in season_data.index:
        season_lines.append(
            f"{s}: ${season_data.loc[s, 'total_sales']:.2f} in sales, "
            f"profit ${season_data.loc[s, 'total_profit']:.2f}"
        )

text = (
    f"Seasonality pattern across all years: "
    + " | ".join(season_lines) + f". "
    f"Fall consistently drives the highest sales volume, followed by Winter, "
    f"making Q4 and Q3 the most critical periods for the business."
)
summary_docs.append({'type': 'seasonality_pattern_overall', 'text': text})


# Comparative Category vs Category
cat_data = category.set_index('Category')
tech = cat_data.loc['Technology']
furn = cat_data.loc['Furniture']
office = cat_data.loc['Office Supplies']

text = (
    f"Comparing product categories: Technology had total sales of ${tech['total_sales']:.2f} "
    f"with profit of ${tech['total_profit']:.2f} (avg margin: {tech['avg_margin']*100:.1f}%). "
    f"Furniture had total sales of ${furn['total_sales']:.2f} "
    f"with profit of ${furn['total_profit']:.2f} (avg margin: {furn['avg_margin']*100:.1f}%). "
    f"Office Supplies had total sales of ${office['total_sales']:.2f} "
    f"with profit of ${office['total_profit']:.2f} (avg margin: {office['avg_margin']*100:.1f}%). "
    f"Technology leads in total sales while Office Supplies has the highest profit margin."
)
summary_docs.append({'type': 'comparative_category', 'text': text})

# Comparative Region v Region
reg_data = regional.set_index('Region')
west = reg_data.loc['West']
east = reg_data.loc['East']
central = reg_data.loc['Central']
south = reg_data.loc['South']

text = (
    f"Regional profit comparison: The West region had total profit of ${west['total_profit']:.2f} "
    f"from ${west['total_sales']:.2f} in sales. The East region had total profit of "
    f"${east['total_profit']:.2f} from ${east['total_sales']:.2f} in sales. "
    f"The Central region had total profit of ${central['total_profit']:.2f} "
    f"from ${central['total_sales']:.2f} in sales. "
    f"The South region had total profit of ${south['total_profit']:.2f} "
    f"from ${south['total_sales']:.2f} in sales. "
    f"The West leads in both sales and profit among all regions."
)
summary_docs.append({'type': 'comparative_regional', 'text': text})

# Comparative Segment v Segment
segment = df.groupby('Segment').agg(
    total_sales=('Sales', 'sum'),
    total_profit=('Profit', 'sum'),
    avg_margin=('Profit Margin', 'mean'),
    num_orders=('Order ID', 'nunique')
).reset_index()

seg_data = segment.set_index('Segment')
consumer = seg_data.loc['Consumer']
corporate = seg_data.loc['Corporate']
home_office = seg_data.loc['Home Office']

text = (
    f"Customer segment comparison: The Consumer segment generated ${consumer['total_sales']:.2f} "
    f"in sales with profit of ${consumer['total_profit']:.2f} across {int(consumer['num_orders'])} orders. "
    f"The Corporate segment generated ${corporate['total_sales']:.2f} in sales "
    f"with profit of ${corporate['total_profit']:.2f} across {int(corporate['num_orders'])} orders. "
    f"The Home Office segment generated ${home_office['total_sales']:.2f} in sales "
    f"with profit of ${home_office['total_profit']:.2f} across {int(home_office['num_orders'])} orders. "
    f"Consumer is the largest segment by volume while Home Office has the highest average profit margin "
    f"at {home_office['avg_margin']*100:.1f}%."
)
summary_docs.append({'type': 'comparative_segment', 'text': text})

# Comparative Year-over-Year
yearly = df.groupby('Order Year').agg(
    total_sales=('Sales', 'sum'),
    total_profit=('Profit', 'sum'),
    num_orders=('Order ID', 'nunique')
).reset_index().sort_values('Order Year')

year_texts = []
for _, row in yearly.iterrows():
    year_texts.append(
        f"{int(row['Order Year'])}: sales=${row['total_sales']:.2f}, "
        f"profit=${row['total_profit']:.2f}, orders={int(row['num_orders'])}"
    )

text = (
    f"Year-over-year performance comparison: "
    + " | ".join(year_texts) + ". "
    f"Sales have grown consistently each year, with {int(yearly.iloc[-1]['Order Year'])} "
    f"being the highest performing year with ${yearly.iloc[-1]['total_sales']:.2f} in total sales."
)
summary_docs.append({'type': 'comparative_yearly', 'text': text})

# COmparative Discount Impact
df['Discount Band'] = pd.cut(
    df['Discount'],
    bins=[-0.01, 0.0, 0.2, 0.5, 0.8],
    labels=['No Discount', 'Low (1-20%)', 'Medium (21-50%)', 'High (51-80%)']
)

discount_analysis = df.groupby('Discount Band', observed=True).agg(
    avg_profit=('Profit', 'mean'),
    avg_sales=('Sales', 'mean'),
    num_orders=('Order ID', 'nunique')
).reset_index()

disc_texts = []
for _, row in discount_analysis.iterrows():
    disc_texts.append(
        f"{row['Discount Band']}: avg profit=${row['avg_profit']:.2f}, "
        f"avg sales=${row['avg_sales']:.2f}, orders={int(row['num_orders'])}"
    )

text = (
    f"Comparison of sales performance by discount level: "
    + " | ".join(disc_texts) + ". "
    f"Higher discounts consistently lead to lower or negative average profits, "
    f"suggesting heavy discounting hurts overall profitability."
)
summary_docs.append({'type': 'comparative_discount_impact', 'text': text})

# Yearly Category Breakdown
yearly_cat = df.groupby(['Order Year', 'Category']).agg(
    total_sales=('Sales', 'sum'),
    total_profit=('Profit', 'sum'),
    avg_margin=('Profit Margin', 'mean'),
    num_orders=('Order ID', 'nunique')
).reset_index()

for _, row in yearly_cat.iterrows():
    year = int(row['Order Year'])
    text = (
        f"In {year}, the {row['Category']} category generated total sales of "
        f"${row['total_sales']:.2f} with a profit of ${row['total_profit']:.2f} "
        f"across {int(row['num_orders'])} orders. "
        f"The average profit margin for {row['Category']} in {year} was "
        f"{row['avg_margin']*100:.1f}%."
    )
    summary_docs.append({
        'type': 'yearly_category_summary', 'text': text,
        'year': year, 'category': row['Category']
    })


# Regional Yearly Summary
regional_yearly = df.groupby(['Order Year', 'Region']).agg(
    total_sales=('Sales', 'sum'),
    total_profit=('Profit', 'sum'),
    num_orders=('Order ID', 'nunique')
).reset_index()

for _, row in regional_yearly.iterrows():
    year = int(row['Order Year'])
    ry_margin = (row['total_profit'] / row['total_sales']
                 * 100) if row['total_sales'] != 0 else 0
    text = (
        f"In {year}, the {row['Region']} region achieved total sales of "
        f"${row['total_sales']:.2f} with total profit of ${row['total_profit']:.2f} "
        f"and a profit margin of {ry_margin:.1f}% from {int(row['num_orders'])} orders."
    )
    summary_docs.append({
        'type': 'regional_yearly_summary', 'text': text,
        'year': year, 'region': row['Region']
    })

# Regional x Category Summary
region_cat = df.groupby(['Region', 'Category']).agg(
    total_sales=('Sales', 'sum'),
    total_profit=('Profit', 'sum'),
    avg_margin=('Profit Margin', 'mean'),
    num_orders=('Order ID', 'nunique')
).reset_index()

for _, row in region_cat.iterrows():
    profit_str = (
        f"profit of ${row['total_profit']:.2f}"
        if row['total_profit'] >= 0 else f"loss of ${abs(row['total_profit']):.2f}"
    )
    text = (
        f"In the {row['Region']} region, the {row['Category']} category generated "
        f"total sales of ${row['total_sales']:.2f} with a {profit_str} across "
        f"{int(row['num_orders'])} orders. "
        f"The average profit margin was {row['avg_margin']*100:.1f}%."
    )
    summary_docs.append({
        'type': 'region_category_summary', 'text': text,
        'region': row['Region'], 'category': row['Category']
    })

# Quarterly x Region Summary
quarterly_region = df.groupby(['Order Quarter', 'Region']).agg(
    total_sales=('Sales', 'sum'),
    total_profit=('Profit', 'sum'),
    num_orders=('Order ID', 'nunique')
).reset_index()

quarter_names = {
    1: 'Q1 (Jan-Mar)', 2: 'Q2 (Apr-Jun)',
    3: 'Q3 (Jul-Sep)', 4: 'Q4 (Oct-Dec)'
}

for _, row in quarterly_region.iterrows():
    q_name = quarter_names[int(row['Order Quarter'])]
    qr_margin = (row['total_profit'] / row['total_sales']
                 * 100) if row['total_sales'] != 0 else 0
    text = (
        f"In {q_name}, the {row['Region']} region generated total sales of "
        f"${row['total_sales']:.2f} with total profit of ${row['total_profit']:.2f} "
        f"and a profit margin of {qr_margin:.1f}% across {int(row['num_orders'])} orders."
    )
    summary_docs.append({
        'type': 'quarterly_region_summary', 'text': text,
        'quarter': int(row['Order Quarter']), 'region': row['Region']
    })


print(f"Created {len(summary_docs)} total summary documents")
print(f"   Breakdown:")
type_counts = Counter(doc['type'] for doc in summary_docs)
for doc_type, count in type_counts.items():
    print(f"   - {doc_type}: {count}")

# Saving to Json
all_docs = []


order_meta = {}
for order_id, group in df.groupby('Order ID', sort=False):
    group = group.reset_index(drop=True)
    first = group.iloc[0]

    top_category = group['Category'].mode()[0]
    top_subcategory = group['Sub-Category'].mode()[0]

    order_meta[order_id] = {
        'doc_type': 'transaction',
        'order_id': order_id,
        'year': int(first['Order Year']),
        'month': int(first['Order Month']),
        'category': top_category,
        'sub_category': top_subcategory,
        'region': first['Region'],
        'segment': first['Segment']
    }

for i, row in order_texts.iterrows():
    all_docs.append({
        'doc_id': i,
        'text': row['text'],
        'metadata': order_meta[row['Order ID']]
    })

for i, doc in enumerate(summary_docs):
    all_docs.append({
        'doc_id': len(order_texts) + i,
        'text': doc['text'],
        'metadata': {
            'doc_type': doc['type'],
            **{k: v for k, v in doc.items() if k not in ['text', 'type']}
        }
    })


print(f"\n   Total documents to save: {len(all_docs)}")
print(f"   Breakdown by doc_type:")
all_type_counts = Counter(doc['metadata']['doc_type'] for doc in all_docs)
for doc_type, count in sorted(all_type_counts.items()):
    print(f"   - {doc_type}: {count}")

with open('../data/datastore/chunks.json', 'w') as f:
    json.dump(all_docs, f, indent=2)

print(f"\n Saved {len(all_docs)} total documents.")
print(f"   - Order transaction docs : {len(order_texts)}")
print(f"   - Summary docs           : {len(summary_docs)}")
