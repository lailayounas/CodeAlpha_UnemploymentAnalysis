# ============================================================
#   TASK 2: Unemployment Analysis with Python
#   CodeAlpha Data Science Internship
#   Author: Laila Younas
# ============================================================

# --- IMPORTS ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f9f9f9'

# ============================================================
# STEP 1: LOAD & CLEAN DATASET
# ============================================================
df = pd.read_csv("unemployment.csv")

# Clean column names
df.columns = df.columns.str.strip()
df.rename(columns={
    'Region': 'State',
    'Estimated Unemployment Rate (%)': 'Unemployment_Rate',
    'Estimated Employed': 'Employed',
    'Estimated Labour Participation Rate (%)': 'Labour_Participation_Rate',
    'Region.1': 'Region' if 'Region.1' in df.columns else 'Region'
}, inplace=True)

# Fix duplicate 'Region' column issue
cols = list(df.columns)
if cols.count('Region') == 2:
    idx = [i for i, c in enumerate(cols) if c == 'Region']
    cols[idx[0]] = 'State'
    df.columns = cols

df['Date'] = pd.to_datetime(df['Date'].str.strip(), format='%d-%m-%Y')
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.strftime('%b')
df['Year'] = df['Date'].dt.year

# Drop duplicates, fill missing
df.drop_duplicates(inplace=True)
df['Unemployment_Rate'].fillna(df['Unemployment_Rate'].mean(), inplace=True)

print("=" * 58)
print("    UNEMPLOYMENT ANALYSIS WITH PYTHON — CodeAlpha")
print("=" * 58)
print(f"\n📊 Dataset Shape        : {df.shape}")
print(f"📅 Date Range           : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"🗺️  States Covered       : {df['State'].nunique()}")
print(f"⚠️  Missing Values       : {df.isnull().sum().sum()}")
print(f"\n📌 Columns: {list(df.columns)}")
print("\n--- First 5 Rows ---")
print(df[['State','Date','Unemployment_Rate','Employed','Labour_Participation_Rate']].head())
print("\n--- Statistical Summary ---")
print(df[['Unemployment_Rate','Employed','Labour_Participation_Rate']].describe().round(2))

# ============================================================
# STEP 2: COVID-19 IMPACT ANALYSIS
# ============================================================
# Pre-Covid: Jan–Mar 2020 | During Covid: Apr–Jun 2020
pre_covid  = df[df['Date'] < '2020-04-01']['Unemployment_Rate'].mean()
peak_covid = df[(df['Date'] >= '2020-04-01') & (df['Date'] <= '2020-06-30')]['Unemployment_Rate'].mean()
post_covid = df[df['Date'] > '2020-06-30']['Unemployment_Rate'].mean()

print("\n--- Covid-19 Impact on Unemployment ---")
print(f"  Pre-Covid  (Jan–Mar 2020) : {pre_covid:.2f}%")
print(f"  Peak-Covid (Apr–Jun 2020) : {peak_covid:.2f}%")
print(f"  Post-Covid (Jul 2020+)    : {post_covid:.2f}%")
print(f"  📈 Spike during Covid     : +{peak_covid - pre_covid:.2f}%")

# ============================================================
# STEP 3: NATIONAL MONTHLY TREND
# ============================================================
monthly_avg = df.groupby('Date')['Unemployment_Rate'].mean().reset_index()

# ============================================================
# STEP 4: TOP/BOTTOM STATES
# ============================================================
state_avg = df.groupby('State')['Unemployment_Rate'].mean().sort_values(ascending=False).reset_index()
top5    = state_avg.head(5)
bottom5 = state_avg.tail(5)

print("\n--- Top 5 States (Highest Avg Unemployment) ---")
print(top5.to_string(index=False))
print("\n--- Bottom 5 States (Lowest Avg Unemployment) ---")
print(bottom5.to_string(index=False))

# ============================================================
# STEP 5: REGION-WISE ANALYSIS
# ============================================================
region_col = 'Frequency' if 'Region' not in df.columns else 'Region'
# Use the actual region column name
actual_region = None
for c in df.columns:
    if c.lower() == 'region' and c != 'State':
        actual_region = c
        break

if actual_region:
    region_avg = df.groupby(actual_region)['Unemployment_Rate'].mean().sort_values(ascending=False)
else:
    region_avg = None

# ============================================================
# STEP 6: VISUALIZATIONS
# ============================================================

# --- FIGURE 1: Overview Dashboard ---
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Unemployment Analysis — India (2020)", fontsize=16, fontweight='bold', y=1.01)

colors_main = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

# Plot 1: National Monthly Trend with Covid shading
ax = axes[0, 0]
ax.plot(monthly_avg['Date'], monthly_avg['Unemployment_Rate'],
        color='#E74C3C', linewidth=2.5, marker='o', markersize=5, label='Unemployment Rate')
ax.axvspan(pd.Timestamp('2020-04-01'), pd.Timestamp('2020-06-30'),
           alpha=0.2, color='red', label='Covid-19 Lockdown')
ax.axhline(y=pre_covid, color='green', linestyle='--', linewidth=1.2, label=f'Pre-Covid Avg ({pre_covid:.1f}%)')
ax.set_title('National Monthly Unemployment Trend', fontsize=12, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Unemployment Rate (%)')
ax.legend(fontsize=8)
ax.tick_params(axis='x', rotation=30)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())

# Plot 2: Covid Impact Bar
ax = axes[0, 1]
phases = ['Pre-Covid\n(Jan–Mar)', 'Peak-Covid\n(Apr–Jun)', 'Post-Covid\n(Jul+)']
values = [pre_covid, peak_covid, post_covid]
bar_colors = ['#2ECC71', '#E74C3C', '#3498DB']
bars = ax.bar(phases, values, color=bar_colors, width=0.5, edgecolor='white', linewidth=1.5)
ax.set_title('Covid-19 Impact on Unemployment Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('Avg Unemployment Rate (%)')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, max(values) * 1.2)

# Plot 3: Top 10 States Horizontal Bar
ax = axes[1, 0]
top10 = state_avg.head(10)
bar_colors_top = ['#E74C3C' if i < 3 else '#F39C12' if i < 6 else '#3498DB' for i in range(10)]
ax.barh(top10['State'][::-1], top10['Unemployment_Rate'][::-1],
        color=bar_colors_top[::-1], edgecolor='white')
ax.set_title('Top 10 States — Highest Unemployment', fontsize=12, fontweight='bold')
ax.set_xlabel('Avg Unemployment Rate (%)')
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
for i, (val, name) in enumerate(zip(top10['Unemployment_Rate'][::-1], top10['State'][::-1])):
    ax.text(val + 0.2, i, f'{val:.1f}%', va='center', fontsize=8)

# Plot 4: Distribution Box Plot by Month
ax = axes[1, 1]
monthly_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
df['Month_Name'] = pd.Categorical(df['Month_Name'], categories=monthly_order, ordered=True)
monthly_data = [df[df['Month_Name'] == m]['Unemployment_Rate'].dropna().values
                for m in monthly_order if m in df['Month_Name'].values]
month_labels = [m for m in monthly_order if m in df['Month_Name'].values]
bp = ax.boxplot(monthly_data, patch_artist=True, notch=False)
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor('#E74C3C' if month_labels[i] in ['Apr','May','Jun'] else '#3498DB')
    patch.set_alpha(0.7)
ax.set_xticklabels(month_labels, fontsize=8)
ax.set_title('Monthly Unemployment Distribution (Boxplot)', fontsize=12, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Unemployment Rate (%)')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())

# Red = Covid months legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#E74C3C', alpha=0.7, label='Covid Months (Apr–Jun)'),
                   Patch(facecolor='#3498DB', alpha=0.7, label='Other Months')]
ax.legend(handles=legend_elements, fontsize=8)

plt.tight_layout()
plt.savefig('unemployment_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Plot saved: unemployment_overview.png")

# --- FIGURE 2: Heatmap State × Month ---
fig, ax = plt.subplots(figsize=(14, 10))
pivot = df.pivot_table(values='Unemployment_Rate', index='State', columns='Month_Name', aggfunc='mean')
pivot = pivot.reindex(columns=[m for m in monthly_order if m in pivot.columns])
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', linewidths=0.3,
            ax=ax, cbar_kws={'label': 'Unemployment Rate (%)'})
ax.set_title('State × Month Unemployment Heatmap — Covid Impact Visible',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Month', fontsize=11)
ax.set_ylabel('State', fontsize=11)
ax.tick_params(axis='y', labelsize=8)
plt.tight_layout()
plt.savefig('unemployment_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot saved: unemployment_heatmap.png")

# ============================================================
# STEP 7: KEY INSIGHTS
# ============================================================
worst_state = state_avg.iloc[0]
best_state  = state_avg.iloc[-1]
peak_month  = monthly_avg.loc[monthly_avg['Unemployment_Rate'].idxmax()]

print("\n--- KEY INSIGHTS ---")
print(f"  📈 Highest Avg Unemployment State : {worst_state['State']} ({worst_state['Unemployment_Rate']:.2f}%)")
print(f"  📉 Lowest Avg Unemployment State  : {best_state['State']} ({best_state['Unemployment_Rate']:.2f}%)")
print(f"  🦠 Covid Spike                    : {pre_covid:.1f}% → {peak_covid:.1f}% (+{peak_covid-pre_covid:.1f}%)")
print(f"  📅 Peak Unemployment Month        : {peak_month['Date'].strftime('%B %Y')} ({peak_month['Unemployment_Rate']:.2f}%)")
print(f"  🔄 Recovery Post-Covid            : {post_covid:.2f}% avg (recovering towards normal)")

print("\n" + "=" * 58)
print("   TASK 2 COMPLETE — Unemployment Analysis ✅")
print("=" * 58)
