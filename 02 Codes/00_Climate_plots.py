import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import os

# Fonts
custom_font_path = ".\Fonts\Archivo_SemiCondensed-Medium.ttf"  
custom_font = FontProperties(fname=custom_font_path)

custom_font_path_2 = ".\Fonts\Archivo_SemiCondensed-Medium.ttf"  
custom_font_2 = FontProperties(fname=custom_font_path_2)

# Apply the custom font globally
rcParams['font.family'] = custom_font.get_name()

# Scenario colors
scenario_colors = {
    'T1': '#22181C',
    'T2': '#59C3C3',
    'T3': '#FFAC27',
    'T4': '#EA002D'
}

# Font sizes
font_small = 10
font_larger = 12

# Adjusted y-axis limits
ylim_pmv = (0.0, 1.02)
ylim_t = (21,36)
ylim_rh = (27,62)


#%% Read and filter
data = pd.read_csv('.\\Data\\00_SL2201_CombinedData.csv')
# Apply filters to the data
filtered_data = data[
    (data['lab-id'] == 'SL') &
    (data['break'] != 1) &
    (data['cluster'].isin(['Morning', 'Afternoon'])) &
    # (data['time-elapsed'] > 9) &
    (data['time-elapsed'] < 180)
]

# Create time bins
filtered_data['time-bin'] = (filtered_data['time-elapsed'] // 10) * 10

# Variables to analyze
variables = ['top', 'tdb', 'mrt', 'rh', "v",'pmv']

# Calculate mean and std per scenario per 10-min bin
binned_stats = (
    filtered_data.groupby(['scenario', 'time-bin'])[variables]
    .agg(['mean', 'std'])
    .reset_index()
)

# Calculate overall mean and std per scenario
summary = (
    filtered_data.groupby('scenario')[variables]
    .agg(['mean', 'std'])
)

# Flatten MultiIndex column names
summary.columns = ['_'.join(col) for col in summary.columns]

# Format as "mean ± std"
summary_formatted = summary.apply(
    lambda row: {
        var: f"{row[f'{var}_mean']:.2f} ± {row[f'{var}_std']:.2f}" for var in variables
    },
    axis=1
)

# Convert to DataFrame for display
summary_df = pd.DataFrame(summary_formatted.tolist(), index=summary.index)

# Display or export
print(summary_df)


# Group by part-id and scenario, then calculate mean and std of PMV
pmv_stats = (
    filtered_data.groupby(['part-id', 'scenario'])[
        ['pmv', 'top', 'tdb', 'mrt', 'rh', 'v', 'clo']
    ]
    .agg(['mean', 'std'])
    .reset_index()
)

# Flatten MultiIndex column names (e.g. ('pmv','mean') → 'pmv_mean')
pmv_stats.columns = [
    f"{col[0]}_{col[1]}" if col[1] else col[0]
    for col in pmv_stats.columns
]

# Display or save
print(pmv_stats)
pmv_stats.to_csv("pmv_stats_by_partid_scenario.csv", index=False)



#%% PMV plots
filtered_data_t2_t3_outliers_removed_v2 = filtered_data.copy()
# Create the figure
fig, axes = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={'width_ratios': [2.5, 1], 'wspace': 0.05})

# Line plot for PMV-ASHRAE over elapsed time
ax0 = axes[0]
for scenario, color in scenario_colors.items():
    scenario_data = binned_stats[binned_stats['scenario'] == scenario]
    ax0.errorbar(
        scenario_data['time-bin'],
        scenario_data[('pmv', 'mean')],
        yerr=scenario_data[('pmv', 'std')],
        label=scenario,
        fmt='o--',
        color=color,
        linewidth=1,
        markersize=4
    )

# Configure the line plot
ax0.axhline(0.5, color='#2d2d2d', linestyle='-', linewidth=1, zorder=1) 
ax0.set_xticks(range(0, 181, 30))
ax0.set_title("", fontsize=font_larger, fontproperties=custom_font)
ax0.set_xlabel('Elapsed time (minutes)', fontsize=font_larger, fontproperties=custom_font)
ax0.set_ylabel('PMV (ASHRAE)', fontsize=font_larger, fontproperties=custom_font)
ax0.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.1),
    ncol=4,
    frameon=False
)
ax0.set_ylim(ylim_pmv)  # Apply the new y-axis limits
ax0.grid(True, linestyle='--', linewidth=0.5)

# Remove top and right frame edges
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)

# Apply custom font to tick labels
for label in ax0.get_xticklabels():
    label.set_fontproperties(custom_font_2)
    label.set_fontsize(font_small)
for label in ax0.get_yticklabels():
    label.set_fontproperties(custom_font_2)
    label.set_fontsize(font_small)


# Boxplot for PMV-ASHRAE by scenario
ax1 = axes[1]
boxplot = filtered_data_t2_t3_outliers_removed_v2.boxplot(
    column='pmv',
    by='scenario',
    ax=ax1,
    patch_artist=True,
    return_type='axes',  # Suppress default title
    widths=0.6,
    boxprops=dict(facecolor='#2d2d2d', color='#2d2d2d'),
    medianprops=dict(color='white', linewidth=1),
    whiskerprops=dict(color='#2d2d2d', linewidth=1.5),
    capprops=dict(color='#2d2d2d', linewidth=1.5),
    flierprops=dict(marker='o', color='#2d2d2d', markerfacecolor='#2d2d2d',
                    markeredgecolor='white', markeredgewidth=0.2, markersize=4)
)

plt.suptitle("")  # Clears the default title

# Add mean values to the boxplot in white font
means = filtered_data_t2_t3_outliers_removed_v2.groupby('scenario')['pmv'].mean()
for i, (scenario, mean) in enumerate(means.items()):
    # Add the mean value as text
    ax1.text(
        i + 1, mean + 0.01, f"{mean:.2f}",
        ha='center', color='white',
        fontproperties=custom_font,
        fontsize=font_small
    )
    # Add the mean value as a small dot
    ax1.plot(
        i + 1, mean, 'x',  # Small dot marker
        color='white', markersize=3  # Adjust size and color of the dot
    )

ax1.set_ylim(ylim_pmv)  # Apply the new y-axis limits
ax1.grid(True, linestyle='--', linewidth=0.5)

# Remove top and right frame edges
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)

# Configure the boxplot
ax1.axhline(0.5, color='#2d2d2d', linestyle='-', linewidth=1, zorder=1)
ax1.yaxis.tick_right()  # Move y-axis to the right
ax1.yaxis.set_label_position("right")  # Label on the right side
ax1.set_title("", fontsize=font_larger, fontproperties=custom_font)
ax1.set_xlabel('Scenario', fontsize=font_larger, fontproperties=custom_font)
ax1.set_ylabel('PMV (ASHRAE)', fontsize=font_larger, fontproperties=custom_font)

# Apply custom font to tick labels
for label in ax1.get_xticklabels():
    label.set_fontproperties(custom_font)
    label.set_fontsize(font_small)

for label in ax1.get_yticklabels():
    label.set_fontproperties(custom_font)
    label.set_fontsize(font_small)
    
# Save the figure
output_folder_path = ".\\Graphs\\Climate"  # Replace with your desired folder path
output_file_name = "pmv_boxplot_and_line"  # File name without extension

os.makedirs(output_folder_path, exist_ok=True)

jpeg_path = os.path.join(output_folder_path, f"{output_file_name}.jpeg")
pdf_path = os.path.join(output_folder_path, f"{output_file_name}.pdf")

fig.savefig(jpeg_path, format='jpeg', dpi=300)  
fig.savefig(pdf_path, format='pdf')  

jpeg_path, pdf_path


#%% top plots
# Create the figure
fig, axes = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={'width_ratios': [2.5, 1], 'wspace': 0.05})

# Line plot for top-ashrae over elapsed time
ax0 = axes[0]
for scenario, color in scenario_colors.items():
    scenario_data = binned_stats[binned_stats['scenario'] == scenario]
    ax0.errorbar(
        scenario_data['time-bin'],
        scenario_data[('top', 'mean')],
        yerr=scenario_data[('top', 'std')],
        label=scenario,
        fmt='o--',
        color=color,
        linewidth=1,
        markersize=4
    )

# Configure the line plot
ax0.axhline(0.5, color='#2d2d2d', linestyle='-', linewidth=1, zorder=1) 
ax0.set_xticks(range(0, 181, 30))
ax0.set_title("", fontsize=font_larger, fontproperties=custom_font)
ax0.set_xlabel('Elapsed time (minutes)', fontsize=font_larger, fontproperties=custom_font)
ax0.set_ylabel('Operative temperature (°C)', fontsize=font_larger, fontproperties=custom_font)
ax0.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.1),
    ncol=4,
    frameon=False
)
ax0.set_ylim(ylim_t)  # Apply the new y-axis limits
ax0.grid(True, linestyle='--', linewidth=0.5)

# Remove top and right frame edges
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)

# Apply custom font to tick labels
for label in ax0.get_xticklabels():
    label.set_fontproperties(custom_font_2)
    label.set_fontsize(font_small)
for label in ax0.get_yticklabels():
    label.set_fontproperties(custom_font_2)
    label.set_fontsize(font_small)


# Boxplot for top-ashrae by scenario
ax1 = axes[1]
boxplot = filtered_data_t2_t3_outliers_removed_v2.boxplot(
    column='top',
    by='scenario',
    ax=ax1,
    patch_artist=True,
    return_type='axes',  # Suppress default title
    widths=0.6,
    boxprops=dict(facecolor='#2d2d2d', color='#2d2d2d'),
    medianprops=dict(color='white', linewidth=1),
    whiskerprops=dict(color='#2d2d2d', linewidth=1.5),
    capprops=dict(color='#2d2d2d', linewidth=.5),
    flierprops=dict(marker='o', color='#2d2d2d', markerfacecolor='#2d2d2d',
                    markeredgecolor='white', markeredgewidth=0.2, markersize=4)
)

plt.suptitle("")  # Clears the default title

# Add mean values to the boxplot in white font
means = filtered_data_t2_t3_outliers_removed_v2.groupby('scenario')['top'].mean()
for i, (scenario, mean) in enumerate(means.items()):
    if scenario == 'T2':  # Adjust position specifically for T2
        x_offset = 0.35  # Move text to the right for T2
        y_offset = -0.1  # Adjust y-offset as needed
        alignment = 'left'  # Align text to the left of the specified x-coordinate
    else:
        x_offset = -0.3  # Default position for other scenarios
        y_offset = -0.1
        alignment = 'right'  # Default alignment

    # Add the mean value as text
    ax1.text(
        i + 1 + x_offset,  # Adjusted x-coordinate
        mean + y_offset,   # Adjusted y-coordinate
        f"{mean:.2f}",
        ha=alignment,  # Horizontal alignment based on scenario
        va='center',  # Vertically align with the dot
        color='#2d2d2d',
        fontproperties=custom_font,
        fontsize=font_small
    )
    # Add the mean value as a small dot
    ax1.plot(
        i + 1, mean, 'x',  # Small dot marker
        color='white', markersize=3  # Adjust size and color of the dot
    )


ax1.set_ylim(ylim_t)  # Apply the new y-axis limits
ax1.grid(True, linestyle='--', linewidth=0.5)

# Remove top and right frame edges
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)

# Configure the boxplot
ax1.axhline(0.5, color='#2d2d2d', linestyle='-', linewidth=1, zorder=1)
ax1.yaxis.tick_right()  # Move y-axis to the right
ax1.yaxis.set_label_position("right")  # Label on the right side
ax1.set_title("", fontsize=font_larger, fontproperties=custom_font)
ax1.set_xlabel('Scenario', fontsize=font_larger, fontproperties=custom_font)
ax1.set_ylabel('Operative temperature (°C)', fontsize=font_larger, fontproperties=custom_font)

# Apply custom font to tick labels
for label in ax1.get_xticklabels():
    label.set_fontproperties(custom_font)
    label.set_fontsize(font_small)

for label in ax1.get_yticklabels():
    label.set_fontproperties(custom_font)
    label.set_fontsize(font_small)
    
# Save the figure
output_folder_path = ".\\Graphs\\Climate"  # Replace with your desired folder path
output_file_name = "top_boxplot_and_line"  # File name without extension

os.makedirs(output_folder_path, exist_ok=True)

jpeg_path = os.path.join(output_folder_path, f"{output_file_name}.jpeg")
pdf_path = os.path.join(output_folder_path, f"{output_file_name}.pdf")

fig.savefig(jpeg_path, format='jpeg', dpi=300)  
fig.savefig(pdf_path, format='pdf')  

jpeg_path, pdf_path


#%% tdb plots
# Create the figure
fig, axes = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={'width_ratios': [2.5, 1], 'wspace': 0.05})

# Line plot for top-ashrae over elapsed time
ax0 = axes[0]
for scenario, color in scenario_colors.items():
    scenario_data = binned_stats[binned_stats['scenario'] == scenario]
    ax0.errorbar(
        scenario_data['time-bin'],
        scenario_data[('tdb', 'mean')],
        yerr=scenario_data[('tdb', 'std')],
        label=scenario,
        fmt='o--',
        color=color,
        linewidth=1,
        markersize=4
    )

# Configure the line plot
ax0.axhline(0.5, color='#2d2d2d', linestyle='-', linewidth=1, zorder=1) 
ax0.set_xticks(range(0, 181, 30))
ax0.set_title("", fontsize=font_larger, fontproperties=custom_font)
ax0.set_xlabel('Elapsed time (minutes)', fontsize=font_larger, fontproperties=custom_font)
ax0.set_ylabel('Air temperature (°C)', fontsize=font_larger, fontproperties=custom_font)
ax0.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.1),
    ncol=4,
    frameon=False
)
ax0.set_ylim(ylim_t)  # Apply the new y-axis limits
ax0.grid(True, linestyle='--', linewidth=0.5)

# Remove top and right frame edges
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)

# Apply custom font to tick labels
for label in ax0.get_xticklabels():
    label.set_fontproperties(custom_font_2)
    label.set_fontsize(font_small)
for label in ax0.get_yticklabels():
    label.set_fontproperties(custom_font_2)
    label.set_fontsize(font_small)


# Boxplot for top-ashrae by scenario
ax1 = axes[1]
boxplot = filtered_data_t2_t3_outliers_removed_v2.boxplot(
    column='tdb',
    by='scenario',
    ax=ax1,
    patch_artist=True,
    return_type='axes',  # Suppress default title
    widths=0.6,
    boxprops=dict(facecolor='#2d2d2d', color='#2d2d2d'),
    medianprops=dict(color='white', linewidth=1),
    whiskerprops=dict(color='#2d2d2d', linewidth=1.5),
    capprops=dict(color='#2d2d2d', linewidth=.5),
    flierprops=dict(marker='o', color='#2d2d2d', markerfacecolor='#2d2d2d',
                    markeredgecolor='white', markeredgewidth=0.2, markersize=4)
)

plt.suptitle("")  # Clears the default title

# Add mean values to the boxplot in white font
means = filtered_data_t2_t3_outliers_removed_v2.groupby('scenario')['tdb'].mean()
for i, (scenario, mean) in enumerate(means.items()):
    if scenario == 'T2':  # Adjust position specifically for T2
        x_offset = 0.35  # Move text to the right for T2
        y_offset = -0.1  # Adjust y-offset as needed
        alignment = 'left'  # Align text to the left of the specified x-coordinate
    else:
        x_offset = -0.3  # Default position for other scenarios
        y_offset = -0.1
        alignment = 'right'  # Default alignment

    # Add the mean value as text
    ax1.text(
        i + 1 + x_offset,  # Adjusted x-coordinate
        mean + y_offset,   # Adjusted y-coordinate
        f"{mean:.2f}",
        ha=alignment,  # Horizontal alignment based on scenario
        va='center',  # Vertically align with the dot
        color='#2d2d2d',
        fontproperties=custom_font,
        fontsize=font_small
    )
    # Add the mean value as a small dot
    ax1.plot(
        i + 1, mean, 'x',  # Small dot marker
        color='white', markersize=3  # Adjust size and color of the dot
    )

ax1.set_ylim(ylim_t)  # Apply the new y-axis limits
ax1.grid(True, linestyle='--', linewidth=0.5)

# Remove top and right frame edges
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)

# Configure the boxplot
ax1.axhline(0.5, color='#2d2d2d', linestyle='-', linewidth=1, zorder=1)
ax1.yaxis.tick_right()  # Move y-axis to the right
ax1.yaxis.set_label_position("right")  # Label on the right side
ax1.set_title("", fontsize=font_larger, fontproperties=custom_font)
ax1.set_xlabel('Scenario', fontsize=font_larger, fontproperties=custom_font)
ax1.set_ylabel('Air temperature (°C)', fontsize=font_larger, fontproperties=custom_font)

# Apply custom font to tick labels
for label in ax1.get_xticklabels():
    label.set_fontproperties(custom_font)
    label.set_fontsize(font_small)

for label in ax1.get_yticklabels():
    label.set_fontproperties(custom_font)
    label.set_fontsize(font_small)
    
# Save the figure
output_folder_path = ".\\Graphs\\Climate"  # Replace with your desired folder path
output_file_name = "tdb_boxplot_and_line"  # File name without extension

os.makedirs(output_folder_path, exist_ok=True)

jpeg_path = os.path.join(output_folder_path, f"{output_file_name}.jpeg")
pdf_path = os.path.join(output_folder_path, f"{output_file_name}.pdf")

fig.savefig(jpeg_path, format='jpeg', dpi=300)  
fig.savefig(pdf_path, format='pdf')  

jpeg_path, pdf_path



#%% tr plots
# Create the figure
fig, axes = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={'width_ratios': [2.5, 1], 'wspace': 0.05})

# Line plot for top-ashrae over elapsed time
ax0 = axes[0]
for scenario, color in scenario_colors.items():
    scenario_data = binned_stats[binned_stats['scenario'] == scenario]
    ax0.errorbar(
        scenario_data['time-bin'],
        scenario_data[('mrt', 'mean')],
        yerr=scenario_data[('mrt', 'std')],
        label=scenario,
        fmt='o--',
        color=color,
        linewidth=1,
        markersize=4
    )

# Configure the line plot
ax0.axhline(0.5, color='#2d2d2d', linestyle='-', linewidth=1, zorder=1) 
ax0.set_xticks(range(0, 181, 30))
ax0.set_title("", fontsize=font_larger, fontproperties=custom_font)
ax0.set_xlabel('Elapsed time (minutes)', fontsize=font_larger, fontproperties=custom_font)
ax0.set_ylabel('Mean radient temperature (°C)', fontsize=font_larger, fontproperties=custom_font)
ax0.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.1),
    ncol=4,
    frameon=False
)
ax0.set_ylim(ylim_t)  # Apply the new y-axis limits
ax0.grid(True, linestyle='--', linewidth=0.5)

# Remove top and right frame edges
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)

# Apply custom font to tick labels
for label in ax0.get_xticklabels():
    label.set_fontproperties(custom_font_2)
    label.set_fontsize(font_small)
for label in ax0.get_yticklabels():
    label.set_fontproperties(custom_font_2)
    label.set_fontsize(font_small)


# Boxplot for top-ashrae by scenario
ax1 = axes[1]
boxplot = filtered_data_t2_t3_outliers_removed_v2.boxplot(
    column='mrt',
    by='scenario',
    ax=ax1,
    patch_artist=True,
    return_type='axes',  # Suppress default title
    widths=0.6,
    boxprops=dict(facecolor='#2d2d2d', color='#2d2d2d'),
    medianprops=dict(color='white', linewidth=1),
    whiskerprops=dict(color='#2d2d2d', linewidth=1.5),
    capprops=dict(color='#2d2d2d', linewidth=.5),
    flierprops=dict(marker='o', color='#2d2d2d', markerfacecolor='#2d2d2d',
                    markeredgecolor='white', markeredgewidth=0.2, markersize=4)
)

plt.suptitle("")  # Clears the default title

# Add mean values to the boxplot in white font
means = filtered_data_t2_t3_outliers_removed_v2.groupby('scenario')['mrt'].mean()
for i, (scenario, mean) in enumerate(means.items()):
    if scenario == 'T2':  # Adjust position specifically for T2
        x_offset = 0.35  # Move text to the right for T2
        y_offset = -0.1  # Adjust y-offset as needed
        alignment = 'left'  # Align text to the left of the specified x-coordinate
    else:
        x_offset = -0.3  # Default position for other scenarios
        y_offset = -0.1
        alignment = 'right'  # Default alignment

    # Add the mean value as text
    ax1.text(
        i + 1 + x_offset,  # Adjusted x-coordinate
        mean + y_offset,   # Adjusted y-coordinate
        f"{mean:.2f}",
        ha=alignment,  # Horizontal alignment based on scenario
        va='center',  # Vertically align with the dot
        color='#2d2d2d',
        fontproperties=custom_font,
        fontsize=font_small
    )
    # Add the mean value as a small dot
    ax1.plot(
        i + 1, mean, 'x',  # Small dot marker
        color='white', markersize=3  # Adjust size and color of the dot
    )

ax1.set_ylim(ylim_t)  # Apply the new y-axis limits
ax1.grid(True, linestyle='--', linewidth=0.5)

# Remove top and right frame edges
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)

# Configure the boxplot
ax1.axhline(0.5, color='#2d2d2d', linestyle='-', linewidth=1, zorder=1)
ax1.yaxis.tick_right()  # Move y-axis to the right
ax1.yaxis.set_label_position("right")  # Label on the right side
ax1.set_title("", fontsize=font_larger, fontproperties=custom_font)
ax1.set_xlabel('Scenario', fontsize=font_larger, fontproperties=custom_font)
ax1.set_ylabel('Mean radient temperature (°C)', fontsize=font_larger, fontproperties=custom_font)

# Apply custom font to tick labels
for label in ax1.get_xticklabels():
    label.set_fontproperties(custom_font)
    label.set_fontsize(font_small)

for label in ax1.get_yticklabels():
    label.set_fontproperties(custom_font)
    label.set_fontsize(font_small)
    
# Save the figure
output_folder_path = ".\\Graphs\\Climate"  # Replace with your desired folder path
output_file_name = "tr-ashrae_boxplot_and_line"  # File name without extension

os.makedirs(output_folder_path, exist_ok=True)

jpeg_path = os.path.join(output_folder_path, f"{output_file_name}.jpeg")
pdf_path = os.path.join(output_folder_path, f"{output_file_name}.pdf")

fig.savefig(jpeg_path, format='jpeg', dpi=300)  
fig.savefig(pdf_path, format='pdf')  

jpeg_path, pdf_path


#%% rh plots
# Create the figure
fig, axes = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={'width_ratios': [2.5, 1], 'wspace': 0.05})

# Line plot for top-ashrae over elapsed time
ax0 = axes[0]
for scenario, color in scenario_colors.items():
    scenario_data = binned_stats[binned_stats['scenario'] == scenario]
    ax0.errorbar(
        scenario_data['time-bin'],
        scenario_data[('rh', 'mean')],
        yerr=scenario_data[('rh', 'std')],
        label=scenario,
        fmt='o--',
        color=color,
        linewidth=1,
        markersize=4
    )

# Configure the line plot
ax0.axhline(0.5, color='#2d2d2d', linestyle='-', linewidth=1, zorder=1) 
ax0.set_xticks(range(0, 181, 30))
ax0.set_title("", fontsize=font_larger, fontproperties=custom_font)
ax0.set_xlabel('Elapsed time (minutes)', fontsize=font_larger, fontproperties=custom_font)
ax0.set_ylabel('Relative humidity (%)', fontsize=font_larger, fontproperties=custom_font)
ax0.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.1),
    ncol=4,
    frameon=False
)
ax0.set_ylim(ylim_rh)  # Apply the new y-axis limits
ax0.grid(True, linestyle='--', linewidth=0.5)

# Remove top and right frame edges
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)

# Apply custom font to tick labels
for label in ax0.get_xticklabels():
    label.set_fontproperties(custom_font_2)
    label.set_fontsize(font_small)
for label in ax0.get_yticklabels():
    label.set_fontproperties(custom_font_2)
    label.set_fontsize(font_small)


# Boxplot for top-ashrae by scenario
ax1 = axes[1]
boxplot = filtered_data_t2_t3_outliers_removed_v2.boxplot(
    column='rh',
    by='scenario',
    ax=ax1,
    patch_artist=True,
    return_type='axes',  # Suppress default title
    widths=0.6,
    boxprops=dict(facecolor='#2d2d2d', color='#2d2d2d'),
    medianprops=dict(color='white', linewidth=1),
    whiskerprops=dict(color='#2d2d2d', linewidth=1.5),
    capprops=dict(color='#2d2d2d', linewidth=.5),
    flierprops=dict(marker='o', color='#2d2d2d', markerfacecolor='#2d2d2d',
                    markeredgecolor='white', markeredgewidth=0.2, markersize=4)
)

plt.suptitle("")  # Clears the default title

means = filtered_data_t2_t3_outliers_removed_v2.groupby('scenario')['rh'].mean()
for i, (scenario, mean) in enumerate(means.items()):
    # Adjust position for a specific scenario (e.g., 'T2')
    if scenario == 'T2':
        vertical_offset = 0.7  # Adjust the vertical offset for 'T2'
    else:
        vertical_offset = -1  # Default vertical offset

    # Add the mean value as text on the right of the dot
    ax1.text(
        i + 1, 
        mean + vertical_offset,  # Use adjusted vertical offset
        f"{mean:.2f}",
        ha='center',  # Align text to the left of the specified x-coordinate
        va='center',  # Vertically align with the dot
        color='white',
        fontproperties=custom_font,
        fontsize=font_small
    )
    # Add the mean value as a small dot
    ax1.plot(
        i + 1, mean, 'x',  # Small dot marker
        color='white', markersize=3  # Adjust size and color of the dot
    )


ax1.set_ylim(ylim_rh)  # Apply the new y-axis limits
ax1.grid(True, linestyle='--', linewidth=0.5)

# Remove top and right frame edges
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)

# Configure the boxplot
ax1.axhline(0.5, color='#2d2d2d', linestyle='-', linewidth=1, zorder=1)
ax1.yaxis.tick_right()  # Move y-axis to the right
ax1.yaxis.set_label_position("right")  # Label on the right side
ax1.set_title("", fontsize=font_larger, fontproperties=custom_font)
ax1.set_xlabel('Scenario', fontsize=font_larger, fontproperties=custom_font)
ax1.set_ylabel('Relative humidity (%)', fontsize=font_larger, fontproperties=custom_font)

# Apply custom font to tick labels
for label in ax1.get_xticklabels():
    label.set_fontproperties(custom_font)
    label.set_fontsize(font_small)

for label in ax1.get_yticklabels():
    label.set_fontproperties(custom_font)
    label.set_fontsize(font_small)
    
# Save the figure
output_folder_path = ".\\Graphs\\Climate"  # Replace with your desired folder path
output_file_name = "rh_boxplot_and_line"  # File name without extension

os.makedirs(output_folder_path, exist_ok=True)

jpeg_path = os.path.join(output_folder_path, f"{output_file_name}.jpeg")
pdf_path = os.path.join(output_folder_path, f"{output_file_name}.pdf")

fig.savefig(jpeg_path, format='jpeg', dpi=300)  
fig.savefig(pdf_path, format='pdf')  

jpeg_path, pdf_path
