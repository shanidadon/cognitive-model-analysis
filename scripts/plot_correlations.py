import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the summary data
summary_df = pd.read_csv('/home/shani_dadon/Documents/shani-s-project-/results/correlations/summary.csv')

# Map categorization models to their groups
group_mapping = {
    'CIFAR10DecisionBoundaryModel': 'Decision Bound',
    'CIFAR10KDE': 'Exemplar',
    'CIFAR10LDA': 'Prototype'
}

summary_df['group'] = summary_df['categorization_model'].map(group_mapping)

# Get unique groups
groups = ['Prototype', 'Exemplar', 'Decision Bound']

# Define models ordered by family and their colors
models = [
    # CLIP-based models (Blue shades, dark to light)
    'Clip_ViT_Large',
    'FAIR_SLIP_ViT_Large_CLIP_25',
    'FAIR_SLIP_ViT_Large_SLIP_25',
    'SigLIP2_Base_P16_224',
    
    # ImageNet classifiers (Green shades, dark to light)
    'Google_ViT_Large_224',
    'ConvNext_Large',
    'Inception_V3',
    'CORnet_RT',
    
    # Self-supervised / Other (Brown to Orange, dark to light)
    'DINOv2_ViT_Large',
    'FAIR_SLIP_ViT_Large_SimCLR_25',
]

model_colors = {
    # CLIP-based models (Blue shades, dark to light)
    'Clip_ViT_Large': '#1f4788',              # Original CLIP - darkest blue
    'FAIR_SLIP_ViT_Large_CLIP_25': '#4a6fa5', # Meta CLIP - medium blue
    'FAIR_SLIP_ViT_Large_SLIP_25': '#7a9bc4', # SLIP - lighter blue
    'SigLIP2_Base_P16_224': '#a8c5e8',        # SigLIP - lightest blue
    
    # ImageNet classifiers (Green shades, dark to light)
    'Google_ViT_Large_224': '#2d5a2d',        # ImageNet classifier - darkest green
    'ConvNext_Large': '#4a8f4a',              # ConvNeXt - medium green
    'Inception_V3': '#7fb77f',                # Inception V3 - lighter green
    'CORnet_RT': '#b8e0b8',                   # CORnet - supervised classifier, lightest green
    
    # Self-supervised / Other (Brown to Orange, dark to light)
    'DINOv2_ViT_Large': '#8b4513',            # DINOv2 - dark brown
    'FAIR_SLIP_ViT_Large_SimCLR_25': '#d2691e', # SimCLR - lighter orange-brown
}

# Prepare data for plotting
fig, ax = plt.subplots(figsize=(14, 8))

# Bar width and positions
bar_width = 0.08
x_positions = np.arange(len(groups))

# Plot bars for each model
for i, model in enumerate(models):
    model_data = summary_df[summary_df['nn_model'] == model]
    values = []
    
    for group in groups:
        row = model_data[model_data['group'] == group]
        if len(row) > 0:
            values.append(row['spearman_rho'].values[0])
        else:
            values.append(0)
    
    # Offset bars for each model
    offset = (i - len(models)/2 + 0.5) * bar_width
    ax.bar(x_positions + offset, values, bar_width, 
           label=model, color=model_colors[model], alpha=0.8)

# Customize the plot
ax.set_xlabel('Categorization Model Group', fontsize=16, fontweight='bold')
ax.set_ylabel('Spearman Correlation Coefficient (ρ)', fontsize=16, fontweight='bold')
ax.set_xticks(x_positions)
ax.set_xticklabels(groups)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/shani_dadon/Documents/shani-s-project-/results/plots/correlations_by_model.png', 
            dpi=300, bbox_inches='tight')
print("Graph saved to: /home/shani_dadon/Documents/shani-s-project-/results/plots/correlations_by_model.png")
plt.show()
