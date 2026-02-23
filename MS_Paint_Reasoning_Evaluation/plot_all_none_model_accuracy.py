#### Temrporary script to plot overall accuracy of all models and blur levels in one bar plot with dotted red lines between blur levels. This is for the final presentation

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

res_files = {
    'none': 'MS_Paint_Reasoning_Evaluation/none_res.txt',
    'med_blur': 'MS_Paint_Reasoning_Evaluation/med_blur_res.txt',
    'heavy_blur': 'MS_Paint_Reasoning_Evaluation/heavy_blur_res.txt',
}
models = ["GPT-4o", "GPT-5.1", "GPT-5.2"]
blur_levels = list(res_files.keys())

# Calculate overall accuracy per model and blur level
accuracies = []
for blur in blur_levels:
    df = pd.read_csv(res_files[blur])
    df.columns = [c.strip() for c in df.columns]
    df = df.replace('-', np.nan)
    for model in models:
        col = model if model in df.columns else model.lower()
        valid = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(valid) > 0:
            acc = valid.sum() / len(valid)
        else:
            acc = np.nan
        accuracies.append((blur, model, acc))

# Prepare bar plot: x-axis is (blur, model), y-axis is accuracy
labels = [f"{model}\n({blur})" for blur, model, _ in accuracies]
values = [acc for _, _, acc in accuracies]


plt.figure(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] * len(blur_levels)
bars = plt.bar(labels, values, color=colors[:len(labels)], alpha=0.85)
plt.title("Overall Model Accuracy by Blur Level")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height*100:.1f}%", ha='center')

# Add dotted red lines between blur levels
num_blur = len(blur_levels)
num_models = len(models)
for b in range(1, num_blur):
    xpos = b * num_models - 0.5
    plt.axvline(x=xpos, color='red', linestyle=':', linewidth=2)

plt.tight_layout()
output_path = "MS_Paint_Reasoning_Evaluation/Results/res_vis/model_accuracy_comparison_all_blur.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
plt.show()
