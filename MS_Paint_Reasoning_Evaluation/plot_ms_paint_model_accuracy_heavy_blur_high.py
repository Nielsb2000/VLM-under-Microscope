### Script to plot the accuracy of GPT-5.1 and GPT-5.2 for both heavy blur and heavy blur high conditions in a single bar plot for comparison, for the presentation. This is a temporary script and can be deleted after use.

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Paths to res.txt files
heavy_blur_file = "MS_Paint_Reasoning_Evaluation/heavy_blur_res.txt"
heavy_blur_high_file = "MS_Paint_Reasoning_Evaluation/heavy_blur_high_res.txt"

models = ["GPT-5.1", "GPT-5.2"]  # Only plot these

# Helper to compute accuracy per model
def compute_accuracies(res_file, models):
    df = pd.read_csv(res_file)
    df.columns = [c.strip() for c in df.columns]
    df = df.replace('-', np.nan)
    accs = []
    for model in models:
        col = model if model in df.columns else model.lower()
        valid = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(valid) > 0:
            acc = valid.sum() / len(valid)
        else:
            acc = np.nan
        accs.append(acc)
    return accs

# Compute accuracies
acc_heavy_blur = compute_accuracies(heavy_blur_file, models)
acc_heavy_blur_high = compute_accuracies(heavy_blur_high_file, models)

# Prepare plot

labels = [f"{model}\nHeavy Blur" for model in models] + [f"{model}\nHeavy Blur High Reasoning" for model in models]
values = acc_heavy_blur + acc_heavy_blur_high
colors = ['#ff7f0e', '#2ca02c', '#ff7f0e', '#2ca02c']


plt.figure(figsize=(8, 6))
bars = plt.bar(labels, values, color=colors, alpha=0.85)
plt.title("MS Paint Model Accuracy: Heavy Blur & Reasoning High")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height*100:.1f}%", ha='center')

# Add dotted red line exactly between Heavy Blur GPT-5.2 and Heavy Blur High Reasoning GPT-5.1
plt.axvline(x=1.5, color='red', linestyle=':', linewidth=2)

plt.tight_layout()
output_path = "MS_Paint_Reasoning_Evaluation/Results/res_vis/ms_paint_model_accuracy_heavy_blur_high.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
plt.show()
