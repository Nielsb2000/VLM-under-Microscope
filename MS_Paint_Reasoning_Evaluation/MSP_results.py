# MS Paint Reasoning Results Visualization (from res.txt)

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np




import os

def main():
    parser = argparse.ArgumentParser(description="Visualize MS Paint Reasoning results for a specific blur level and reasoning mode.")
    parser.add_argument('--blur-level', default='none', choices=['none', 'med_blur', 'heavy_blur'],
                        help="Which blur level to visualize. Choices: none, med_blur, heavy_blur. Default: none.")
    parser.add_argument('--reasoning-mode', default='none', choices=['none', 'low', 'medium', 'high'],
                        help="Reasoning mode to visualize. Choices: none, low, medium, high. Default: none.")
    parser.add_argument('--res-file', default=None, help="Path to the results .txt file. If not set, will use default for blur level and reasoning mode.")
    args = parser.parse_args()

    # Clean output folder naming: if reasoning-mode is 'none', just use blur-level as folder, else use blur_reasoning
    if args.reasoning_mode == 'none':
        res_file = args.res_file or f'MS_Paint_Reasoning_Evaluation/{args.blur_level}_res.txt'
        res_vis_dir = f'MS_Paint_Reasoning_Evaluation/Results/res_vis/{args.blur_level}'
    else:
        res_file = args.res_file or f'MS_Paint_Reasoning_Evaluation/{args.blur_level}_{args.reasoning_mode}_res.txt'
        res_vis_dir = f'MS_Paint_Reasoning_Evaluation/Results/res_vis/{args.blur_level}_{args.reasoning_mode}'
    os.makedirs(res_vis_dir, exist_ok=True)

    # Load results from res.txt
    df = pd.read_csv(res_file)


    # Clean up column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Convert '-' to np.nan for missing answers
    df = df.replace('-', np.nan)

    # Use actual model columns from the DataFrame, including gpt-image-1.5 if present
    model_cols = [c for c in df.columns if c.lower().startswith('gpt') or c.lower() == 'gpt-image-1.5']

    # Convert to int where possible
    for col in model_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate accuracy for each model
    accuracies = {}
    for col in model_cols:
        valid = df[col].dropna()
        if len(valid) > 0:
            accuracies[col] = valid.sum() / len(valid)
        else:
            accuracies[col] = 0

    # Plot accuracy
    plt.figure(figsize=(8, 5))
    colors = ['blue', 'orange', 'green'][:len(model_cols)]
    plt.bar(accuracies.keys(), accuracies.values(), color=colors)
    plt.ylabel('Accuracy')
    if args.reasoning_mode != 'none':
        plt.title(f'MS Paint Model Accuracy ({args.blur_level}, reasoning: {args.reasoning_mode})')
    else:
        plt.title(f'MS Paint Model Accuracy ({args.blur_level})')
    plt.ylim(0, 1)
    for i, m in enumerate(accuracies):
        plt.text(i, accuracies[m]+0.02, f"{accuracies[m]*100:.1f}%", ha='center')
    plt.tight_layout()
    plt.savefig(f'{res_vis_dir}/model_accuracy_plot.png')
    plt.show()

    # --- Simple heatmap: one per model, rows=images, cols=questions (max found), missing=black ---
    img_list = sorted(df['IMG'].dropna().unique(), key=lambda x: int(x))
    q_list = sorted(df['Question'].dropna().unique(), key=lambda x: int(x))
    max_q = max(df.groupby('IMG')['Question'].nunique())
    for col in model_cols:
        heatmap = np.full((len(img_list), max_q), np.nan)
        for i, img in enumerate(img_list):
            img_df = df[df['IMG'] == img]
            img_qs = sorted(img_df['Question'].dropna().unique(), key=lambda x: int(x))
            for j, q in enumerate(img_qs):
                val = img_df[img_df['Question'] == q][col]
                if not val.empty:
                    heatmap[i, j] = val.values[0]
        plt.figure(figsize=(max_q+2, len(img_list)))
        masked = np.ma.masked_invalid(heatmap)
        cmap = plt.cm.Greens
        cmap.set_bad(color='black')
        plt.imshow(masked, cmap=cmap, vmin=0, vmax=1)
        if args.reasoning_mode != 'none':
            plt.title(f'{col} Correctness ({args.blur_level}, reasoning: {args.reasoning_mode})')
        else:
            plt.title(f'{col} Correctness ({args.blur_level})')
        plt.xlabel('Question #')
        plt.ylabel('Image #')
        plt.xticks(np.arange(max_q), [str(i+1) for i in range(max_q)])
        plt.yticks(np.arange(len(img_list)), img_list)
        for i in range(len(img_list)):
            for j in range(max_q):
                val = heatmap[i, j]
                if not np.isnan(val):
                    plt.text(j, i, str(int(val)), ha='center', va='center', color='black')
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', edgecolor='black', label='Correct (1)'),
            Patch(facecolor='white', edgecolor='black', label='Incorrect (0)'),
            Patch(facecolor='black', edgecolor='black', label='No answer')
        ]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'{res_vis_dir}/{col}_correctness_heatmap.png')
        plt.show()

if __name__ == "__main__":
    main()
