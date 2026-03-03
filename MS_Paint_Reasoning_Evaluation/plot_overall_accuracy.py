import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Plot overall model accuracy for MS Paint Reasoning with flexible CLI.")
    parser.add_argument('--image-type', default='original', choices=['original', 'greyscale', 'inverted_greyscale'],
                        help="Image type to plot. Default: original.")
    parser.add_argument('--blur-levels', nargs='+', default=['no_blur', 'med_blur', 'heavy_blur'],
                        help="Blur levels to include. Default: all.")
    parser.add_argument('--reasoning-modes', nargs='+', default=None,
                        help="Reasoning modes to use for each blur level (same order as --blur-levels). If not set, uses --reasoning-mode for all.")
    parser.add_argument('--reasoning-mode', default='none', choices=['none', 'low', 'medium', 'high'],
                        help="(Legacy) Reasoning mode to plot for all blur levels. Default: none.")
    parser.add_argument('--models', nargs='+', default=None,
                        help="Models to plot (default: all in results file).")
    parser.add_argument('--output-dir', default=None,
                        help="Directory to save plots. Default: Results/res_vis/<image_type>_<blur_level>_<reasoning_mode>/")
    return parser.parse_args()


def find_results_file(image_type, blur_level, reasoning_mode):
    # Always include reasoning_mode in both folder and filename, even for 'none'
    base = f"MS_Paint_Reasoning_Evaluation/Results/{image_type}_{blur_level}_{reasoning_mode}"
    fname = f"{image_type}_{blur_level}_{reasoning_mode}"
    return f"{base}/{fname}_res.txt"

def main():
    args = parse_args()
    accuracies = []
    all_models = set()
    # Determine reasoning mode per blur
    if args.reasoning_modes:
        if len(args.reasoning_modes) != len(args.blur_levels):
            raise ValueError("If --reasoning-modes is set, it must have the same number of entries as --blur-levels.")
        blur_reasoning = list(zip(args.blur_levels, args.reasoning_modes))
    else:
        blur_reasoning = [(blur, args.reasoning_mode) for blur in args.blur_levels]

    for blur, reasoning_mode in blur_reasoning:
        res_file = find_results_file(args.image_type, blur, reasoning_mode)
        # Try both _res.txt and .txt endings
        if not os.path.exists(res_file):
            alt_file = res_file.replace('_res.txt', '.txt')
            if os.path.exists(alt_file):
                res_file = alt_file
            else:
                print(f"[WARN] Results file not found: {res_file}")
                continue
        df = pd.read_csv(res_file)
        df.columns = [c.strip() for c in df.columns]
        df = df.replace('-', np.nan)
        # Use all model columns if not specified
        model_cols = args.models if args.models else [c for c in df.columns if c.lower().startswith('gpt') or c.lower() == 'gpt-image-1.5']
        all_models.update(model_cols)
        for model in model_cols:
            if model not in df.columns:
                continue
            valid = pd.to_numeric(df[model], errors='coerce').dropna()
            acc = valid.sum() / len(valid) if len(valid) > 0 else np.nan
            accuracies.append((f"{blur} ({reasoning_mode})", model, acc))
    if not accuracies:
        print("No accuracy data found.")
        return
    # Prepare plot
    labels = [f"{model}\n({blur})" for blur, model, _ in accuracies]
    values = [acc for _, _, acc in accuracies]
    # Assign a unique color per model (repeatable for each blur)
    unique_models = list(dict.fromkeys([model for _, model, _ in accuracies]))
    color_map = plt.get_cmap('tab10')
    model_colors = {model: color_map(i % 10) for i, model in enumerate(unique_models)}
    bar_colors = [model_colors[model] for _, model, _ in accuracies]

    # Increase width and font size for legibility
    plt.figure(figsize=(max(12, len(labels)*1.2), 7))
    bars = plt.bar(labels, values, color=bar_colors, alpha=0.85)
    plt.title(f"Overall Model Accuracy by Blur Level\n({args.image_type}, reasoning: {args.reasoning_mode})", fontsize=18)
    plt.xlabel("Model (Blur Level)", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.ylim(0, 1)
    plt.xticks(fontsize=13, rotation=25, ha='right')
    plt.yticks(fontsize=13)

    # Add legend for models
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=model_colors[model], label=model) for model in unique_models]
    plt.legend(handles=legend_handles, title="Model", loc="best", fontsize=13, title_fontsize=14)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height*100:.1f}%", ha='center', fontsize=13)
    # Add dotted red lines between blur levels
    num_blur = len(args.blur_levels)
    num_models = len(all_models)
    for b in range(1, num_blur):
        xpos = b * num_models - 0.5
        plt.axvline(x=xpos, color='red', linestyle=':', linewidth=2)
    plt.tight_layout()
    # Output dir
    # Output dir reflects all blur/reasoning combos
    blur_reasoning_part = '_'.join([f"{b}-{r}" for b, r in blur_reasoning])
    out_dir = args.output_dir or f"MS_Paint_Reasoning_Evaluation/Results/res_vis/{args.image_type}_{blur_reasoning_part}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model_accuracy_comparison.png")
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")
    plt.show()

if __name__ == "__main__":
    main()
