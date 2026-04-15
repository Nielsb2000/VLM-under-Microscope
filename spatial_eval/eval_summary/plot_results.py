import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito model colors
MODEL_COLORS = {
    "gpt-4o":  "#56B4E9",
    "gpt-5.1": "#E69F00",
    "gpt-5.2": "#009E73",
}


TASKS = ["mazenav", "spatialmap", "spatialgrid"]#, "spatialreal"]
MODES = ["vqa", "vtqa"]


def read_accuracy_csv(path: str):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    # Expecting a model name column and one numeric accuracy column
    if df.empty:
        return None

    # Heuristic: first column is model name, find first numeric column
    cols = df.columns.tolist()
    model_col = cols[0]
    acc_col = None
    for c in cols[1:]:
        if pd.api.types.is_numeric_dtype(df[c]):
            acc_col = c
            break
        # sometimes accuracy stored as float in string, try convert
        try:
            tmp = pd.to_numeric(df[c], errors='coerce')
            if tmp.notna().sum() > 0:
                df[c] = tmp
                acc_col = c
                break
        except Exception:
            continue

    if acc_col is None:
        return None

    # Return dataframe with model and accuracy as float (0-1 or 0-100)
    models = df[model_col].astype(str).tolist()
    accs = df[acc_col].astype(float).tolist()

    # If values are in 0-1 range, convert to 0-100
    if max(accs) <= 1.0:
        accs = [a * 100.0 for a in accs]

    return models, accs


def plot_bar(models, accs, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    colors = [MODEL_COLORS.get(m, "#555555") for m in models]
    bars = ax.bar(x, accs, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, 110)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    # Annotate bars
    for bar, acc in zip(bars, accs):
        h = bar.get_height()
        ax.annotate(f'{acc:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(base_dir, 'result_vis')
    os.makedirs(result_dir, exist_ok=True)

    summary_dir = base_dir


    created = []
    for mode in MODES:
        for task in TASKS:
            csv_path = os.path.join(summary_dir, mode, f"{task}_acc.csv")
            if not os.path.isfile(csv_path):
                print(f"Summary CSV not found: {csv_path}, skipping")
                continue

            data = read_accuracy_csv(csv_path)
            if data is None:
                print(f"Could not parse CSV: {csv_path}, skipping")
                continue

            models, accs = data
            # Show all models present in the CSV
            filtered = list(zip(models, accs))
            if not filtered:
                print(f"No gpt-4o or gpt-5.1 results in {csv_path}, skipping plot.")
                continue
            models_filt, accs_filt = zip(*filtered)
            title = f"{task.capitalize()} — {mode.upper()} Accuracy by Model"
            out_filename = f"{task}_{mode}_results.png"
            out_path = os.path.join(result_dir, out_filename)
            plot_bar(models_filt, accs_filt, title, out_path)
            created.append(out_path)
            print(f"Created plot: {out_path}")

    if created:
        print(f"Created {len(created)} plots in {result_dir}")
    else:
        print("No plots were created.")


if __name__ == '__main__':
    main()
