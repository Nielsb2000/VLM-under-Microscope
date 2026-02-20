
import os
import re
import ast
import matplotlib.pyplot as plt
import numpy as np

# Minimal imports
import os
import re
import ast
import matplotlib.pyplot as plt
import numpy as np
import argparse

def extract_token_dict(text):
    # Extract the token usage dictionary from answer file text
    for line in text.splitlines():
        if line.strip().startswith('Token usage:'):
            return line.split('Token usage:')[1].strip()
    return None

def main():
    parser = argparse.ArgumentParser(description="Plot token/time stats for MS Paint Reasoning, supporting blur levels and reasoning modes.")
    parser.add_argument('--blur-level', default='none', choices=['none', 'med_blur', 'heavy_blur'],
                        help="Which blur level to plot. Choices: none, med_blur, heavy_blur. Default: none.")
    parser.add_argument('--reasoning-mode', default='none', choices=['none', 'low', 'medium', 'high'],
                        help="Reasoning mode to plot. Choices: none, low, medium, high. Default: none.")
    args = parser.parse_args()

    # Clean output folder naming: if reasoning-mode is 'none', just use blur-level as folder, else use blur_reasoning
    if args.reasoning_mode == 'none':
        results_dir = os.path.join("MS_Paint_Reasoning_Evaluation/Results", f"{args.blur_level}")
        plot_dir = os.path.join("MS_Paint_Reasoning_Evaluation/Results", "res_vis", f"{args.blur_level}")
    else:
        results_dir = os.path.join("MS_Paint_Reasoning_Evaluation/Results", f"{args.blur_level}_{args.reasoning_mode}")
        plot_dir = os.path.join("MS_Paint_Reasoning_Evaluation/Results", "res_vis", f"{args.blur_level}_{args.reasoning_mode}")
    os.makedirs(plot_dir, exist_ok=True)

    TIME_RE = re.compile(r"Elapsed time:\s*([0-9.]+) seconds")

    # Dynamically detect present models from answer files
    present_models = set()
    for img_name in sorted(os.listdir(results_dir)):
        img_path = os.path.join(results_dir, img_name)
        if not os.path.isdir(img_path):
            continue
        for q_name in sorted(os.listdir(img_path)):
            q_path = os.path.join(img_path, q_name)
            if not os.path.isdir(q_path):
                continue
            for fname in os.listdir(q_path):
                if fname.startswith("answer_") and fname.endswith(".txt"):
                    model = fname[len("answer_"):-4]
                    present_models.add(model)
    MODELS = sorted(present_models)

    # Gather stats from all answer files
    stats = []  # (img, q, model, input_tokens, output_tokens, total_tokens, elapsed_time)
    for img_name in sorted(os.listdir(results_dir)):
        img_path = os.path.join(results_dir, img_name)
        if not os.path.isdir(img_path):
            continue
        for q_name in sorted(os.listdir(img_path)):
            q_path = os.path.join(img_path, q_name)
            if not os.path.isdir(q_path):
                continue
            for model in MODELS:
                ans_file = os.path.join(q_path, f"answer_{model}.txt")
                if not os.path.exists(ans_file):
                    continue
                with open(ans_file, "r") as f:
                    content = f.read()
                token_dict_str = extract_token_dict(content)
                time_match = TIME_RE.search(content)
                if not token_dict_str or not time_match:
                    print(f"Warning: Could not extract stats from {ans_file}")
                    continue
                try:
                    token_dict = ast.literal_eval(token_dict_str)
                    input_tokens = token_dict.get("input_tokens", 0)
                    output_tokens = token_dict.get("output_tokens", 0)
                    total_tokens = token_dict.get("total_tokens", 0)
                    elapsed_time = float(time_match.group(1))
                    stats.append((img_name, q_name, model, input_tokens, output_tokens, total_tokens, elapsed_time))
                except Exception as e:
                    print(f"Parse error in {ans_file}: {e}")

    if not stats:
        print("No stats found!")
        return

    # Organize stats for plotting
    imgq_set = sorted(set((img, q) for img, q, _, _, _, _, _ in stats))
    imgq_idx = {label: i for i, label in enumerate(imgq_set)}

    tokens_matrix = np.zeros((len(imgq_set), len(MODELS)))
    time_matrix = np.zeros((len(imgq_set), len(MODELS)))

    for img, q, model, _, _, total_tokens, elapsed_time in stats:
        row = imgq_idx[(img, q)]
        col = MODELS.index(model)
        tokens_matrix[row, col] = total_tokens
        time_matrix[row, col] = elapsed_time

    # Pricing and cost calculation (Feb 2026, per 1M tokens, in USD)
    PRICING = {
        "gpt-4o": {"input": 5.00/1e6, "output": 15.00/1e6},
        "gpt-5.1": {"input": 10.00/1e6, "output": 30.00/1e6},
        "gpt-5.2": {"input": 12.00/1e6, "output": 36.00/1e6},
    }
    USD_TO_EUR = 0.92

    # Compute cost matrix in EUR
    cost_matrix = np.zeros((len(imgq_set), len(MODELS)))
    for img, q, model, input_tokens, output_tokens, total_tokens, elapsed_time in stats:
        row = imgq_idx[(img, q)]
        col = MODELS.index(model)
        price = PRICING[model]
        cost_usd = input_tokens * price["input"] + output_tokens * price["output"]
        cost_eur = cost_usd * USD_TO_EUR
        cost_matrix[row, col] = cost_eur

    # Plotting: tokens and time per image/question/model
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(imgq_set)*0.7), 10), sharex=True)
    bar_width = 0.2
    x = np.arange(len(imgq_set))

    # Tokens plot with cost annotation and total cost in legend
    total_costs = [cost_matrix[:, i].sum() for i in range(len(MODELS))]
    legend_labels = [f"{model} (Total: €{total_costs[i]:.2f})" for i, model in enumerate(MODELS)]
    for i, model in enumerate(MODELS):
        bars = axes[0].bar(x + i*bar_width, tokens_matrix[:, i], width=bar_width, label=legend_labels[i])
        for j, bar in enumerate(bars):
            cost = cost_matrix[j, i]
            if tokens_matrix[j, i] > 0:
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02*bar.get_height(),
                             f"€{cost:.3f}", ha='center', va='bottom', fontsize=8, rotation=90)
    axes[0].set_ylabel("Total Tokens")
    axes[0].set_title("Token Usage per Image/Question per Model\n(Cost in EUR above bars, total in legend)")
    axes[0].legend()

    # Time plot
    for i, model in enumerate(MODELS):
        axes[1].bar(x + i*bar_width, time_matrix[:, i], width=bar_width, label=model)
    axes[1].set_ylabel("Elapsed Time (s)")
    axes[1].set_title("Elapsed Time per Image/Question per Model")
    axes[1].set_xticks(x + bar_width)
    axes[1].set_xticklabels([f"{img}/{q}" for img, q in imgq_set], rotation=90)
    axes[1].legend()

    plt.tight_layout()

    output_path = os.path.join(plot_dir, "token_time_stats.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()

    def plot_stats(stats, blur, model, reasoning):
        plt.figure(figsize=(8, 5))
        plt.plot(stats['tokens'], label='Tokens')
        plt.plot(stats['time'], label='Time (s)')
        title_reasoning = f" with Reasoning ({reasoning})" if reasoning != 'none' else " (No Reasoning)"
        plt.title(f"Token/Time Stats: {model} - {blur}{title_reasoning}")
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()
        print(f"[INFO] Showing plot for model '{model}', blur '{blur}', reasoning '{reasoning}'")
        plt.show()
if __name__ == "__main__":
    main()