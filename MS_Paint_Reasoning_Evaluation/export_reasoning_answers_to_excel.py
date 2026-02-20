import os
# This script exports ONLY reasoning-mode answers (e.g., heavy_blur_high, med_blur_medium, etc.) to Excel.
# It will skip non-reasoning results and warn if none are found.

print("[WIP] Reasoning-mode Excel export: Output may look odd due to multi-step reasoning text in answers. Use for inspection only!")
import pandas as pd
import openpyxl

# This script exports ONLY reasoning-mode answers (e.g., heavy_blur_high, med_blur_medium, etc.) to Excel.
# It will skip non-reasoning results and warn if none are found.

base_dir = os.path.dirname(__file__)
answers_dir = os.path.join(base_dir, 'MS_paint_images', 'MS paint answers')
questions_dir = os.path.join(base_dir, 'MS_paint_images', 'MS paint questions')
results_root = os.path.join(base_dir, 'Results')

sheet_dfs = {}
reasoning_folders = [f for f in os.listdir(results_root)
                     if '_' in f and not f.endswith(('none', 'med_blur', 'heavy_blur', 'none'))]

if not reasoning_folders:
    print("No reasoning-mode results found. Exiting.")
    exit(0)

for folder in sorted(reasoning_folders):
    blur, reasoning = folder.split('_', 1)
    results_dir = os.path.join(results_root, folder)
    rows = []
    for img_num in range(1, 9):
        ans_file = os.path.join(answers_dir, f'Answers{img_num}.txt')
        q_file = os.path.join(questions_dir, f'Questions{img_num}.txt')
        # Read GT answers
        gt_answers = {}
        with open(ans_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        qnum = parts[0].replace('Answer', '').strip()
                        gt_answers[qnum] = parts[1].strip()
        # Read questions
        questions = {}
        with open(q_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        qnum = parts[0].replace('Question', '').strip()
                        questions[qnum] = parts[1].strip()
        # For each question
        for qnum in questions:
            row = {
                'Blur Level': blur,
                'Reasoning Mode': reasoning,
                'Image': img_num,
                'Question #': qnum,
                'Question': questions[qnum],
                'GT Answer': gt_answers.get(qnum, '')
            }
            # Add model answers (detect models dynamically)
            q_dir = os.path.join(results_dir, f'img{img_num}', f'q{qnum}')
            if os.path.isdir(q_dir):
                for fname in os.listdir(q_dir):
                    if fname.startswith('answer_') and fname.endswith('.txt'):
                        model = fname[len('answer_'):-4]
                        with open(os.path.join(q_dir, fname), 'r') as f:
                            row[f'{model} Answer'] = f.read().strip()
            rows.append(row)
    sheet_name = f'{blur}_{reasoning}_reasoning'
    sheet_dfs[sheet_name] = pd.DataFrame(rows)

excel_path = os.path.join(base_dir, 'reasoning_answers_table.xlsx')
with pd.ExcelWriter(excel_path) as writer:
    for sheet, df in sheet_dfs.items():
        df.to_excel(writer, sheet_name=sheet, index=False)
print(f"Reasoning Excel file saved to {excel_path}")
