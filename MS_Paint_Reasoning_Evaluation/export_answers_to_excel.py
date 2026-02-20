import os 
import pandas as pd
import openpyxl
# Paths
base_dir = os.path.dirname(__file__)
answers_dir = os.path.join(base_dir, 'MS_paint_images', 'MS paint answers')
questions_dir = os.path.join(base_dir, 'MS_paint_images', 'MS paint questions')
models = ['gpt-4o', 'gpt-5.1', 'gpt-5.2']

blur_levels = {
    'none': os.path.join(base_dir, 'Results'),
    'med_blur': os.path.join(base_dir, 'Results', 'med_blur'),
    'heavy_blur': os.path.join(base_dir, 'Results', 'heavy_blur'),
}

sheet_dfs = {}

# Collect all rows for a combined DataFrame
all_rows = []
for blur, results_dir in blur_levels.items():
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
                'Image': img_num,
                'Question #': qnum,
                'Question': questions[qnum],
                'GT Answer': gt_answers.get(qnum, '')
            }
            # Add model answers
            for model in models:
                ans_path = os.path.join(results_dir, f'img{img_num}', f'q{qnum}', f'answer_{model}.txt')
                if os.path.exists(ans_path):
                    with open(ans_path, 'r') as f:
                        row[f'{model} Answer'] = f.read().strip()
                else:
                    row[f'{model} Answer'] = ''
            rows.append(row)
            all_rows.append(row)
    sheet_dfs[blur] = pd.DataFrame(rows)

# Combined DataFrame for all blur levels
combined_df = pd.DataFrame(all_rows)

# Export to Excel with multiple sheets
excel_path = os.path.join(base_dir, 'all_answers_table.xlsx')
with pd.ExcelWriter(excel_path) as writer:
    # Only write per-blur sheets, each with 'Blur Level' as first column
    for blur, df in sheet_dfs.items():
        # Ensure 'Blur Level' is the first column
        if 'Blur Level' not in df.columns:
            df.insert(0, 'Blur Level', blur)
        else:
            # Move 'Blur Level' to first column if not already
            cols = ['Blur Level'] + [c for c in df.columns if c != 'Blur Level']
            df = df[cols]
        df.to_excel(writer, sheet_name=blur, index=False)
print(f"Excel file saved to {excel_path}")
