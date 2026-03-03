# json_results_to_df.py
"""
Parse all results JSONs in Results/dashboard_data into a hierarchical pandas DataFrame.
Usage: import this script and call load_results_df(), or run as a script to print the DataFrame.
"""
import os
import re
import json
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "Results", "dashboard_data")

def load_results_df():
    records = []
    if not os.path.exists(RESULTS_DIR):
        return pd.DataFrame()
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith('.json'):
            continue
        meta = re.match(r"llm_results_([a-zA-Z0-9_\-]+)_([a-zA-Z0-9_\-]+)_([a-zA-Z0-9_\-.]+)_?([a-zA-Z0-9_]*)?.json", fname)
        if not meta:
            continue
        # Parse skills/no_skills from filename (robust to multiple underscores)
        base = fname.rsplit('.', 1)[0]
        last_part = base.split('_')[-2:]  # e.g. ['no', 'skills'] or ['skills']
        if last_part == ['no', 'skills']:
            skills_mode = 'no_skills'
        elif last_part[-1] == 'skills':
            skills_mode = 'skills'
        else:
            skills_mode = 'unknown'

        # Fix: parse image_type and blur_level correctly
        # image_type may contain _no, _med, or _heavy at the end, which is actually the blur_level
        image_type_raw = meta.group(1)
        blur_level = meta.group(2)
        model = meta.group(3)
        reasoning_mode = meta.group(4) or "none"
        # If image_type ends with _no, _med, or _heavy, that's the blur level
        if image_type_raw.endswith('_no'):
            image_type = image_type_raw[:-3]
            blur_level = 'no_blur'
        elif image_type_raw.endswith('_med'):
            image_type = image_type_raw[:-4]
            blur_level = 'med_blur'
        elif image_type_raw.endswith('_heavy'):
            image_type = image_type_raw[:-6]
            blur_level = 'heavy_blur'
        else:
            image_type = image_type_raw
        with open(os.path.join(RESULTS_DIR, fname), 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            m = re.match(r"img(\d+)_Q(\d+)_([a-zA-Z0-9\-.]+)", key)
            if not m:
                continue
            img = int(m.group(1))
            q = int(m.group(2))
            model_in_key = m.group(3)
            # Only keep image filename and question number
            image_filename = f"img{img}.png"
            correct = value[0] if isinstance(value, (list, tuple)) and len(value) > 0 else None
            input_tokens = None
            output_tokens = None
            if isinstance(value, (list, tuple)) and len(value) > 1 and isinstance(value[1], (list, tuple)):
                input_tokens = value[1][0]
                output_tokens = value[1][1]
            elapsed_time = value[2] if isinstance(value, (list, tuple)) and len(value) > 2 else None
            rec = {
                "image_type": image_type,
                "blur_level": blur_level,
                "Model": model_in_key,
                "Correct": correct,
                "image_num": img,
                "question_num": q,
                "Input Tokens": input_tokens,
                "Output Tokens": output_tokens,
                "Elapsed Time": elapsed_time,
                "skills_mode": skills_mode
            }
            # Only add reasoning_mode for non-gpt-4o models
            if model_in_key != "gpt-4o":
                rec["reasoning_mode"] = reasoning_mode
            records.append(rec)
    df = pd.DataFrame.from_records(records)
    # Map 'none' reasoning_mode to 'low' for dashboard compatibility
    if 'reasoning_mode' in df.columns:
        df['reasoning_mode'] = df['reasoning_mode'].replace({'none': 'low'})
    # Drop reasoning_mode column from gpt-4o rows
    if 'Model' in df.columns and 'reasoning_mode' in df.columns:
        gpt4o_mask = df['Model'] == 'gpt-4o'
        if gpt4o_mask.any():
            # Remove reasoning_mode column for gpt-4o rows by splitting DataFrame
            df_gpt4o = df[gpt4o_mask].drop(columns=['reasoning_mode'])
            df_other = df[~gpt4o_mask]
            df = pd.concat([df_gpt4o, df_other], ignore_index=True, sort=False)
            # Reorder columns to keep token/time columns at the end
            token_time_cols = ["Input Tokens", "Output Tokens", "Elapsed Time"]
            other_cols = [col for col in df.columns if col not in token_time_cols]
            df = df[other_cols + token_time_cols]
    # Move token/time columns to the end
    else:
        token_time_cols = ["Input Tokens", "Output Tokens", "Elapsed Time"]
        other_cols = [col for col in df.columns if col not in token_time_cols]
        df = df[other_cols + token_time_cols]
    return df

if __name__ == "__main__":
    df = load_results_df()
    print(df)
