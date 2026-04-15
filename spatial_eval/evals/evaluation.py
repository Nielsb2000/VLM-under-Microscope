import json
import os
import csv
import argparse
import pandas as pd
import re
from typing import Optional, Tuple, List, Dict

def extract_model_name(filename: str, suffix: str = None) -> Optional[str]:
    """Extracts the model name (without .jsonl extension) from the filename.

    Handles both legacy filenames (no timestamp) and the current timestamped
    format: ``m-{model}_{variant}_{YYYYMMDD_HHMMSS}.jsonl``.
    """
    prefix = "m-"

    if not filename.startswith(prefix):
        return None

    if not filename.endswith(".jsonl"):
        return None

    # Strip prefix and .jsonl suffix — return the whole body as the model-name
    # key so that accuracy CSVs use the full unique identifier.
    return filename[len(prefix):-len(".jsonl")]

def remove_redundancy(text: str) -> str:
    """Removes redundant characters from the text."""
    return text.replace('**', '').replace('.', '')

def extract_before_is(input_string: str) -> str:
    """Extracts the part of the string before the first occurrence of 'is'."""
    parts = input_string.split(' is', 1)
    return parts[0].strip()


def extract_answer_from_text_spatialmap(text: str, question_id: int = 0, model_name: Optional[str] = None) -> Optional[str]:
    """Extracts answers from spatial map text based on the question ID."""
    number_mapping = {
        'zero': 0, 'no': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
    }
    
    dirs = ['southeast', 'northeast', 'northwest', 'southwest']
    dir_pattern = rf"\b({'|'.join(dirs)})\b"
    
    if text is None:
        return None
    
    if question_id == 0:  
        direction_match = re.search(r'\b[A-D]\.\s*(' + '|'.join(dirs) + r')\b', text, re.IGNORECASE)
        if direction_match:
            return direction_match.group(1).lower()
        
        match = re.search(dir_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
        return None
        
    elif question_id == 1:
        match = re.search(rf'^([\w\s\'’]+?)\s+is\s+(?:located\s+|in\s+the\s+|located\s+to\s+the\s+)({dir_pattern})', text, re.IGNORECASE)
        if match:
            return match.group(1)
        
        match = re.search(r'\b[A-D]\.\s*(.*)', text)
        if match:
            string = match.group(1)
            string = remove_redundancy(string)
            string = extract_before_is(string)
            return string
        
        match = re.search(r'\b([ABCD][.,]|[(][abcdABCD][)])\s*(.*?)(?=\sis\b|\.|,|<|$)', text)
        if match:
            answer = match.group(1).strip()
            # Remove trailing punctuation if any
            answer = re.sub(r'[\.,\?!<]+$', '', answer)
            return answer
        
        match = re.search(rf'Therefore, the object in the {dir_pattern} of [\w\s\'’]+ is ([\w\s\'’]+)', text, re.IGNORECASE)
        if match:
            string = match.group(2)
            return string
        
        if 'claude' in model_name.lower():
            match = re.search(rf'^([\w\s\'’]+?)\s+is\s+(to\s+the\s+)({dir_pattern})', text, re.IGNORECASE)
            # match = re.search(rf'(?:\*\*Concise Answer:\*\*\n)?([\w\s\'’]+?)\s+is\s+(?:located\s+|in\s+the\s+|in\s+|located\s+to\s+the\s+)({dir_pattern})', text, re.IGNORECASE)
            if match:
                string = match.group(1)
                return string
        
        if 'geminiv' in model_name.lower():
            patterns = [
            rf'\*\*Concise Answer:\*\*\n([\w\s\'’]+?)\s+is\s+(?:located\s+|in\s+the\s+|in\s+|located\s+to\s+the\s+)({dir_pattern})',
            rf'\*\*Answer:\*\*\s+([\w\s\'’]+?)\s+is\s+in\s+the\s+({dir_pattern})\s+of\s+([\w\s\'’]+)',
            r'\*\*Answer:\*\*\n([\w\s\'’]+)',
            r'\*\*Answer\*\*:\s+([\w\s\'’]+)',
            r'\*\*Answer:\*\*\s+([\w\s\'’]+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1)
        
        if 'gpt4o' in model_name.lower():
            match = re.search(rf'Concise Answer:\s+([\w\s\'’]+?)\s+is\s+(?:located\s+|in\s+the\s+|in\s+|located\s+to\s+the\s+)({dir_pattern})', text, re.IGNORECASE)
            if match:
                string = match.group(1)
                return string
        
        # If no match, check for an answer following "is", with specific end markers defined
        match = re.search(r'\bis\b\s+(.*?)(?=\.|,|<|$)', text)
        if match:
            answer = match.group(1).strip()
            # Remove trailing punctuation if any
            answer = re.sub(r'[\.,\?!<]+$', '', answer)
            return answer
        
        return None  # Return None if no match is found

    elif question_id == 2:
        match = re.search(r'\b[A-D]\.\s*(\d+)', text) # match number only
        if match:
            return convert_str_to_int(match.group(1))
        
        found_numbers = []
        # Check for textual numbers and their positions
        for text_num, num in number_mapping.items():
            for match in re.finditer(rf'\b{text_num}\b', text, re.IGNORECASE):
                found_numbers.append((match.start(), num))

        # Check for digit sequences and their positions, specifically ignoring list markers at the start
        # Exclude numbers following "\n\n" and directly followed by ". "
        text = re.sub(r'^\n\n\d+\.\s', '', text)  # Remove the leading list marker if it exists

        for match in re.finditer(r'\d+', text):
            found_numbers.append((match.start(), int(match.group(0))))

        # Sort found numbers by their positions (smallest position first)
        if found_numbers:
            found_numbers.sort(key=lambda x: x[0])
            # Return the number associated with the earliest position
            return found_numbers[0][1]
        return None

    else: 
        raise ValueError(f"Question ID {question_id} is not supported.")
    
    return None  # Return None if no numbers are found

def extract_answer_from_text_mazenav(text: str, question_id: int = 0, model_name: Optional[str] = None) -> Optional[str]:
    """Extracts answers from maze navigation text based on the question ID."""
    number_mapping = {
        'zero': 0, 'no': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
    }
    
    if question_id == 2:  # require binary answers
        yes_patterns = [r'\byes\b', r'the answer is yes', r'\"yes\"', r"\'yes\'", r'is the shortest path']
        no_patterns = [r'\bno\b', r'the answer is no', r'\"no\"', r"\'no\'", r'\bnot\b']
        
        # Check for "Yes" answers
        for pattern in yes_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return "Yes"
        
        # Check for "No" answers
        for pattern in no_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return "No"

    else:
        # Check for textual number patterns first
        for text_num, num in number_mapping.items():
            pattern = rf'\b{text_num}\b'
            if re.search(pattern, text, re.IGNORECASE):
                return num

        patterns = {
            0: [  # For right turns
                r'\bThere are\s*(\d+)\s*right turns\b', # for proprietary
                r'\bThere is\s*(\d+)\s*right turn\b',
                r'\b(\d+)\s+right turn(s)?',
                r'answer is\s+(\d+)',
                r'answer is:\s*\n*\s*(\d+)',
                r'from S to E is\s+(\d+)',
                r'Answer:\*\*\s*(\d+)\b',
            ],
            1: [  # For total turns
                r'\bThere are\s*(\d+)\s*total turns\b', # for proprietary
                r'\bThere are\s*(\d+)\s*turns\b',
                r'There is\s*(\d+)\s*turn\b',
                r'There is\s*(\d+)\s*total turn\b',
                r'answer is\s+(\d+)',
                r'answer is:\s*\n*\s*(\d+)',
                r'from S to E is\s+(\d+)',
                r'\btotal of\s+(\d+)\s+turn(s)?',
                r'Answer:\*\*\s*(\d+)\b',
                r'(\d+)\s+total turn(s)?',
                r'\b(\d+)\s+turn(s)?',  # Matches "8 turns" broadly; consider specificity vs. overlap
            ]
        }
        
        for pattern in patterns[question_id]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))  # Return the first matching group as integer

        # If no specific pattern matches, try to extract the first number in the text
        fallback_match = re.search(r'\d+', text)
        if fallback_match:
            return int(fallback_match.group(0))
    
    return None  # Return None if no number or textual number is found at all

def extract_answer_from_text_spatialgrid(text: str, question_id: int = 0, model_name: Optional[str] = None) -> Optional[int]:
    """Extracts answers from spatial grid text based on the question ID."""
    number_mapping = {
        'zero': 0, 'no': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
    }
    
    animals = ['giraffe', 'cat', 'dog', 'elephant', 'rabbit']
    animal_pattern = rf"\b({'|'.join(animals)})\b"
    
    if question_id >= 1:
        match = re.search(animal_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    else:
        # check answer that is a number
        if 'claude' in model_name.lower():
            specific_phrases = [
            (r'\bthere are\s*(\d+)\s*blocks\b', 1),
            (r'\bthere appear to be\s*(\d+)\s*blocks\b', 1),
            (r'\bcontains\s*(\d+)\s*blocks\b', 1)
            ]
            
            for phrase_pattern, group_index in specific_phrases:
                match = re.search(phrase_pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(group_index))
        # If no specific phrases found, proceed with other checks
        found_numbers = []

        # Check for textual numbers and their positions
        for text_num, num in number_mapping.items():
            for match in re.finditer(rf'\b{text_num}\b', text, re.IGNORECASE):
                found_numbers.append((match.start(), num))

        # Check for digit sequences and their positions, specifically ignoring list markers at the start
        # Exclude numbers following "\n\n" and directly followed by ". "
        text = re.sub(r'^\n\n\d+\.\s', '', text)  # Remove the leading list marker if it exists

        for match in re.finditer(r'\d+', text):
            found_numbers.append((match.start(), int(match.group(0))))

        # Sort found numbers by their positions (smallest position first)
        if found_numbers:
            found_numbers.sort(key=lambda x: x[0])
            # Return the number associated with the earliest position
            return found_numbers[0][1]

    return None

def convert_str_to_int(var: str) -> Optional[int]:
    """Attempts to convert a variable to an integer if it is a string that represents an integer."""
    if isinstance(var, str):
        try:
            return int(var)
        except ValueError:
            return var
    return var
    
def _check_answer(model_answer: str, data: dict) -> Tuple[int, int]:
    """
    Match the model's raw answer against oracle fields in priority order:
      1. oracle_full_answer  (e.g. "C. 2")   — substring match in full text
      2. oracle_answer       (e.g. "2")       — whole-word match in last non-empty line only
      3. oracle_option       (e.g. "C")       — explicit answer marker in last non-empty line only
    Restricting steps 2 and 3 to the final line avoids false positives from
    intermediate reasoning (e.g. "total 7" when the final answer is "A. 8").
    Returns (is_correct: int, step_matched: int) where step_matched is 1/2/3 or 0 if wrong.
    """
    if not model_answer:
        return 0, 0
    text = model_answer.lower()

    # Derive the last non-empty line for focused matching
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    last_line = lines[-1] if lines else ''

    # 1. oracle_full_answer — substring in full text (e.g. "c. 2" is specific enough)
    full = str(data.get('oracle_full_answer') or '').lower().strip()
    if full and full in text:
        return 1, 1

    # 2. oracle_answer — whole-word match, last line only
    ans = str(data.get('oracle_answer') or '').lower().strip()
    if ans and re.search(rf'\b{re.escape(ans)}\b', last_line):
        return 1, 2

    # 3. oracle_option — explicit answer marker, last line only
    opt = str(data.get('oracle_option') or '').strip()
    if opt:
        o = re.escape(opt.lower())
        if re.search(rf'(?<![a-z]){o}[.)]', last_line) or re.search(rf'\({o}\)', last_line):
            return 1, 3

    return 0, 0


def evaluate_model_accuracy(model_output_path: str, eval_summary_path: str, model_name: Optional[str] = None) -> Tuple[float, int]:
    """Evaluates the accuracy of the model based on the output and summary paths.

    Matching strategy (in order):
      1. oracle_full_answer substring in model answer  (e.g. "C. 2")
      2. oracle_answer substring in model answer       (e.g. "2")
      3. oracle_option substring in model answer       (e.g. "C")
    If none match, the answer is considered wrong.
    """
    eval_summary: List[Dict[str, str]] = []
    correct_answers = 0
    line_count = 0
    step_counts = {1: 0, 2: 0, 3: 0, 0: 0}  # 0 = wrong/no match

    with open(model_output_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            line_count += 1
            raw_answer = data.get('answer') or ''
            eval_result, step = _check_answer(raw_answer, data)
            correct_answers += eval_result
            step_counts[step] += 1
            eval_summary.append({
                'id': data.get('id'),
                'oracle_full_answer': data.get('oracle_full_answer'),
                'oracle_option': data.get('oracle_option'),
                'oracle_answer': data.get('oracle_answer'),
                'model_output': raw_answer[:300],
                'eval_result': eval_result,
                'match_step': step,
            })

    total = line_count or 1
    print(f"  Match stats: step1(full)={step_counts[1]} ({step_counts[1]/total:.0%})  "
          f"step2(answer)={step_counts[2]} ({step_counts[2]/total:.0%})  "
          f"step3(option)={step_counts[3]} ({step_counts[3]/total:.0%})  "
          f"no_match={step_counts[0]} ({step_counts[0]/total:.0%})")

    with open(eval_summary_path, 'w') as outfile:
        for entry in eval_summary:
            json.dump(entry, outfile)
            outfile.write('\n')
    return correct_answers / line_count if line_count > 0 else 0, line_count


def main(args):
    output_dir = os.path.join(args.output_dir, args.task)
    output_csv = os.path.join(args.eval_summary_dir, f"{args.task}_acc.csv")

    model_accuracies = []
    for filename in os.listdir(output_dir):
        if filename.endswith(".jsonl"):
            model_name = extract_model_name(filename)
            if model_name:
                output_filename = os.path.join(output_dir, filename)
                eval_summary_path = os.path.join(args.eval_summary_dir, f"{args.task}_{model_name}_eval_summary.jsonl")
                accuracy, num_outputs = evaluate_model_accuracy(output_filename, eval_summary_path, model_name)
                model_accuracies.append({'Model Name': model_name, f'Acc': accuracy})
                print(f"Evaluated {model_name}: {accuracy:.2%} ({int(accuracy * num_outputs)}/{num_outputs})")

    if not model_accuracies:
        print(f"No model outputs found in {output_dir}")
        return
    
    df = pd.DataFrame(model_accuracies)
    df_sorted = df.sort_values(by='Model Name', ascending=True)
    df_sorted.to_csv(output_csv, index=False)
    
    print(f"\n{args.task} | {args.mode} | CSV file with model accuracies has been created at {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model accuracy for open source models.')
    parser.add_argument('--mode', choices=['tqa', 'vqa', 'vtqa'], default='tqa')
    parser.add_argument('--output_folder', type=str, default='outputs/', help='Path to the directory containing model outputs.')
    parser.add_argument('--dataset_id', type=str, default='MilaWang/SpatialEval', help='Dataset identifier for Hugging Face.')
    parser.add_argument('--eval_summary_dir', type=str, default='eval_summary', help='Path to the directory to save evaluation summaries.')
    parser.add_argument('--task', type=str, default='spatialgrid', choices=['all', 'spatialmap', 'mazenav', 'spatialgrid', 'spatialreal'], help='Task to evaluate.')
    args = parser.parse_args()
    
    args.output_folder = os.path.join(args.output_folder, args.dataset_id.replace("/", "__"))
    args.output_dir = os.path.join(args.output_folder, args.mode)
    args.eval_summary_dir = os.path.join(args.eval_summary_dir, args.mode)
    os.makedirs(args.eval_summary_dir, exist_ok=True)
    
    main(args)