import os
import argparse
from pathlib import Path
import datetime

def format_output_path_vlm(args):
    file_ext = ".jsonl"
    if hasattr(args, 'model_id') and args.model_id is not None:
        model_name = args.model_id.replace("/", "__")
    elif hasattr(args, 'model_path') and args.model_path is not None:
        model_name = args.model_path.replace("/", "__")
    else:
        raise ValueError("Both model_id and model_path are missing or None.")

    output_dir = Path(args.output_folder) / args.dataset_id.replace("/", "__") / args.mode / args.task

    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename= f"m-{model_name}"

    if args.random_image:
        filename += "_random_image"
    elif args.noise_image:
        filename += "_noise_image"
    if args.w_reason:
        filename += "_w_reason"
    elif args.completion:
        filename += "_completion"
    else:
        filename += "_bare"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    modified_output_filename = f"{filename}_{timestamp}{file_ext}"

    output_path = output_dir / modified_output_filename

    return output_path


def format_output_path_lm(args):
    file_ext = ".jsonl"
    model_name = args.model_path.replace("/", "__")
    
    output_suffix = ""

    output_dir = Path(args.output_folder) / args.dataset_id.replace("/", "__") / args.mode / args.task

    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.completion:
        output_suffix += "_completion"
    elif args.w_reason:
        output_suffix += "_w_reason"
    else:
        output_suffix += "_bare"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    modified_output_filename = f"m-{model_name}{output_suffix}_{timestamp}{file_ext}"
    
    output_path = output_dir / modified_output_filename

    return output_path
