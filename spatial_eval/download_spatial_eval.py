#!/usr/bin/env python3
"""
Download and organize SpatialEval datasets from HuggingFace.

This script downloads the SpatialEval benchmark datasets with two visual modalities
(VQA, VTQA) and four tasks (Spatial-Map, Maze-Nav, Spatial-Grid, Spatial-Real),
organizing them into a structured folder hierarchy.

Dataset: MilaWang/SpatialEval
Paper: https://arxiv.org/pdf/2406.14852
"""

import json
from pathlib import Path
from typing import Dict, List
from datasets import load_dataset
from PIL import Image
import io


# Task name mapping from dataset IDs to folder names
TASK_MAPPING = {
    'spatialmap': 'spatial-map',
    'mazenav': 'maze-nav',
    'spatialgrid': 'spatial-grid',
    'spatialreal': 'spatial-real'
}


def extract_task_name(sample_id: str) -> str:
    """Extract task name from sample ID (e.g., 'spatialmap.0.123' -> 'spatial-map')."""
    task_key = sample_id.split('.')[0]
    return TASK_MAPPING.get(task_key, task_key)


def create_folder_structure(base_dir: Path) -> None:
    """Create the spatial_eval folder structure."""
    modalities = ['vqa', 'vtqa']
    tasks = ['spatial-map', 'maze-nav', 'spatial-grid', 'spatial-real']
    
    for modality in modalities:
        for task in tasks:
            task_dir = base_dir / modality / task
            task_dir.mkdir(parents=True, exist_ok=True)
            
            # Create images subfolder for visual modalities
            (task_dir / 'images').mkdir(exist_ok=True)
    
    print(f"✓ Created folder structure in {base_dir}")


def save_modality_data(modality: str, base_dir: Path) -> None:
    """Download and save data for a single modality."""
    print(f"\n{'='*60}")
    print(f"Processing modality: {modality.upper()}")
    print(f"{'='*60}")
    
    # Load dataset from HuggingFace
    print(f"Loading dataset from HuggingFace...")
    dataset = load_dataset("MilaWang/SpatialEval", modality, split="test")
    print(f"✓ Loaded {len(dataset)} samples")
    
    # Organize samples by task
    task_samples: Dict[str, List] = {
        'spatial-map': [],
        'maze-nav': [],
        'spatial-grid': [],
        'spatial-real': []
    }
    
    # Process each sample
    print(f"Organizing samples by task...")
    for idx, sample in enumerate(dataset):
        if idx % 500 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(dataset)} samples...")
        
        task_name = extract_task_name(sample['id'])
        
        # Prepare sample data
        sample_data = {
            'id': sample['id'],
            'text': sample['text'],
            'oracle_answer': sample['oracle_answer'],
            'oracle_option': sample['oracle_option'],
            'oracle_full_answer': sample['oracle_full_answer']
        }
        
        # Handle images
        if sample.get('image') is not None:
            image = sample['image']
            
            # Save image
            task_dir = base_dir / modality / task_name
            image_filename = f"{sample['id'].replace('.', '_')}.png"
            image_path = task_dir / 'images' / image_filename
            
            # Convert PIL Image to file
            if isinstance(image, Image.Image):
                image.save(image_path)
            
            # Store relative path in metadata
            sample_data['image_path'] = f"images/{image_filename}"
        
        task_samples[task_name].append(sample_data)
    
    print(f"✓ Organized all {len(dataset)} samples")
    
    # Save JSON files for each task
    print(f"Saving data files...")
    for task_name, samples in task_samples.items():
        task_dir = base_dir / modality / task_name
        json_path = task_dir / 'data.json'
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        num_images = len(list((task_dir / 'images').glob('*.png')))
        
        print(f"  ✓ {task_name}: {len(samples)} samples + {num_images} images")
    
    print(f"✓ Completed {modality.upper()}")


def main():
    """Main execution function."""
    base_dir = Path(__file__).parent / 'spatial_eval'
    
    print("=" * 60)
    print("SpatialEval Dataset Downloader")
    print("=" * 60)
    print(f"Destination: {base_dir.absolute()}")
    print(f"Dataset: MilaWang/SpatialEval")
    print(f"Modalities: VQA, VTQA (visual only)")
    print(f"Tasks: Spatial-Map, Maze-Nav, Spatial-Grid, Spatial-Real")
    print()
    
    # Create folder structure
    create_folder_structure(base_dir)
    
    # Download and organize each modality
    modalities = ['vqa', 'vtqa']
    
    for modality in modalities:
        try:
            save_modality_data(modality, base_dir)
        except Exception as e:
            print(f"✗ Error processing {modality}: {e}")
            raise
    
    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"Location: {base_dir.absolute()}")
    print(f"Structure:")
    print(f"  spatial_eval/")
    print(f"    ├── vqa/        (vision-only)")
    print(f"    └── vtqa/       (vision-text)")
    print(f"        ├── spatial-map/")
    print(f"        ├── maze-nav/")
    print(f"        ├── spatial-grid/")
    print(f"        └── spatial-real/")
    print()
    print("Each task folder contains:")
    print("  - data.json       (sample metadata)")
    print("  - images/         (PNG image files)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
