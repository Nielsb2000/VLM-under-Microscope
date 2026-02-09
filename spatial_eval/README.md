# SpatialEval Dataset

This folder contains the **SpatialEval** benchmark datasets for evaluating spatial intelligence in Large Language Models (LLMs) and Vision-Language Models (VLMs).

## 📊 Dataset Overview

**Source**: [MilaWang/SpatialEval on HuggingFace](https://huggingface.co/datasets/MilaWang/SpatialEval)  
**Paper**: [Is A Picture Worth A Thousand Words? Delving Into Spatial Reasoning for Vision Language Models](https://arxiv.org/pdf/2406.14852)  
**Code**: [GitHub - jiayuww/SpatialEval](https://github.com/jiayuww/SpatialEval)

- **Size**: ~1.43 GB
- **Total Samples**: ~9,270 (4,635 per modality)
- **Modalities**: 2 (VQA, VTQA - visual only)
- **Tasks**: 4 (Spatial-Map, Maze-Nav, Spatial-Grid, Spatial-Real)

## 📁 Folder Structure

```
spatial_eval/
├── vqa/                    # Vision-only Questions & Answers
│   ├── spatial-map/
│   │   ├── data.json
│   │   └── images/         # PNG images
│   ├── maze-nav/
│   │   ├── data.json
│   │   └── images/
│   ├── spatial-grid/
│   │   ├── data.json
│   │   └── images/
│   └── spatial-real/
│       ├── data.json
│       └── images/
└── vtqa/                   # Vision-Text Questions & Answers
    ├── spatial-map/
    │   ├── data.json
    │   └── images/         # PNG images
    ├── maze-nav/
    │   ├── data.json
    │   └── images/
    ├── spatial-grid/
    │   ├── data.json
    │   └── images/
    └── spatial-real/
        ├── data.json
        └── images/
```

## 🎯 Tasks

### 1. Spatial-Map
Understanding spatial relationships between objects in map-based scenarios.

### 2. Maze-Nav
Testing navigation capabilities through complex maze environments.

### 3. Spatial-Grid
Evaluating spatial reasoning within structured grid environments.

### 4. Spatial-Real
Assessing real-world spatial understanding with practical scenarios.

## 📝 Data Format

Each `data.json` file contains an array of samples with the following structure:

```json
{
  "id": "spatialmap.0.123",
  "text": "Question text...",
  "oracle_answer": "A",
  "oracle_option": "northeast",
  "oracle_full_answer": "The answer is A. northeast because...",
  "image_path": "images/spatialmap_0_123.png"
}
```

### Fields:
- **id**: Unique identifier (format: `{task}.{question_type}.{index}`)
- **text**: Question text prompt
- **oracle_answer**: Concise answer (letter option or value)
- **oracle_option**: Detailed answer option
- **oracle_full_answer**: Complete answer with reasoning
- **image_path**: Relative path to image file

## 🚀 Usage

### Loading Data

```python
import json
from pathlib import Path

# Load VQA Spatial-Map data
with open('spatial_eval/vqa/spatial-map/data.json', 'r') as f:
    vqa_spatial_map = json.load(f)

print(f"Loaded {len(vqa_spatial_map)} samples")
print(f"Sample: {vqa_spatial_map[0]}")
```

### Loading Images

```python
from PIL import Image
from pathlib import Path

# Load VQA sample with image
base_dir = Path('spatial_eval/vqa/spatial-map')
with open(base_dir / 'data.json', 'r') as f:
    vqa_data = json.load(f)

sample = vqa_data[0]
image_path = base_dir / sample['image_path']
image = Image.open(image_path)
image.show()
```

### Task-Specific Loading

```python
def load_task_data(modality: str, task: str):
    """Load data for a specific modality and task."""
    data_path = Path(f'spatial_eval/{modality}/{task}/data.json')
    with open(data_path, 'r') as f:
        return json.load(f)

# Examples
vqa_mazenav = load_task_data('vqa', 'maze-nav')
vqa_spatialgrid = load_task_data('vqa', 'spatial-grid')
vtqa_spatialreal = load_task_data('vtqa', 'spatial-real')
```

## 🔄 Re-downloading

To re-download the datasets:

```bash
uv run python download_spatial_eval.py
```

This will overwrite existing data in the `spatial_eval/` folder.

## 📚 Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{wang2024spatial,
  title={Is A Picture Worth A Thousand Words? Delving Into Spatial Reasoning for Vision Language Models},
  author={Wang, Jiayu and Ming, Yifei and Shi, Zhenmei and Vineet, Vibhav and Wang, Xin and Li, Yixuan and Joshi, Neel},
  booktitle={The Thirty-Eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```

## 📖 References

- **Project Page**: https://spatialeval.github.io/
- **Paper**: https://arxiv.org/pdf/2406.14852
- **Dataset**: https://huggingface.co/datasets/MilaWang/SpatialEval
- **Code**: https://github.com/jiayuww/SpatialEval
- **NeurIPS Talk**: https://neurips.cc/virtual/2024/poster/94371

## ℹ️ Dataset Statistics

| Modality | Task | Samples | Images |
|----------|------|---------|--------|
| VQA | Spatial-Map | ~1,500 | ~1,500 |
| VQA | Maze-Nav | ~1,500 | ~1,500 |
| VQA | Spatial-Grid | ~1,500 | ~1,500 |
| VQA | Spatial-Real | ~135 | ~135 |
| VTQA | Spatial-Map | ~1,500 | ~1,500 |
| VTQA | Maze-Nav | ~1,500 | ~1,500 |
| VTQA | Spatial-Grid | ~1,500 | ~1,500 |
| VTQA | Spatial-Real | ~135 | ~135 |

**Total**: ~9,270 samples across 8 task-modality combinations (visual modalities only)

---

*Downloaded on February 9, 2026 using `download_spatial_eval.py`*
