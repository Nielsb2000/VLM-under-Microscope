import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SKILLS_DIR = Path('spatial_eval/models/skills')

skills = []
examples = {}

for skill_dir in SKILLS_DIR.iterdir():
    if skill_dir.is_dir():
        skills.append(skill_dir.name)
        assets_dir = skill_dir / 'assets'
        if assets_dir.exists():
            imgs = [f.name for f in assets_dir.glob('example*.png')]
            examples[skill_dir.name] = imgs
        else:
            examples[skill_dir.name] = []

# Hierarchical layout: master-skill at top, task skills horizontally below
if 'master-skill' in skills:
    task_skills = [s for s in skills if s != 'master-skill']
    n = len(task_skills)
    fig, ax = plt.subplots(figsize=(2 + n * 2, 4))
    ax.axis('off')

    # Master-skill box at top center
    master_x = 0.5
    master_y = 3.2
    ax.add_patch(mpatches.Rectangle((master_x - 0.4, master_y - 0.3), 0.8, 0.6, fill=True, color='#ffe0b2'))
    ax.text(master_x, master_y, 'master-skill', fontsize=13, va='center', ha='center', fontweight='bold', color='black')

    # Task skill boxes horizontally below
    xs = [0.2 + i * (1.2 / max(1, n-1)) for i in range(n)] if n > 1 else [0.5]
    task_y = 1.2
    for i, skill in enumerate(task_skills):
        x = xs[i]
        ax.add_patch(mpatches.Rectangle((x - 0.3, task_y - 0.3), 0.6, 0.6, fill=True, color='#e0e0e0'))
        ax.text(x, task_y, skill, fontsize=12, va='center', ha='center', fontweight='bold')
        imgs = examples[skill]
        if imgs:
            ax.text(x, task_y - 0.5, ', '.join(imgs), fontsize=10, va='center', ha='center', color='blue')
        else:
            ax.text(x, task_y - 0.5, '(no examples)', fontsize=10, va='center', ha='center', color='gray')
        # Arrow from master-skill to task skill
        ax.annotate('', xy=(x, task_y + 0.3), xytext=(master_x, master_y - 0.3),
                    arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 4)
    plt.title('Skills Routing and Example Images', fontsize=14)
    plt.tight_layout()
    plt.savefig('skills_example_diagram.png')
    plt.show()
else:
    # Fallback: just show skills and examples vertically
    fig, ax = plt.subplots(figsize=(6, 2 + len(skills)))
    ax.axis('off')
    for i, skill in enumerate(skills):
        y = len(skills) - i - 1
        ax.add_patch(mpatches.Rectangle((0.1, y), 0.8, 0.6, fill=True, color='#e0e0e0'))
        ax.text(0.15, y + 0.3, skill, fontsize=12, va='center', ha='left', fontweight='bold')
        imgs = examples[skill]
        if imgs:
            ax.text(0.5, y + 0.3, ', '.join(imgs), fontsize=10, va='center', ha='left', color='blue')
        else:
            ax.text(0.5, y + 0.3, '(no examples)', fontsize=10, va='center', ha='left', color='gray')
    ax.set_ylim(-0.5, len(skills)+0.5)
    plt.title('Skills and Example Images', fontsize=14)
    plt.tight_layout()
    plt.savefig('skills_example_diagram.png')
    plt.show()
