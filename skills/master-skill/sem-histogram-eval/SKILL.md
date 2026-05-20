---
name: sem-histogram-eval
description: Run the objective evaluation script after the agent has finished its visual refinement pass. Call this once, as the final step of a case-study run.
---

# SEM Histogram Evaluation Skill

This skill covers **only** how to invoke the evaluation script at the end of a
case-study run. It does not describe what the metric measures or what a good
score looks like — that information is withheld intentionally.

> **IMPORTANT — this script must NOT be used to guide refinement.**
> Adjust sliders based solely on your own visual assessment of the image.
> Call this script **once**, after you have declared you are satisfied.

> **STRICT PROTOCOL RULE — NO READING OF EVALUATION OUTPUTS.**
> The agent is **never permitted** to read, inspect, or act on any evaluation
> output files, including anything under `/workspace/histograms/`.
> Reading these files at any point constitutes a protocol violation and
> invalidates the experiment. The outputs are for the researcher only.
> If you do look at the outputs, you must disclose this explicitly.

---

## When to call this

Call the evaluation script **once** as the very last step, after you have:

1. Pressed Randomize (or called `paint_canvas("randomize_filters")`).
2. Iteratively adjusted `brightness`, `contrast`, and `saturation` using
   `paint_canvas("set_filters", {...})` and your own visual judgment.
3. Declared that you are satisfied with the image quality.

---

## How to call it

```python
import subprocess
res = subprocess.run(
    ["python3",
     "/workspace/skills/master-skill/sem-histogram-eval/sem_histogram_error.py",
     "--paint-url", "http://host.docker.internal:3000"],
    capture_output=True, text=True, timeout=60
)
print(res.stdout)
if res.returncode != 0:
    print("STDERR:", res.stderr)
```

> If `host.docker.internal` does not resolve, use `http://172.17.0.1:3000`.

---

## What the script prints

The script prints a brief completion message confirming the run finished and
lists metadata (reference image name, timestamp, session counters).
It does **not** print any score values — those are saved to files for the
researcher and must not be read by the agent.
