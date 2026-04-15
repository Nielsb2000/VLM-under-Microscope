"""One-off script to generate example_N.txt + example_N.png symlinks for the
img-qa-val-v2 preload architecture.

Run from project root:
    uv run python spatial_eval/models/generate_preload_examples.py
"""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "skills_img_qa_val_v2", "examples")
ASSETS_BASE = os.path.join(SCRIPT_DIR, "skills_img_qa_val", "skills")

# ---------------------------------------------------------------------------
# Q&A data extracted from SKILL.md files
# Format: list of 10 examples, each example = list of (question, options_str, answer)
# ---------------------------------------------------------------------------

MAZENAV_DATA = [
    # example 0 — maze0
    [
        (
            "How many right turns are there in the provided path (marked by Blue) from S to E?",
            "A. 4 | B. 8 | C. 2 | D. 7",
            "C. 2",
        ),
        (
            "How many total turns are there in the provided path (marked by Blue) from S to E?",
            "A. 2 | B. 9 | C. 3 | D. 7",
            "A. 2",
        ),
        (
            "Is the exit (E) directly to the left of the starting point (S), with no vertical displacement?",
            "A. Yes | B. No",
            "B. No",
        ),
    ],
    # example 1 — maze1
    [
        (
            "How many right turns are there in the provided path (marked by Blue) from S to E?",
            "A. 4 | B. 3 | C. 7 | D. 2",
            "B. 3",
        ),
        (
            "How many total turns are there in the provided path (marked by Blue) from S to E?",
            "A. 4 | B. 1 | C. 9 | D. 5",
            "A. 4",
        ),
        (
            "Is the exit (E) directly below the starting point (S), with no horizontal displacement?",
            "A. Yes | B. No",
            "A. Yes",
        ),
    ],
    # example 2 — maze2
    [
        (
            "How many right turns are there in the provided path (marked by Blue) from S to E?",
            "A. 4 | B. 1 | C. 3 | D. 8",
            "C. 3",
        ),
        (
            "How many total turns are there in the provided path (marked by Blue) from S to E?",
            "A. 0 | B. 1 | C. 3 | D. 7",
            "C. 3",
        ),
        (
            "Is the exit (E) directly above the starting point (S), with no horizontal displacement?",
            "A. Yes | B. No",
            "A. Yes",
        ),
    ],
    # example 3 — maze3
    [
        (
            "How many right turns are there in the provided path (marked by Blue) from S to E?",
            "A. 1 | B. 5 | C. 9 | D. 0",
            "A. 1",
        ),
        (
            "How many total turns are there in the provided path (marked by Blue) from S to E?",
            "A. 5 | B. 2 | C. 8 | D. 1",
            "D. 1",
        ),
        (
            "Is the exit (E) to the bottom right of the starting point (S)?",
            "A. Yes | B. No",
            "A. Yes",
        ),
    ],
    # example 4 — maze4
    [
        (
            "How many right turns are there in the provided path (marked by Blue) from S to E?",
            "A. 4 | B. 3 | C. 7 | D. 0",
            "D. 0",
        ),
        (
            "How many total turns are there in the provided path (marked by Blue) from S to E?",
            "A. 2 | B. 9 | C. 4 | D. 7",
            "A. 2",
        ),
        (
            "Is the exit (E) to the bottom left of the starting point (S)?",
            "A. Yes | B. No",
            "B. No",
        ),
    ],
    # example 5 — maze5
    [
        (
            "How many right turns are there in the provided path (marked by Blue) from S to E?",
            "A. 0 | B. 9 | C. 8 | D. 7",
            "A. 0",
        ),
        (
            "How many total turns are there in the provided path (marked by Blue) from S to E?",
            "A. 2 | B. 1 | C. 6 | D. 8",
            "A. 2",
        ),
        (
            "Is the exit (E) directly to the left of the starting point (S), with no vertical displacement?",
            "A. Yes | B. No",
            "A. Yes",
        ),
    ],
    # example 6 — maze6
    [
        (
            "How many right turns are there in the provided path (marked by Blue) from S to E?",
            "A. 0 | B. 7 | C. 3 | D. 9",
            "A. 0",
        ),
        (
            "How many total turns are there in the provided path (marked by Blue) from S to E?",
            "A. 3 | B. 8 | C. 7 | D. 0",
            "D. 0",
        ),
        (
            "Is the exit (E) directly above the starting point (S), with no horizontal displacement?",
            "A. Yes | B. No",
            "A. Yes",
        ),
    ],
    # example 7 — maze7
    [
        (
            "How many right turns are there in the provided path (marked by Blue) from S to E?",
            "A. 0 | B. 3 | C. 9 | D. 7",
            "A. 0",
        ),
        (
            "How many total turns are there in the provided path (marked by Blue) from S to E?",
            "A. 3 | B. 7 | C. 0 | D. 6",
            "C. 0",
        ),
        (
            "Is the exit (E) directly to the right of the starting point (S), with no vertical displacement?",
            "A. Yes | B. No",
            "A. Yes",
        ),
    ],
    # example 8 — maze8
    [
        (
            "How many right turns are there in the provided path (marked by Blue) from S to E?",
            "A. 5 | B. 7 | C. 0 | D. 9",
            "C. 0",
        ),
        (
            "How many total turns are there in the provided path (marked by Blue) from S to E?",
            "A. 9 | B. 8 | C. 0 | D. 3",
            "C. 0",
        ),
        (
            "Is the exit (E) directly below the starting point (S), with no horizontal displacement?",
            "A. Yes | B. No",
            "A. Yes",
        ),
    ],
    # example 9 — maze9
    [
        (
            "How many right turns are there in the provided path (marked by Blue) from S to E?",
            "A. 6 | B. 3 | C. 0 | D. 2",
            "C. 0",
        ),
        (
            "How many total turns are there in the provided path (marked by Blue) from S to E?",
            "A. 7 | B. 6 | C. 1 | D. 3",
            "C. 1",
        ),
        (
            "Is the exit (E) to the top right of the starting point (S)?",
            "A. Yes | B. No",
            "A. Yes",
        ),
    ],
]

SPATIALGRID_DATA = [
    # example 0 — grid0
    [
        ("How many blocks contain dog?", "A. 5 | B. 6 | C. 2 | D. 7", "C. 2"),
        (
            "What is the animal of the block located at the top-left corner (the first row and the first column) of the grid?",
            "A. giraffe | B. cat | C. elephant | D. rabbit",
            "C. elephant",
        ),
        (
            "What is the animal of the block located at the first row, second column of the grid?",
            "A. elephant | B. giraffe | C. cat | D. rabbit",
            "D. rabbit",
        ),
    ],
    # example 1 — grid1
    [
        ("How many blocks contain rabbit?", "A. 0 | B. 1 | C. 3 | D. 4", "C. 3"),
        (
            "What is the animal of the block located at the top-left corner (the first row and the first column) of the grid?",
            "A. cat | B. rabbit | C. elephant | D. giraffe",
            "C. elephant",
        ),
        (
            "What is the animal of the block located at the first row, second column of the grid?",
            "A. rabbit | B. giraffe | C. dog | D. cat",
            "C. dog",
        ),
    ],
    # example 2 — grid2
    [
        ("How many blocks contain dog?", "A. 6 | B. 2 | C. 5 | D. 0", "C. 5"),
        (
            "What is the animal of the block located at the top-left corner (the first row and the first column) of the grid?",
            "A. elephant | B. dog | C. cat | D. rabbit",
            "B. dog",
        ),
        (
            "What is the animal of the block located at the first row, second column of the grid?",
            "A. elephant | B. rabbit | C. cat | D. giraffe",
            "B. rabbit",
        ),
    ],
    # example 3 — grid3
    [
        ("How many blocks contain rabbit?", "A. 8 | B. 2 | C. 7 | D. 6", "D. 6"),
        (
            "What is the animal of the block located at the top-left corner (the first row and the first column) of the grid?",
            "A. rabbit | B. giraffe | C. cat | D. elephant",
            "D. elephant",
        ),
        (
            "What is the animal of the block located at the first row, second column of the grid?",
            "A. rabbit | B. cat | C. giraffe | D. dog",
            "B. cat",
        ),
    ],
    # example 4 — grid4
    [
        ("How many blocks contain cat?", "A. 8 | B. 7 | C. 5 | D. 3", "B. 7"),
        (
            "What is the animal of the block located at the top-left corner (the first row and the first column) of the grid?",
            "A. dog | B. elephant | C. cat | D. rabbit",
            "A. dog",
        ),
        (
            "What is the animal of the block located at the first row, second column of the grid?",
            "A. dog | B. giraffe | C. cat | D. rabbit",
            "B. giraffe",
        ),
    ],
    # example 5 — grid5
    [
        ("How many blocks contain dog?", "A. 0 | B. 2 | C. 8 | D. 6", "B. 2"),
        (
            "What is the animal of the block located at the top-left corner (the first row and the first column) of the grid?",
            "A. dog | B. rabbit | C. cat | D. giraffe",
            "A. dog",
        ),
        (
            "What is the animal of the block located at the first row, second column of the grid?",
            "A. cat | B. rabbit | C. giraffe | D. elephant",
            "B. rabbit",
        ),
    ],
    # example 6 — grid6
    [
        ("How many blocks contain cat?", "A. 1 | B. 4 | C. 3 | D. 2", "D. 2"),
        (
            "What is the animal of the block located at the top-left corner (the first row and the first column) of the grid?",
            "A. giraffe | B. cat | C. elephant | D. dog",
            "D. dog",
        ),
        (
            "What is the animal of the block located at the first row, second column of the grid?",
            "A. giraffe | B. dog | C. cat | D. elephant",
            "D. elephant",
        ),
    ],
    # example 7 — grid7
    [
        ("How many blocks contain elephant?", "A. 8 | B. 9 | C. 5 | D. 2", "C. 5"),
        (
            "What is the animal of the block located at the top-left corner (the first row and the first column) of the grid?",
            "A. giraffe | B. cat | C. rabbit | D. elephant",
            "D. elephant",
        ),
        (
            "What is the animal of the block located at the first row, second column of the grid?",
            "A. dog | B. rabbit | C. elephant | D. giraffe",
            "B. rabbit",
        ),
    ],
    # example 8 — grid8
    [
        ("How many blocks contain dog?", "A. 2 | B. 3 | C. 5 | D. 0", "C. 5"),
        (
            "What is the animal of the block located at the top-left corner (the first row and the first column) of the grid?",
            "A. rabbit | B. cat | C. dog | D. giraffe",
            "A. rabbit",
        ),
        (
            "What is the animal of the block located at the first row, second column of the grid?",
            "A. cat | B. rabbit | C. elephant | D. giraffe",
            "C. elephant",
        ),
    ],
    # example 9 — grid9
    [
        ("How many blocks contain cat?", "A. 4 | B. 1 | C. 7 | D. 8", "A. 4"),
        (
            "What is the animal of the block located at the top-left corner (the first row and the first column) of the grid?",
            "A. dog | B. rabbit | C. giraffe | D. cat",
            "C. giraffe",
        ),
        (
            "What is the animal of the block located at the first row, second column of the grid?",
            "A. rabbit | B. dog | C. cat | D. elephant",
            "A. rabbit",
        ),
    ],
]

SPATIALMAP_DATA = [
    # example 0 — map0
    [
        (
            "In which direction is Planetarium Prints relative to Police Supply Store?",
            "A. Northeast | B. Northwest | C. Southwest | D. Southeast",
            "A. Northeast",
        ),
        (
            "Which object is in the Southwest of Ice Queen Ice Cream?",
            "A. Coral Crafts | B. Narwhal's Novelties | C. Planetarium Prints | D. Police Supply Store",
            "B. Narwhal's Novelties",
        ),
        (
            "How many objects are in the Southeast of Oz Oddities?",
            "A. 1 | B. 5 | C. 3 | D. 0",
            "D. 0",
        ),
    ],
    # example 1 — map1
    [
        (
            "In which direction is Wolf's Wardrobe relative to Tremor Toys?",
            "A. Southwest | B. Northeast | C. Southeast | D. Northwest",
            "C. Southeast",
        ),
        (
            "Which object is in the Southwest of Fresh Foods?",
            "A. Wolf's Wardrobe | B. Salmon Sushi | C. Mantis's Maps | D. Tremor Toys",
            "D. Tremor Toys",
        ),
        (
            "How many objects are in the Southwest of Mantis's Maps?",
            "A. 5 | B. 3 | C. 2 | D. 1",
            "D. 1",
        ),
    ],
    # example 2 — map2
    [
        (
            "In which direction is Narwhal's Novels relative to Iris's Ice Skates?",
            "A. Northwest | B. Northeast | C. Southeast | D. Southwest",
            "B. Northeast",
        ),
        (
            "Which object is in the Southwest of Iris's Ice Skates?",
            "A. Art Supplies | B. Mordor Supplies | C. Andy's Autos | D. Narwhal's Novels",
            "A. Art Supplies",
        ),
        (
            "How many objects are in the Southeast of Mordor Supplies?",
            "A. 0 | B. 4 | C. 2 | D. 5",
            "C. 2",
        ),
    ],
    # example 3 — map3
    [
        (
            "In which direction is Recycle Center relative to Nightingale Novelties?",
            "A. Southeast | B. Southwest | C. Northeast | D. Northwest",
            "C. Northeast",
        ),
        (
            "Which object is in the Southwest of Andy's Autos?",
            "A. Recycle Center | B. Trail Hiking Gear | C. Sally's Salon | D. Unicorn Umbrellas",
            "D. Unicorn Umbrellas",
        ),
        (
            "How many objects are in the Southeast of Nightingale Novelties?",
            "A. 0 | B. 1 | C. 2 | D. 5",
            "B. 1",
        ),
    ],
    # example 4 — map4
    [
        (
            "In which direction is Arctic Apparel relative to Mantis Maternity?",
            "A. Southwest | B. Northeast | C. Northwest | D. Southeast",
            "C. Northwest",
        ),
        (
            "Which object is in the Northwest of Peet's Coffee?",
            "A. Prairie Provisions | B. Arctic Apparel | C. Raven's Records | D. Zodiac Zumba",
            "C. Raven's Records",
        ),
        (
            "How many objects are in the Northeast of Arctic Apparel?",
            "A. 1 | B. 3 | C. 2 | D. 0",
            "B. 3",
        ),
    ],
    # example 5 — map5
    [
        (
            "In which direction is Hogwarts Magic Supplies relative to Zephyr Zucchini?",
            "A. Northeast | B. Southeast | C. Northwest | D. Southwest",
            "C. Northwest",
        ),
        (
            "Which object is in the Northeast of Zephyr Zucchini?",
            "A. Zorilla's Zero-Waste Goods | B. X-U Souvenirs | C. Factory Finds | D. Hogwarts Magic Supplies",
            "C. Factory Finds",
        ),
        (
            "How many objects are in the Southeast of Zephyr Zucchini?",
            "A. 5 | B. 1 | C. 0 | D. 3",
            "C. 0",
        ),
    ],
    # example 6 — map6
    [
        (
            "In which direction is Albatross's Astronomy Accessories relative to Lumber's Marketplace?",
            "A. Northwest | B. Northeast | C. Southwest | D. Southeast",
            "D. Southeast",
        ),
        (
            "Which object is in the Southwest of Factory Finds?",
            "A. Wonders Candle Shop | B. Albatross's Astronomy Accessories | C. Fresh Foods | D. Titan Tailoring",
            "C. Fresh Foods",
        ),
        (
            "How many objects are in the West of Lumber's Marketplace?",
            "A. 0 | B. 3 | C. 5 | D. 1",
            "A. 0",
        ),
    ],
    # example 7 — map7
    [
        (
            "In which direction is Hummingbird Hats relative to Arctic Apparel?",
            "A. Southwest | B. Northeast | C. Southeast | D. Northwest",
            "D. Northwest",
        ),
        (
            "Which object is in the Southwest of Arctic Apparel?",
            "A. Xenopus's Xylophones | B. Hummingbird Hats | C. Junkyard Jewels | D. Pool Hall Provisions",
            "A. Xenopus's Xylophones",
        ),
        (
            "How many objects are in the Northwest of Xenopus's Xylophones?",
            "A. 4 | B. 3 | C. 0 | D. 2",
            "D. 2",
        ),
    ],
    # example 8 — map8
    [
        (
            "In which direction is Umbrella Universe relative to Quokka's Quilts?",
            "A. Southwest | B. Northwest | C. Southeast | D. Northeast",
            "A. Southwest",
        ),
        (
            "Which object is in the Southeast of Planetarium Prints?",
            "A. Umbrella Universe | B. Cheetah's Chocolates | C. Arctic Apparel | D. Quokka's Quilts",
            "C. Arctic Apparel",
        ),
        (
            "How many objects are in the South of Arctic Apparel?",
            "A. 5 | B. 2 | C. 3 | D. 0",
            "D. 0",
        ),
    ],
    # example 9 — map9
    [
        (
            "In which direction is Tiger's Tapestries relative to K University?",
            "A. Southwest | B. Southeast | C. Northwest | D. Northeast",
            "A. Southwest",
        ),
        (
            "Which object is in the Northeast of K University?",
            "A. Tiger's Tapestries | B. Bolt Books | C. Safari Supplies | D. Bookmark Bookstore",
            "C. Safari Supplies",
        ),
        (
            "How many objects are in the Northwest of K University?",
            "A. 3 | B. 0 | C. 5 | D. 2",
            "A. 3",
        ),
    ],
]

TASK_CONFIG = {
    "mazenav": {
        "data": MAZENAV_DATA,
        "image_prefix": "maze",
        "image_suffix": "_q0.png",
    },
    "spatialgrid": {
        "data": SPATIALGRID_DATA,
        "image_prefix": "grid",
        "image_suffix": "_q0.png",
    },
    "spatialmap": {
        "data": SPATIALMAP_DATA,
        "image_prefix": "map",
        "image_suffix": "_q0.png",
    },
}


def make_example_txt(n: int, qs: list) -> str:
    lines = [f"# Example image {n}", ""]
    for i, (question, options, answer) in enumerate(qs):
        lines.append(f"Question {i + 1} (q{i}): {question}")
        lines.append(f"Options: {options}")
        lines.append(f"Answer: {answer}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    for task, cfg in TASK_CONFIG.items():
        out_dir = os.path.join(BASE_DIR, task)
        os.makedirs(out_dir, exist_ok=True)

        assets_dir = os.path.join(ASSETS_BASE, task, "assets")

        for n, qs in enumerate(cfg["data"]):
            # Write .txt
            txt_path = os.path.join(out_dir, f"example_{n}.txt")
            with open(txt_path, "w") as f:
                f.write(make_example_txt(n, qs))
            print(f"  wrote {txt_path}")

            # Create symlink .png → actual image
            img_name = f"{cfg['image_prefix']}{n}{cfg['image_suffix']}"
            src_img = os.path.join(assets_dir, img_name)
            dst_link = os.path.join(out_dir, f"example_{n}.png")

            if os.path.lexists(dst_link):
                os.remove(dst_link)
            os.symlink(src_img, dst_link)
            print(f"  linked {dst_link} -> {src_img}")

        print(f"[{task}] Done — {len(cfg['data'])} examples created.")

    print("\nAll tasks done.")


if __name__ == "__main__":
    main()
