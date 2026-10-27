"""
run_project.py
==============
Unified CLI manager to launch, inspect, or test any experiment in `ALL PROJECTS`.

Usage:
  python run_project.py 4       # Run Experiment 4 (Text-to-Image Diffusion) on http://127.0.0.1:5004
  python run_project.py 5       # Run Experiment 5 (Conditional VAE MNIST) on http://127.0.0.1:5005
  python run_project.py 6       # Run Experiment 6 (DCGAN Face Generation) on http://127.0.0.1:5006
  python run_project.py 7       # Run Experiment 7 (StyleGAN Face Studio) on http://127.0.0.1:8507
  python run_project.py 8       # Run Experiment 8 (Audio Generation MusicGen) on http://127.0.0.1:5001
  python run_project.py 9       # Run Experiment 9 (Personalized AI Assistant RAG) on http://127.0.0.1:8509
  python run_project.py 10      # Run Experiment 10 (Text-to-Video Foundation Model) on http://127.0.0.1:5002

  python run_project.py portal  # Run unified GenAI Lab Portal on http://127.0.0.1:5000
  python run_project.py test    # Run automated test suite across all projects (38 tests)
  python run_project.py list    # Show status and details of all 10 experiments
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = BASE_DIR / "ALL PROJECTS"

PROJECT_REGISTRY = {
    1: {
        "title": "Exploratory Data Analysis",
        "folder": "EXP_01_Exploratory_Data_Analysis",
        "type": "notebook",
        "notebook": "Gen_AI_EXP_1.ipynb",
    },
    2: {
        "title": "Text Generation & Q&A",
        "folder": "EXP_02_Text_Generation_and_QA",
        "type": "notebook",
        "notebook": "Gen_AI_Exp_2.ipynb",
    },
    3: {
        "title": "Tensor Operations & Autograd",
        "folder": "EXP_03_Tensor_Operations_and_Autograd",
        "type": "notebook",
        "notebook": "GEN_AIEXP_3.ipynb",
    },
    4: {
        "title": "Text-to-Image Diffusion (Stable Diffusion v1.5)",
        "folder": "EXP_04_Text_to_Image_Diffusion",
        "type": "app",
        "default_port": 5004,
        "runner": "run.py",
        "args": ["--port", "5004"],
    },
    5: {
        "title": "Conditional VAE MNIST Digit Studio",
        "folder": "EXP_05_Conditional_VAE_MNIST",
        "type": "app",
        "default_port": 5005,
        "runner": "run.py",
        "args": ["--port", "5005"],
    },
    6: {
        "title": "DCGAN CelebA Face Generation Studio",
        "folder": "EXP_06_DCGAN_Face_Generation",
        "type": "app",
        "default_port": 5006,
        "runner": "run.py",
        "args": ["--port", "5006"],
    },
    7: {
        "title": "StyleGAN Face Studio Gallery",
        "folder": "EXP_07_StyleGAN_Face_Studio",
        "type": "app",
        "default_port": 8507,
        "runner": "run.py",
        "args": ["--port", "8507"],
    },
    8: {
        "title": "MusicGen Audio Studio",
        "folder": "EXP_08_Audio_Generation_MusicGen",
        "type": "app",
        "default_port": 5001,
        "runner": "run.py",
        "args": ["--port", "5001"],
    },
    9: {
        "title": "Personalized AI Assistant (RAG)",
        "folder": "EXP_09_Personalized_AI_Assistant_RAG",
        "type": "app",
        "default_port": 8509,
        "runner": "run.py",
        "args": ["--port", "8509"],
    },
    10: {
        "title": "Text-to-Video Foundation Studio",
        "folder": "EXP_10_Text_to_Video_Foundation_Model",
        "type": "app",
        "default_port": 5002,
        "runner": "run.py",
        "args": ["--port", "5002"],
    },
}


def list_projects():
    print("\n" + "=" * 78)
    print("                    GENAI LAB — ALL PROJECTS STATUS")
    print("=" * 78)
    for pid in range(1, 11):
        info = PROJECT_REGISTRY[pid]
        p_dir = PROJECTS_DIR / info["folder"]
        exists = p_dir.is_dir()
        status = "READY" if exists else "MISSING"

        if info["type"] == "notebook":
            print(f" [{pid:02d}] {info['title']:<40} [Notebook] {info['notebook']} ({status})")
        else:
            port = info.get("default_port", "-")
            runner = p_dir / info["runner"]
            run_status = "RUNNABLE" if runner.is_file() else "NO RUNNER"
            print(f" [{pid:02d}] {info['title']:<40} [Port {port}]  python run.py ({run_status})")
    print("=" * 78 + "\n")


def run_tests():
    print("\n" + "=" * 78)
    print("       RUNNING AUTOMATED TEST SUITE ACROSS ALL PROJECTS (4 to 10)")
    print("=" * 78 + "\n")
    test_paths = [
        str(PROJECTS_DIR / "EXP_04_Text_to_Image_Diffusion" / "tests"),
        str(PROJECTS_DIR / "EXP_05_Conditional_VAE_MNIST" / "tests"),
        str(PROJECTS_DIR / "EXP_06_DCGAN_Face_Generation" / "tests"),
        str(PROJECTS_DIR / "EXP_07_StyleGAN_Face_Studio" / "tests"),
        str(PROJECTS_DIR / "EXP_08_Audio_Generation_MusicGen" / "tests"),
        str(PROJECTS_DIR / "EXP_09_Personalized_AI_Assistant_RAG" / "tests"),
        str(PROJECTS_DIR / "EXP_10_Text_to_Video_Foundation_Model" / "tests"),
    ]
    cmd = [sys.executable, "-m", "pytest", *test_paths, "-v"]
    res = subprocess.run(cmd, cwd=str(BASE_DIR))
    sys.exit(res.returncode)


def run_portal(port: int = 5000):
    print(f"\n* Launching GenAI Lab Portal on http://127.0.0.1:{port} ...\n")
    os.environ["PORT"] = str(port)
    cmd = [sys.executable, str(BASE_DIR / "app.py")]
    try:
        subprocess.run(cmd, cwd=str(BASE_DIR))
    except KeyboardInterrupt:
        print("\nPortal stopped.")


def run_project(target: str, extra_args: list[str]):
    if target in {"list", "status"}:
        list_projects()
        return

    if target in {"test", "tests"}:
        run_tests()
        return

    if target in {"portal", "main", "web", "0"}:
        run_portal()
        return

    try:
        pid = int(target)
    except ValueError:
        print(f"Error: Unknown target '{target}'. Use a number 4-10, 'portal', 'test', or 'list'.")
        sys.exit(1)

    if pid not in PROJECT_REGISTRY:
        print(f"Error: Project {pid} does not exist. Valid IDs: 1 to 10.")
        sys.exit(1)

    info = PROJECT_REGISTRY[pid]
    p_dir = PROJECTS_DIR / info["folder"]

    if not p_dir.is_dir():
        print(f"Error: Project directory not found: {p_dir}")
        sys.exit(1)

    if info["type"] == "notebook":
        print(f"\nProject {pid} is a Jupyter notebook: {info['notebook']}")
        print(f"Opening with Jupyter...")
        cmd = [sys.executable, "-m", "jupyter", "notebook", str(p_dir / info["notebook"])]
        subprocess.run(cmd, cwd=str(p_dir))
        return

    runner_path = p_dir / info["runner"]
    if not runner_path.is_file():
        print(f"Error: Runner script '{info['runner']}' not found in {p_dir}")
        sys.exit(1)

    cmd_args = list(info.get("args", []))
    if extra_args:
        cmd_args = extra_args

    print("\n" + "=" * 78)
    print(f"   STARTING EXPERIMENT {pid:02d}: {info['title'].upper()}")
    print("=" * 78)
    print(f"   Directory: {p_dir}")
    print(f"   Command:   {sys.executable} {info['runner']} {' '.join(cmd_args)}")
    print(f"   URL:       http://127.0.0.1:{info.get('default_port', '')}")
    print("=" * 78 + "\n")

    cmd = [sys.executable, str(runner_path), *cmd_args]
    try:
        subprocess.run(cmd, cwd=str(p_dir))
    except KeyboardInterrupt:
        print(f"\nProject {pid} stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="GenAI Lab Project Manager — Run, test, or inspect any project in ALL PROJECTS"
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="list",
        help="Project ID (1-10), 'portal', 'test', or 'list' (default: list)",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Optional arguments passed directly to the project runner",
    )

    args = parser.parse_args()
    run_project(args.target, args.extra_args)


if __name__ == "__main__":
    main()
