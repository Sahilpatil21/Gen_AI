from __future__ import annotations

import os
from pathlib import Path
import sys

from flask import Flask, abort, jsonify, render_template, send_from_directory, url_for


BASE_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = BASE_DIR / "ALL PROJECTS"

PROJECTS = [
    {
        "id": 1,
        "folder": "EXP_01_Exploratory_Data_Analysis",
        "title": "Exploratory Data Analysis",
        "short_title": "Explore data",
        "category": "Data science",
        "summary": "Turn an unfamiliar dataset into clear, useful insights with a repeatable exploratory analysis workflow.",
        "description": "A notebook-led introduction to understanding datasets: inspect their shape and quality, summarize distributions, investigate relationships, and communicate findings with visualizations.",
        "icon": "01",
        "gradient": "mint",
        "topics": ["Data profiling", "Statistics", "Visualization"],
        "workflow": ["Load a dataset", "Inspect columns and missing values", "Explore distributions and relationships", "Summarize findings"],
    },
    {
        "id": 2,
        "folder": "EXP_02_Text_Generation_and_QA",
        "title": "Text Generation & Q&A",
        "short_title": "Generate text",
        "category": "Language models",
        "summary": "Explore transformer-powered text generation and question answering.",
        "description": "An introductory language-model experiment that explores generating text from prompts and answering questions with pretrained transformer models.",
        "icon": "02",
        "gradient": "blue",
        "topics": ["Transformers", "Text generation", "Question answering"],
        "workflow": ["Choose a prompt or question", "Run a pretrained language model", "Inspect and compare model responses"],
    },
    {
        "id": 3,
        "folder": "EXP_03_Tensor_Operations_and_Autograd",
        "title": "Tensor Operations & Autograd",
        "short_title": "Learn autograd",
        "category": "Deep learning",
        "summary": "Build intuition for tensors, computational graphs, and automatic differentiation.",
        "description": "A notebook-based tour of tensor operations and gradient calculation: follow values through computations and see how automatic differentiation supports model training.",
        "icon": "03",
        "gradient": "violet",
        "topics": ["Tensor operations", "Computational graphs", "Gradients"],
        "workflow": ["Create and transform tensors", "Build a differentiable computation", "Calculate gradients with autograd"],
    },
    {
        "id": 4,
        "folder": "EXP_04_Text_to_Image_Diffusion",
        "title": "Text-to-Image Diffusion",
        "short_title": "Create images",
        "category": "Image generation",
        "summary": "Create images from text prompts with a diffusion-model studio.",
        "description": "A Stable Diffusion-based image-generation project with a web studio, generation settings, and optional accelerated or offline generation paths.",
        "icon": "04",
        "gradient": "pink",
        "topics": ["Stable Diffusion", "Prompt design", "Image synthesis"],
        "workflow": ["Describe the image", "Choose a style and settings", "Generate and review image variations"],
    },
    {
        "id": 5,
        "folder": "EXP_05_Conditional_VAE_MNIST",
        "title": "Conditional VAE · MNIST",
        "short_title": "Draw digits",
        "category": "Generative models",
        "summary": "Generate MNIST-style digits by conditioning a variational autoencoder on a chosen label.",
        "description": "A conditional variational autoencoder (CVAE) that combines a sampled latent vector with a digit label to generate class-specific handwritten digit images.",
        "icon": "05",
        "gradient": "amber",
        "topics": ["Conditional VAE", "MNIST", "Latent space"],
        "workflow": ["Select a digit from 0 to 9", "Sample a point in latent space", "Decode a conditioned digit image"],
    },
    {
        "id": 6,
        "folder": "EXP_06_DCGAN_Face_Generation",
        "title": "DCGAN Face Generation",
        "short_title": "Generate faces",
        "category": "Generative models",
        "summary": "Explore face generation and latent-space morphing with a deep convolutional GAN.",
        "description": "A CelebA-inspired DCGAN experiment with a face-generation app and tools for sampling and exploring transitions through latent space.",
        "icon": "06",
        "gradient": "orange",
        "topics": ["DCGAN", "CelebA", "Latent morphing"],
        "workflow": ["Sample latent noise", "Generate synthetic faces", "Explore model outputs and latent transitions"],
    },
    {
        "id": 7,
        "folder": "EXP_07_StyleGAN_Face_Studio",
        "title": "StyleGAN Face Studio",
        "short_title": "Explore StyleGAN",
        "category": "Generative models",
        "summary": "Explore style-based face generation, mapping networks, and adaptive instance normalization.",
        "description": "A compact StyleGAN-inspired CelebA project featuring a mapping network, learned constant input, per-pixel noise, and style injection.",
        "icon": "07",
        "gradient": "violet",
        "topics": ["StyleGAN", "Mapping network", "AdaIN"],
        "workflow": ["Map latent noise into style space", "Inject styles into generator layers", "Inspect generated samples and training code"],
    },
    {
        "id": 8,
        "folder": "EXP_08_Audio_Generation_MusicGen",
        "title": "MusicGen Audio Studio",
        "short_title": "Compose music",
        "category": "Audio generation",
        "summary": "Explore text-conditioned music generation and original MusicGen compositions.",
        "description": "A MusicGen project with a browser-based studio, genre presets, generation history, and notebook experiments in music generation and fine-tuning.",
        "icon": "08",
        "gradient": "blue",
        "topics": ["MusicGen", "Text-to-audio", "Composition"],
        "workflow": ["Describe a musical idea", "Choose generation settings", "Generate and listen to a composition"],
    },
    {
        "id": 9,
        "folder": "EXP_09_Personalized_AI_Assistant_RAG",
        "title": "Personalized AI Assistant",
        "short_title": "Ask your documents",
        "category": "Retrieval-augmented generation",
        "summary": "Connect document retrieval with an open-source language model for grounded answers.",
        "description": "A research starter combining document ingestion, semantic retrieval, and an open-source instruction model, with optional LoRA/QLoRA experiments.",
        "icon": "09",
        "gradient": "mint",
        "topics": ["RAG", "Document search", "LoRA / QLoRA"],
        "workflow": ["Add your documents", "Retrieve relevant passages", "Generate an answer with source context"],
    },
    {
        "id": 10,
        "folder": "EXP_10_Text_to_Video_Foundation_Model",
        "title": "Text-to-Video Studio",
        "short_title": "Create video",
        "category": "Video generation",
        "summary": "Explore short text-to-video generation with a pretrained foundation model.",
        "description": "A text-to-video project with a generation studio, configurable visual presets, saved outputs, and an experiment notebook.",
        "icon": "10",
        "gradient": "pink",
        "topics": ["Text-to-video", "Foundation models", "Generation presets"],
        "workflow": ["Write a scene prompt", "Choose motion and visual settings", "Generate and preview a short video"],
    },
]

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))


def project_files(project: dict) -> list[dict[str, str]]:
    """Return only useful, directly downloadable project documentation and entry points."""
    folder = PROJECTS_DIR / project["folder"]
    files: list[dict[str, str]] = []
    readme = folder / "README.md"
    if readme.is_file():
        files.append({"name": "Project guide", "path": "README.md", "kind": "README"})

    for notebook in sorted(folder.rglob("*.ipynb")):
        relative_path = notebook.relative_to(folder).as_posix()
        files.append({"name": notebook.stem.replace("_", " "), "path": relative_path, "kind": "NOTEBOOK"})

    entrypoint = folder / "app.py"
    if entrypoint.is_file():
        files.append({"name": "App source", "path": "app.py", "kind": "SOURCE"})

    runner = folder / "run.py"
    if runner.is_file():
        files.append({"name": "Run script", "path": "run.py", "kind": "RUNNER"})
    return files


import atexit
import subprocess
import time
import urllib.request

RUN_CONFIGS = {
    1: {"runnable": False, "type": "notebook", "command": "jupyter notebook Gen_AI_EXP_1.ipynb"},
    2: {"runnable": False, "type": "notebook", "command": "jupyter notebook Gen_AI_Exp_2.ipynb"},
    3: {"runnable": False, "type": "notebook", "command": "jupyter notebook GEN_AIEXP_3.ipynb"},
    4: {"runnable": True, "type": "app", "command": "python run.py --port 5004", "port": 5004},
    5: {"runnable": True, "type": "app", "command": "python run.py --port 5005", "port": 5005},
    6: {"runnable": True, "type": "app", "command": "python run.py --port 5006", "port": 5006},
    7: {"runnable": True, "type": "app", "command": "python run.py --port 5007", "port": 5007},
    8: {"runnable": True, "type": "app", "command": "python run.py --port 5001", "port": 5001},
    9: {"runnable": True, "type": "app", "command": "python run.py --port 5009", "port": 5009},
    10: {"runnable": True, "type": "app", "command": "python run.py --port 5002", "port": 5002},
}


class StudioSupervisor:
    def __init__(self):
        self.processes: dict[int, subprocess.Popen] = {}

    def is_port_open(self, port: int) -> bool:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
            with urllib.request.urlopen(req, timeout=0.6) as resp:
                return resp.status == 200
        except Exception:
            return False

    def is_running(self, project_id: int) -> bool:
        proc = self.processes.get(project_id)
        if proc is not None:
            if proc.poll() is None:
                return True
            else:
                self.processes.pop(project_id, None)
        cfg = RUN_CONFIGS.get(project_id)
        if cfg and cfg.get("port"):
            return self.is_port_open(cfg["port"])
        return False

    def start(self, project_id: int) -> dict:
        cfg = RUN_CONFIGS.get(project_id)
        if not cfg or not cfg.get("runnable"):
            return {"success": False, "status": "unsupported", "error": f"Project {project_id} is not runnable"}

        port = cfg["port"]
        if self.is_running(project_id):
            return {"success": True, "status": "running", "port": port, "url": f"http://127.0.0.1:{port}"}

        project = next((p for p in PROJECTS if p["id"] == project_id), None)
        if not project:
            return {"success": False, "status": "not_found", "error": f"Project {project_id} not found"}

        p_dir = PROJECTS_DIR / project["folder"]
        runner = p_dir / "run.py"
        if not runner.is_file():
            return {"success": False, "status": "no_runner", "error": f"Runner not found in {p_dir}"}

        cmd = [sys.executable, str(runner), "--port", str(port)]
        env = os.environ.copy()
        env["PORT"] = str(port)
        env["PYTHONIOENCODING"] = "utf-8"

        proc = subprocess.Popen(
            cmd,
            cwd=str(p_dir),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.processes[project_id] = proc

        # Wait up to 3.5 seconds for port health check
        for _ in range(7):
            time.sleep(0.5)
            if self.is_port_open(port):
                break

        return {
            "success": True,
            "status": "running",
            "port": port,
            "url": f"http://127.0.0.1:{port}",
        }

    def stop(self, project_id: int) -> dict:
        proc = self.processes.pop(project_id, None)
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        return {"success": True, "status": "stopped"}

    def stop_all(self):
        for pid in list(self.processes.keys()):
            self.stop(pid)


supervisor = StudioSupervisor()
atexit.register(supervisor.stop_all)


def project_context(project: dict) -> dict:
    cfg = RUN_CONFIGS.get(project["id"], {})
    return {**project, "files": project_files(project), **cfg}


def show_project(project_id: int):
    project = next((item for item in PROJECTS if item["id"] == project_id), None)
    if project is None:
        abort(404)
    enriched = project_context(project)
    enriched["previous"] = next((item for item in PROJECTS if item["id"] == project_id - 1), None)
    enriched["next"] = next((item for item in PROJECTS if item["id"] == project_id + 1), None)
    return render_template(
        "project.html",
        project=enriched,
        projects=PROJECTS,
        project_url=lambda item: url_for(f"project_{item['id']:02d}"),
    )


@app.route("/")
def home():
    projects = [project_context(project) for project in PROJECTS]
    return render_template(
        "index.html",
        projects=projects,
        categories=sorted({project["category"] for project in PROJECTS}),
    )


@app.route("/api/projects")
def projects_api():
    return jsonify({"projects": [project_context(project) for project in PROJECTS], "count": len(PROJECTS)})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "projects": len(PROJECTS)})


@app.route("/api/studio/<int:project_id>/status")
def studio_status(project_id: int):
    cfg = RUN_CONFIGS.get(project_id, {})
    running = supervisor.is_running(project_id)
    port = cfg.get("port")
    return jsonify({
        "project_id": project_id,
        "runnable": cfg.get("runnable", False),
        "running": running,
        "port": port,
        "url": f"http://127.0.0.1:{port}" if port else None,
    })


@app.route("/api/studio/<int:project_id>/start", methods=["POST"])
def studio_start(project_id: int):
    return jsonify(supervisor.start(project_id))


@app.route("/api/studio/<int:project_id>/stop", methods=["POST"])
def studio_stop(project_id: int):
    return jsonify(supervisor.stop(project_id))


@app.route("/api/studio/statuses")
def studio_statuses():
    res = {}
    for pid, cfg in RUN_CONFIGS.items():
        if cfg.get("runnable"):
            res[pid] = {
                "running": supervisor.is_running(pid),
                "port": cfg.get("port"),
                "url": f"http://127.0.0.1:{cfg['port']}",
            }
    return jsonify(res)


@app.route("/api/studio/start-all", methods=["POST"])
def studio_start_all():
    results = {}
    for pid, cfg in RUN_CONFIGS.items():
        if cfg.get("runnable"):
            results[pid] = supervisor.start(pid)
    return jsonify({"success": True, "results": results})


@app.route("/api/studio/stop-all", methods=["POST"])
def studio_stop_all():
    supervisor.stop_all()
    return jsonify({"success": True, "status": "all_stopped"})


@app.route("/files/<int:project_id>/<path:filename>")
def project_file(project_id: int, filename: str):
    project = next((item for item in PROJECTS if item["id"] == project_id), None)
    if project is None:
        abort(404)
    allowed_files = {item["path"] for item in project_files(project)}
    if filename not in allowed_files:
        abort(404)
    return send_from_directory(str(PROJECTS_DIR / project["folder"]), filename, as_attachment=True)


def register_project_routes() -> None:
    for project in PROJECTS:
        endpoint = f"project_{project['id']:02d}"
        app.add_url_rule(
            f"/projects/{project['id']:02d}",
            endpoint=endpoint,
            view_func=lambda project_id=project["id"]: show_project(project_id),
        )


register_project_routes()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
