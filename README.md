# GenAI Lab

A unified Flask portal for the ten projects in `ALL PROJECTS`. The portal provides a dashboard, one dedicated page per experiment (`/projects/01` through `/projects/10`), a project index API, and safe downloads for each project's README, notebooks, and app entry point when present.

The original experiments remain in their project folders. This portal does not import or start their machine-learning models, so it can launch without downloading model weights or installing each experiment's separate dependencies.

## Run locally

Python 3.10+ and Flask are required. From the repository root:

```powershell
python -m pip install Flask
python app.py
```

Open <http://127.0.0.1:5000>. Set the `PORT` environment variable to use another port.

## Routes

| Route | Purpose |
| --- | --- |
| `/` | Searchable dashboard for all ten experiments |
| `/projects/01` – `/projects/10` | Dedicated experiment pages |
| `/api/projects` | JSON project index |
| `/health` | Portal health check |
| `/files/<project_id>/<filename>` | Download an allowlisted project README, notebook, or app source |

## Unified Project Runner CLI (`run_project.py`)

You can launch, test, or inspect any project directly from the root using the unified CLI:

```powershell
# List all 10 projects and status
python run_project.py list

# Run any interactive project studio (projects 4 to 10)
python run_project.py 4    # Text-to-Image Diffusion (Port 5004)
python run_project.py 5    # Conditional VAE MNIST Digit Studio (Port 5005)
python run_project.py 6    # DCGAN Face Generation Studio (Port 5006)
python run_project.py 7    # StyleGAN Face Studio Gallery (Port 8507)
python run_project.py 8    # MusicGen Audio Studio (Port 5001)
python run_project.py 9    # Personalized AI Assistant RAG (Port 8509)
python run_project.py 10   # Text-to-Video Foundation Studio (Port 5002)

# Run the GenAI Lab Web Portal (Port 5000)
python run_project.py portal

# Run all 38 automated test suites across all projects
python run_project.py test
```

## Running Projects Inside `ALL PROJECTS`

All files, dependencies, models, and weights are kept directly inside `ALL PROJECTS`. Each project can also be run from its own directory:

| Experiment | Directory | Run Command | Default Port |
| :--- | :--- | :--- | :--- |
| **EXP 01** | `ALL PROJECTS/EXP_01_Exploratory_Data_Analysis` | `jupyter notebook Gen_AI_EXP_1.ipynb` | Jupyter |
| **EXP 02** | `ALL PROJECTS/EXP_02_Text_Generation_and_QA` | `jupyter notebook Gen_AI_Exp_2.ipynb` | Jupyter |
| **EXP 03** | `ALL PROJECTS/EXP_03_Tensor_Operations_and_Autograd` | `jupyter notebook GEN_AIEXP_3.ipynb` | Jupyter |
| **EXP 04** | `ALL PROJECTS/EXP_04_Text_to_Image_Diffusion` | `python run.py --port 5004` | `http://127.0.0.1:5004` |
| **EXP 05** | `ALL PROJECTS/EXP_05_Conditional_VAE_MNIST` | `python run.py --port 5005` | `http://127.0.0.1:5005` |
| **EXP 06** | `ALL PROJECTS/EXP_06_DCGAN_Face_Generation` | `python run.py --port 5006` | `http://127.0.0.1:5006` |
| **EXP 07** | `ALL PROJECTS/EXP_07_StyleGAN_Face_Studio` | `python run.py --port 8507` | `http://127.0.0.1:8507` |
| **EXP 08** | `ALL PROJECTS/EXP_08_Audio_Generation_MusicGen` | `python run.py --port 5001` | `http://127.0.0.1:5001` |
| **EXP 09** | `ALL PROJECTS/EXP_09_Personalized_AI_Assistant_RAG` | `python run.py --port 8509` | `http://127.0.0.1:8509` |
| **EXP 10** | `ALL PROJECTS/EXP_10_Text_to_Video_Foundation_Model` | `python run.py --port 5002` | `http://127.0.0.1:5002` |

