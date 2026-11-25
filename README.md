# KD-SCA: Side-Channel Analysis with Knowledge Distillation

A TensorFlow/Keras research reproduction framework implementing Knowledge Distillation for Profiled Side-Channel Analysis (KD-SCA).

## Project Structure

```text
mtp_project/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── configs/
│   ├── teacher/
│   │   └── ascad_fixed.yaml
│   ├── student/
│   │   └── ascad_fixed.yaml
│   └── kd/
│       └── ascad_fixed.yaml
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
│
├── notebooks/
│   ├── 01_data_preprocessing/
│   ├── 02_teacher_training/
│   ├── 03_student_baseline/
│   ├── 04_distillation/
│   └── 05_evaluation/
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset_loader.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── teacher/
│   │   │   └── __init__.py
│   │   └── student/
│   │       └── __init__.py
│   ├── losses/
│   │   ├── __init__.py
│   │   └── distillation_loss.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── teacher_trainer.py
│   │   ├── student_trainer.py
│   │   └── kd_trainer.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── guessing_entropy.py
│   │   └── correlation.py
│   └── utils/
│       ├── __init__.py
│       ├── seed.py
│       └── config.py
│
├── checkpoints/
│   ├── teacher/.gitkeep
│   ├── student_baseline/.gitkeep
│   └── student_kd/.gitkeep
│
└── results/
    ├── figures/
    └── logs/
```

## Setup & Installation

1. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
