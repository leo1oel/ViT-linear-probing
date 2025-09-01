# ViT Linear Probing

This project provides a simple framework for evaluating the feature representation capabilities of vision models. The main functionalities include feature extraction and linear probing evaluation.

## Installation

1. Clone the repository:
```bash
git clone git@github.com:leo1oel/ViT-linear-probing.git
cd ViT-linear-probing
```

2. Install uv (if not already installed):
```bash
pip install uv
```

3. Create a virtual environment and install dependencies using uv:
```bash
uv venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
uv pip install -r requirements.txt
```

## Usage

### Configuration

All configurations are located in the `conf/` directory:
- `conf/config.yaml`: Main configuration file
- `conf/model/`: Contains specific configurations for different models

Key configuration items include:
- Data path configuration
- Feature extraction configuration
- Linear probing configuration
- Wandb configuration (optional)

### Running Evaluation

Use the following command to run the complete evaluation pipeline:
```bash
python evaluate.py model=dino  # Use DINO model configuration
```

To enable wandb logging:
```bash
python evaluate.py model=dino use_wandb=true
```

### Running Separately

You can also run feature extraction and linear probing separately:

1. Feature extraction:
```bash
python extract_features.py model=dino
```

2. Linear probing:
```bash
python linear_probe.py model=dino
```

## Project Structure

```
.
├── conf/                   # Configuration files directory
│   ├── config.yaml        # Main configuration file
│   └── model/            # Model-specific configurations
├── evaluate.py            # Main evaluation script
├── extract_features.py    # Feature extraction script
├── linear_probe.py        # Linear probing script
├── models.py             # Model definitions
└── feature_extractor.py  # Feature extractor implementation
```

## Extension

To add support for new models:
1. Add a new model configuration under `conf/model/`
2. Implement the corresponding model loading logic in `models.py`. If it's a Hugging Face model that uses the last layer CLS token as feature representation, no modification is needed
