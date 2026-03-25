# Quick Start

## Installation

```bash
pip install pillow numpy matplotlib scikit-learn tqdm
```

## Utilisation

### Compression simple
```bash
python3 main.py
```

### Avec modèle ML
```bash
python3 pipeline_optimise.py
```

### Analyser
```bash
python3 analyse.py
```

## Préparation

```bash
mkdir -p _in
cp images/*.jpg _in/
```

## Configuration

Éditer `main.py`:
```python
TAILLE_CIBLE = (224, 224)
QUALITES_A_TESTER = [95, 85, 75, 65, 55, 45, 35, 25, 15]
SIMILARITE_MIN = 0.85
```
