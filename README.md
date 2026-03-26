# Préparation des Données d'Images

Script Python pour redimensionner et compresser les images.

## Utilisation

```bash
python3 main.py
```

Ajoute images dans `_in/`. Résultats dans `_out/`.

## Fonctionnalités

- Redimensionne à **224×224 pixels** exactement
- Préserve le ratio d'aspect avec padding blanc
- Compresse en JPEG avec qualité optimale
- Teste 9 niveaux de qualité pour minimiser la taille

## Configuration

`main.py`:
```python
TAILLE_CIBLE = (224, 224)
QUALITES_A_TESTER = [95, 85, 75, 65, 55, 45, 35, 25, 15]
SIMILARITE_MIN = 0.95
```
