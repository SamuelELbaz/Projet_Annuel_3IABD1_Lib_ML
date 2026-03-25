"""
Visualisation des rapports de compression
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt


def charger_rapport(chemin_rapport="_out/rapport_compression.json"):
    with open(chemin_rapport, 'r', encoding='utf-8') as f:
        return json.load(f)


def afficher_graphiques(chemin_rapport="_out/rapport_compression.json"):
    rapport = charger_rapport(chemin_rapport)
    images = rapport['images']
    resume = rapport['resume']
    
    fichiers = [img['fichier'] for img in images]
    qualites = [img['qualite_jpg'] for img in images]
    similarites = [img['similarite'] for img in images]
    reductions = [img['reduction_percent'] for img in images]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Analyse de Compression', fontsize=14, fontweight='bold')
    
    axes[0, 0].bar(range(len(qualites)), qualites, color='steelblue')
    axes[0, 0].set_title('Qualite JPEG')
    axes[0, 0].set_ylabel('Qualite')
    axes[0, 0].set_ylim(0, 100)
    
    axes[0, 1].bar(range(len(similarites)), similarites, color='green', alpha=0.7)
    axes[0, 1].axhline(y=0.85, color='r', linestyle='--', label='Seuil')
    axes[0, 1].set_title('Similarite apres compression')
    axes[0, 1].set_ylabel('Similarite')
    axes[0, 1].set_ylim(0.7, 1.0)
    axes[0, 1].legend()
    
    axes[1, 0].bar(range(len(reductions)), reductions, color='orange')
    axes[1, 0].set_title('Reduction de taille')
    axes[1, 0].set_ylabel('Reduction (%)')
    
    axes[1, 1].axis('off')
    texte = f"""
RESUME
════════════════════
Images: {resume['images_traitees']}
Init: {resume['taille_initiale_bytes'] / (1024*1024):.2f} MB
Final: {resume['taille_finale_bytes'] / (1024*1024):.2f} MB

Reduction: {resume['reduction_percent']:.1f}%
Qualite moy: {sum(qualites)/len(qualites):.1f}
Similarite moy: {sum(similarites)/len(similarites):.4f}
    """
    axes[1, 1].text(0.5, 0.5, texte, ha='center', va='center', 
                   fontsize=10, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('_out/analyse_compression.png', dpi=150, bbox_inches='tight')
    print("Graphique sauvegarde: _out/analyse_compression.png")
    plt.show()


if __name__ == "__main__":
    afficher_graphiques()
