"""
Modele lineaire pour predire qualite JPEG optimale
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle


class ModeleQualiteJPEG:
    
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.est_entralne = False
        self.historique_images = []
    
    def extraire_caracteristiques(self, img_path):
        img = Image.open(img_path).convert('RGB')
        arr = np.array(img, dtype=np.float32)
        
        w, h = img.size
        ratio_aspect = w / h if h != 0 else 1
        variation = np.std(arr)
        pixels_flat = arr.reshape(-1, 3)
        entropie = np.std(pixels_flat)
        
        pixels_reduced = np.round(arr / 32) * 32
        colors = len(np.unique(pixels_reduced.reshape(-1, 3), axis=0))
        colors_ratio = colors / (w * h) if w * h > 0 else 0
        
        return np.array([w, h, ratio_aspect, variation, entropie, colors_ratio])
    
    def entrainer_sur_rapport(self, rapport_path="rapport_compression.json"):
        if not Path(rapport_path).exists():
            print(f"Rapport non trouve: {rapport_path}")
            return False
        
        with open(rapport_path, 'r') as f:
            rapport = json.load(f)
        
        X = []
        y = []
        
        for img_info in rapport.get('images', []):
            fichier = img_info['fichier']
            qualite = img_info['qualite_jpg']
            img_path = f"_in/{fichier}"
            
            if Path(img_path).exists():
                features = self.extraire_caracteristiques(img_path)
                X.append(features)
                y.append(qualite)
                self.historique_images.append({
                    'fichier': fichier,
                    'features': features,
                    'qualite': qualite,
                    'similarite': img_info['similarite']
                })
        
        if len(X) < 2:
            print("Pas assez d'images pour entrainer")
            return False
        
        X = np.array(X)
        y = np.array(y)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.est_entralne = True
        
        print(f"Modele entraine sur {len(X)} images")
        print(f"R2 score: {self.model.score(X_scaled, y):.4f}")
        return True
    
    def predire_qualite(self, img_path):
        if not self.est_entralne:
            return None
        
        features = self.extraire_caracteristiques(img_path)
        features_scaled = self.scaler.transform([features])[0]
        qualite_predite = self.model.predict([features_scaled])[0]
        return int(np.clip(qualite_predite, 15, 95))
    
    def sauvegarder(self, path="modele.pkl"):
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'est_entralne': self.est_entralne,
            'historique': self.historique_images
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Modele sauvegarde: {path}")
    
    def charger(self, path="modele.pkl"):
        if Path(path).exists():
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.est_entralne = data['est_entralne']
            self.historique_images = data['historique']
            print(f"Modele charge: {path}")
            return True
        return False
    
    def visualiser_predictions(self):
        if not self.historique_images:
            return
        
        fichiers = [img['fichier'] for img in self.historique_images]
        qualites_reelles = [img['qualite'] for img in self.historique_images]
        
        qualites_predites = []
        for img_info in self.historique_images:
            features_scaled = self.scaler.transform([img_info['features']])[0]
            pred = int(np.clip(self.model.predict([features_scaled])[0], 15, 95))
            qualites_predites.append(pred)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        x = np.arange(len(fichiers))
        ax = axes[0, 0]
        ax.bar(x - 0.2, qualites_reelles, 0.4, label='Reelle', alpha=0.8)
        ax.bar(x + 0.2, qualites_predites, 0.4, label='Predite', alpha=0.8)
        ax.set_ylabel('Qualite JPEG')
        ax.set_title('Qualite: Reelle vs Predite')
        ax.set_xticks(x)
        ax.set_xticklabels([f.split('.')[0][:12] for f in fichiers], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        erreurs = np.array(qualites_predites) - np.array(qualites_reelles)
        ax = axes[0, 1]
        ax.bar(x, erreurs, color=['red' if e > 0 else 'green' for e in erreurs])
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_ylabel('Erreur')
        ax.set_title('Erreurs de Prediction')
        ax.set_xticks(x)
        ax.set_xticklabels([f.split('.')[0][:12] for f in fichiers], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        similarites = [img['similarite'] for img in self.historique_images]
        ax = axes[1, 0]
        scatter = ax.scatter(qualites_reelles, similarites, s=100, alpha=0.6, c=qualites_reelles, cmap='RdYlGn')
        ax.axhline(y=0.85, color='red', linestyle='--', label='Seuil')
        ax.set_xlabel('Qualite JPEG')
        ax.set_ylabel('Similarite')
        ax.set_title('Similarite vs Qualite')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Qualite')
        
        ax = axes[1, 1]
        ax.axis('off')
        score = self.model.score(self.scaler.transform([img['features'] for img in self.historique_images]), 
                                 [img['qualite'] for img in self.historique_images])
        stats_text = f"""
STATISTIQUES

Images: {len(self.historique_images)}
R2 score: {score:.4f}

Erreur moyenne: {np.mean(np.abs(erreurs)):.2f}
Erreur max: {np.max(np.abs(erreurs)):.2f}

Qual. moy pred: {np.mean(qualites_predites):.1f}
Qual. moy reelle: {np.mean(qualites_reelles):.1f}

Similarite moy: {np.mean(similarites):.4f}
Min similarite: {np.min(similarites):.4f}
        """
        ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('_out/modele_predictions.png', dpi=150, bbox_inches='tight')
        print("Graphique sauvegarde: _out/modele_predictions.png")
        plt.close()


if __name__ == "__main__":
    modele = ModeleQualiteJPEG()
    
    if modele.entrainer_sur_rapport("_out/rapport_compression.json"):
        modele.sauvegarder("modele.pkl")
        modele.visualiser_predictions()
        print("Modele pret pour predictions")
    else:
        print("Erreur lors de l'entrainement")
