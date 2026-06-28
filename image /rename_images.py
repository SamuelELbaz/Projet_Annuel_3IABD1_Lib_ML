import os

# =========================================================
# Renommage des images par catégorie
# Format : compost_001.jpg, recyclable_001.jpg, etc.
# =========================================================

CATEGORIES = ['recyclable', 'compost', 'dechets_chimiques', 'menagere']

EXTENSIONS_VALIDES = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}

for categorie in CATEGORIES:
    dossier = os.path.join(os.path.dirname(__file__), categorie)

    if not os.path.exists(dossier):
        print(f"⚠️  Dossier '{categorie}' introuvable, ignoré.")
        continue

    # Récupérer tous les fichiers images
    fichiers = sorted([
        f for f in os.listdir(dossier)
        if os.path.splitext(f)[1].lower() in EXTENSIONS_VALIDES
    ])

    if not fichiers:
        print(f"⚠️  Aucune image dans '{categorie}'.")
        continue

    print(f"\n📁 {categorie} — {len(fichiers)} images à renommer...")

    # Renommage en deux passes pour éviter les conflits
    # Passe 1 : renommer en .tmp
    for i, nom_actuel in enumerate(fichiers, start=1):
        ext = os.path.splitext(nom_actuel)[1].lower()
        if ext == '.jpeg':
            ext = '.jpg'
        ancien = os.path.join(dossier, nom_actuel)
        tmp = os.path.join(dossier, f"__tmp_{i:04d}{ext}")
        os.rename(ancien, tmp)

    # Passe 2 : renommer en nom final
    tmp_fichiers = sorted([
        f for f in os.listdir(dossier)
        if f.startswith('__tmp_')
    ])

    for i, nom_tmp in enumerate(tmp_fichiers, start=1):
        ext = os.path.splitext(nom_tmp)[1].lower()
        ancien = os.path.join(dossier, nom_tmp)
        nouveau_nom = f"{categorie}_{i:04d}{ext}"
        nouveau = os.path.join(dossier, nouveau_nom)
        os.rename(ancien, nouveau)

    print(f"   ✅ Renommé : {categorie}_0001{ext} → {categorie}_{len(tmp_fichiers):04d}{ext}")

print("\n🎉 Renommage terminé !")
