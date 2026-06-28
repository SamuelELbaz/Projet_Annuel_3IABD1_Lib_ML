import os
from icrawler.builtin import BingImageCrawler

# =========================================================
# Scraper COMPOST uniquement — nouveaux keywords
# Nommage automatique : compost_XXXX.jpg
# Continue depuis le dernier numéro existant
# =========================================================

DOSSIER = 'compost'
os.makedirs(DOSSIER, exist_ok=True)

filters = dict(type='photo', size='large')

# =========================================================
# Nouveaux keywords — noms de fruits/légumes pourris
# (aucun keyword déjà utilisé dans scraper.py)
# =========================================================

keywords_compost_new = [

    # Fruits pourris
    'mangue pourrie moisie déchet',
    'poire pourrie moisie compost',
    'pêche abricot pourri moisi',
    'prune pourrie moisie déchet organique',
    'cerise moisie pourrie compost',
    'kiwi pourri moisi biodéchet',
    'ananas pourri moisi déchet',
    'pastèque pourrie moisie compost',
    'figue pourrie moisie biodéchet',
    'grenade moisie pourrie compost',
    'citron moisi pourri déchet',
    'avocat noirci pourri déchet organique',
    'papaye moisie pourrie compost',
    'litchi moisi pourri biodéchet',

    # Légumes pourris
    'courgette pourrie moisie compost',
    'aubergine pourrie moisie déchet',
    'poivron moisi pourri biodéchet',
    'brocoli jauni pourri compost',
    'chou fleur pourri moisi déchet',
    'épinards pourris flétris compost',
    'céleri pourri moisi biodéchet',
    'poireau pourri flétri compost',
    'radis pourri moisi déchet organique',
    'navet pourri moisi compost',
    'betterave pourrie moisie biodéchet',
    'artichaut pourri moisi compost',
    'fenouil pourri flétri biodéchet',
    'haricots verts pourris moisis compost',
    'petits pois pourris moisis déchet',
    'maïs pourri moisi biodéchet',

    # Anglais — fruits
    'rotten mango moldy organic waste',
    'rotten pear moldy compost',
    'rotten peach apricot moldy waste',
    'rotten plum moldy organic compost',
    'rotten cherries moldy food waste',
    'rotten kiwi moldy biodegradable',
    'rotten pineapple moldy compost',
    'rotten watermelon moldy waste',
    'rotten fig moldy organic waste',
    'rotten lemon moldy compost',
    'rotten avocado blackened waste',
    'rotten papaya moldy compost',

    # Anglais — légumes
    'rotten zucchini moldy compost',
    'rotten eggplant moldy waste',
    'rotten pepper moldy biodegradable',
    'rotten broccoli yellowed compost',
    'rotten cauliflower moldy waste',
    'wilted spinach rotten compost',
    'rotten celery moldy compost',
    'rotten leek wilted compost',
    'rotten radish moldy organic',
    'rotten beet moldy compost',
    'rotten artichoke moldy waste',
    'rotten green beans moldy compost',
    'rotten corn moldy biodegradable waste',
]

# =========================================================
# Déterminer le prochain numéro à partir des fichiers existants
# =========================================================

def get_next_index(dossier):
    extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
    fichiers = [f for f in os.listdir(dossier) if os.path.splitext(f)[1].lower() in extensions]
    return len(fichiers) + 1

# =========================================================
# Fonction de scraping + renommage immédiat
# =========================================================

def scrape_et_renommer(keywords, dossier, max_par_keyword=50):

    for keyword in keywords:
        print(f"\n🔎 Recherche : {keyword}")

        # Index avant le crawl
        idx_avant = get_next_index(dossier)

        crawler = BingImageCrawler(
            storage={'root_dir': dossier},
            downloader_threads=4,
            parser_threads=2,
        )

        try:
            crawler.crawl(
                keyword=keyword,
                filters=filters,
                max_num=max_par_keyword,
                file_idx_offset='auto'
            )
        except Exception as e:
            print(f"  ⚠️ Erreur : {e}")

        # Renommer les nouvelles images téléchargées
        extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
        nouveaux = sorted([
            f for f in os.listdir(dossier)
            if os.path.splitext(f)[1].lower() in extensions
            and not f.startswith('compost_')
        ])

        for nom in nouveaux:
            ext = os.path.splitext(nom)[1].lower()
            if ext == '.jpeg':
                ext = '.jpg'
            idx = get_next_index(dossier) - len([
                f for f in os.listdir(dossier)
                if not f.startswith('compost_')
                and os.path.splitext(f)[1].lower() in extensions
            ])
            # Trouver le bon numéro
            i = idx_avant
            while os.path.exists(os.path.join(dossier, f"compost_{i:04d}{ext}")):
                i += 1

            ancien = os.path.join(dossier, nom)
            nouveau = os.path.join(dossier, f"compost_{i:04d}{ext}")
            os.rename(ancien, nouveau)
            idx_avant = i + 1

        total = len([f for f in os.listdir(dossier) if os.path.splitext(f)[1].lower() in extensions])
        print(f"  📁 Total compost : {total} images")


# =========================================================
# Lancement
# =========================================================

print("\n==============================")
print("🌱 Scraping COMPOST (nouveaux keywords)")
print("==============================")

scrape_et_renommer(keywords_compost_new, DOSSIER)

total_final = len([
    f for f in os.listdir(DOSSIER)
    if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
])

print(f"\n✅ Terminé ! Total compost : {total_final} images")
