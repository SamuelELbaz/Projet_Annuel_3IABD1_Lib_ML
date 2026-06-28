import os
from icrawler.builtin import BingImageCrawler

# =========================================================
# Création des dossiers
# =========================================================

CATEGORIES = {
    'recyclable': 'recyclable',
    'compost': 'compost',
    'dechets_chimiques': 'dechets_chimiques',
    'menagere': 'menagere'
}

for folder in CATEGORIES.values():
    os.makedirs(folder, exist_ok=True)

# =========================================================
# Filtres images
# =========================================================

# type='photo' -> évite dessins / icônes
# size='large' -> meilleure qualité
filters = dict(
    type='photo',
    size='large'
)

# =========================================================
# ♻️ DÉCHETS RECYCLABLES
# =========================================================

keywords_recyclable = [

    # --- DÉJÀ UTILISÉS (commentés) ---
    # 'bouteilles plastique poubelle',
    # 'bouteilles plastique recyclage tas',
    # 'canettes aluminium bière soda déchet',
    # 'carton emballage déchet recyclage',
    # 'journaux papier vieux déchet',
    # 'boîtes de conserve métal déchet',
    # 'bouteilles en verre déchet recyclage',
    # 'bocaux verre usagés déchet',
    # 'emballages métalliques aérosol déchet',
    # 'bac jaune tri sélectif déchet plastique',
    # 'déchets plastique tas recyclage',
    # 'papier carton bac recyclage',
    # 'centre tri recyclage plastique',
    # 'poubelle recyclage maison',
    # 'déchets recyclables ménagers',
    # 'sac recyclage plastique',
    # 'tri sélectif déchets recyclables',
    # 'déchets recyclage extérieur',
    # 'conteneurs recyclage verre',
    # 'tas canettes aluminium',
    # 'plastic bottles recycling bin',
    # 'plastic waste sorting center',
    # 'aluminum cans trash recycling',
    # 'cardboard boxes recycling waste',
    # 'glass bottles recycling container',
    # 'metal packaging waste',
    # 'paper waste recycling pile',
    # 'household recyclable waste',
    # 'yellow recycling bin plastic',
    # 'recyclable garbage outdoor',
    # 'recycling center plastic bottles',
    # 'trash sorting recyclable materials',
    # 'used cans and bottles waste',
    # 'plastic containers recyclable waste',
    # 'paper cardboard waste sorting',
    # 'domestic recyclable waste',
    # 'mixed recyclable trash',
    # 'waste management recycling plastics',

    # --- NOUVEAUX ---

    # Français
    'bouteille eau minérale vide déchet',
    'barquette plastique alimentaire déchet',
    'flacon shampooing vide recyclage',
    'brique lait carton recyclage',
    'tube dentifrice vide déchet',
    'pot yaourt plastique déchet recyclage',
    'sachet plastique alimentaire déchet',
    'boîte métal sardines thon vide',
    'canette soda écrasée recyclage',
    'emballage plastique alimentaire tri',
    'bouteille huile vide recyclage',
    'boîte carton céréales recyclage',
    'bidon plastique détergent vide',
    'verre bocal confiture vide recyclage',
    'papier journal imprimé recyclage pile',
    'tuyau PVC plastique déchet recyclage',
    'sac poubelle jaune recyclage',
    'ferraille métal recyclage tas',

    # Anglais
    'empty milk carton recycling',
    'crushed plastic bottle waste',
    'food tin can empty recycling',
    'shampoo bottle empty plastic waste',
    'PET plastic bottle recycling',
    'tetra pak carton recycling',
    'yogurt cup plastic recyclable',
    'steel food can empty recycling',
    'glass jar empty recycling',
    'newspaper stack recycling',
    'cardboard cereal box recycling',
    'empty detergent bottle plastic',
    'scrap metal recycling pile',
    'plastic film packaging waste',
    'aluminum foil tray recycling',
    'empty aerosol can recycling',
    'plastic bag waste sorting',
    'mixed packaging recyclable waste',
]

# =========================================================
# 🌱 DÉCHETS COMPOSTABLES
# =========================================================

keywords_compost = [

    # --- DÉJÀ UTILISÉS (commentés) ---
    # 'épluchures carottes pommes de terre compost',
    # 'épluchures légumes cuisine biodéchet',
    # 'pelures fruits banane pomme déchet organique',
    # 'restes fruits pourris biodégradable',
    # 'marc de café filtre usagé compost',
    # 'feuilles de thé sachet usagé compost',
    # 'coquilles oeufs composteur',
    # 'feuilles mortes tas jardin automne',
    # 'herbe tondue sac déchet vert',
    # 'pain rassis moisissure biodéchet',
    # 'épluchures oignon ail déchet cuisine',
    # 'restes repas nourriture composteur',
    # 'déchets cuisine biodégradables seau compost',
    # 'compost maison biodéchets',
    # 'seau compost cuisine',
    # 'déchets organiques ménagers',
    # 'fruits légumes compostables',
    # 'déchets verts jardin',
    # 'bac compost biodéchets',
    # 'compost déchets alimentaires',
    # 'food scraps compost bin',
    # 'organic kitchen waste',
    # 'vegetable peels compost',
    # 'fruit waste composting',
    # 'banana peels organic waste',
    # 'garden waste compost pile',
    # 'coffee grounds compost',
    # 'eggshell compost organic',
    # 'biodegradable food waste',
    # 'household compost bucket',
    # 'green waste composting',
    # 'rotting vegetables compost',
    # 'organic trash kitchen',
    # 'yard waste leaves compost',
    # 'compost bin food scraps',
    # 'bio waste container',
    # 'kitchen organic garbage',
    # 'decomposing food waste',
    # 'organic waste recycling',
    # 'compostable household waste',

    # --- NOUVEAUX ---

    # Français
    'tomate pourrie moisie déchet',
    'carotte pourrie compost biodéchet',
    'pomme pourrie moisie déchet organique',
    'fraise moisie pourrie biodéchet',
    'orange moisie pourrie compost',
    'laitue salade pourrie déchet',
    'banane noire pourrie compost',
    'raisin moisi pourri déchet',
    'champignon pourri déchet organique',
    'trognon pomme déchet compost',
    'pelure melon concombre compost',
    'fleurs fanées compost jardin',
    'gazon tondu compost déchet vert',
    'sciure bois copeaux compost',
    'pain moisi vieux déchet compost',
    'reste salade tomate cuisine biodéchet',
    'brin herbes aromatiques déchet compost',
    'feuilles arbres mortes bac compost',

    # Anglais
    'rotten tomato compost organic',
    'rotten carrot vegetable compost',
    'moldy strawberry fruit waste',
    'overripe banana black organic waste',
    'rotten apple moldy compost',
    'wilted lettuce food waste compost',
    'moldy orange rotten fruit',
    'expired food compost bin',
    'grass clippings green compost',
    'apple core organic compost',
    'fallen leaves compost pile',
    'rotting grape food waste',
    'old bread moldy compost',
    'vegetable tops scraps compost',
    'wood chips sawdust compost',
    'wilted flowers garden compost',
    'decomposing fruit vegetable compost',
    'overripe melon organic waste',
]

# =========================================================
# ☣️ DÉCHETS CHIMIQUES
# =========================================================

keywords_chimiques = [

    # --- DÉJÀ UTILISÉS (commentés) ---
    # 'piles usagées batteries déchet chimique',
    # 'batteries voiture usagées déchet dangereux',
    # 'peinture bidon vide déchet chimique',
    # 'pot peinture déchet dangereux',
    # 'solvant bouteille déchet chimique dangereux',
    # 'produits nettoyage chimiques dangereux déchets',
    # 'huile moteur bidon usagé déchet',
    # 'pesticide herbicide bidon déchet chimique',
    # 'ampoule fluocompacte déchet chimique',
    # 'médicaments périmés boite pharmacie déchet',
    # 'produits chimiques ménagers dangereux décheterie',
    # 'aérosol bombe déchet chimique dangereux',
    # 'thermomètre mercure produit chimique déchet',
    # 'déchets toxiques ménagers',
    # 'produits corrosifs déchets',
    # 'liquides chimiques dangereux',
    # 'déchets industriels chimiques',
    # 'centre collecte déchets dangereux',
    # 'déchets batterie lithium',
    # 'bidons produits toxiques',
    # 'hazardous waste batteries',
    # 'chemical waste containers',
    # 'toxic household waste',
    # 'paint cans hazardous waste',
    # 'used motor oil waste',
    # 'cleaning chemicals disposal',
    # 'battery recycling hazardous',
    # 'medical waste medicine disposal',
    # 'expired drugs trash',
    # 'chemical products dangerous waste',
    # 'industrial chemical waste',
    # 'toxic waste landfill',
    # 'hazardous garbage household',
    # 'spray cans hazardous waste',
    # 'electronic waste batteries',
    # 'fluorescent bulbs hazardous',
    # 'dangerous waste collection center',
    # 'corrosive chemical waste',
    # 'toxic liquid waste container',
    # 'chemical waste disposal',

    # --- NOUVEAUX ---

    # Français
    'tube néon fluorescent déchet chimique',
    'pile bouton montre usagée déchet',
    'antigel liquide voiture déchet chimique',
    'vernis ongles dissolvant déchet chimique',
    'colle forte époxy déchet chimique dangereux',
    'engrais chimique sac vide déchet',
    'désinfectant biocide bouteille déchet',
    'radiographie film médical déchet chimique',
    'acide batterie déchet dangereux',
    'spray insecticide vide déchet chimique',
    'huile de frein liquide déchet dangereux',
    'détartrant acide produit chimique déchet',
    'white spirit bouteille déchet dangereux',
    'toner cartouche imprimante déchet chimique',
    'plombs pêche mercure déchet toxique',
    'bouteille acide chlorhydrique déchet dangereux',
    'déodorant bille vide déchet chimique',
    'fiole produit laboratoire déchet chimique',

    # Anglais
    'fluorescent tube neon light hazardous waste',
    'small button battery watch hazardous',
    'antifreeze car chemical waste',
    'nail polish remover acetone chemical waste',
    'epoxy glue chemical hazardous disposal',
    'chemical fertilizer empty bag waste',
    'biocide disinfectant bottle waste',
    'medical xray film chemical waste',
    'car battery acid hazardous',
    'insecticide spray empty chemical waste',
    'brake fluid chemical waste disposal',
    'white spirit solvent chemical waste',
    'printer toner cartridge chemical waste',
    'hydrochloric acid bottle hazardous',
    'laboratory chemical waste disposal',
    'toxic fumes chemical container waste',
    'pesticide empty bottle chemical hazardous',
    'oil filter chemical hazardous waste',

    # Français (suite)
    'bonbonne gaz vide déchet dangereux',
    'extincteur usagé déchet chimique',
    'radiateur voiture liquide refroidissement déchet',
    'vernis bois bidon déchet chimique',
    'décapant four nettoyant chimique déchet',
    'chlore piscine bidon déchet chimique',
    'acide sulfurique batterie déchet dangereux',
    'spray peinture bombe aérosol vide déchet chimique',
    'fluide réfrigérant climatiseur déchet chimique',
    'résine époxy durcisseur déchet chimique',
    'mercure thermomètre cassé déchet toxique',
    'plomb soudure déchet chimique dangereux',
    'chiffon imbibé solvant déchet dangereux',
    'tube silicone mastic déchet chimique',
    'colle néoprène bidon déchet chimique',
    'désherbant roundup bidon déchet chimique',
    'raticide poison déchet chimique dangereux',
    'lingette nettoyante chimique produit déchet',
    'bidon engrais liquide chimique déchet',
    'bocal produit chimique laboratoire déchet',

    # Anglais (suite)
    'gas cylinder empty hazardous waste',
    'fire extinguisher used chemical waste',
    'car coolant antifreeze chemical hazardous',
    'wood varnish can chemical waste',
    'oven cleaner chemical waste disposal',
    'pool chlorine chemical container waste',
    'sulfuric acid car battery hazardous',
    'spray paint aerosol can chemical waste',
    'refrigerant coolant chemical hazardous waste',
    'epoxy resin hardener chemical waste',
    'mercury broken thermometer toxic waste',
    'lead solder hazardous chemical waste',
    'solvent soaked rag chemical waste',
    'silicone sealant tube chemical waste',
    'contact cement glue chemical waste',
    'weed killer roundup chemical waste',
    'rat poison chemical hazardous waste',
    'liquid fertilizer bottle chemical waste',
    'chemical lab bottle hazardous disposal',
    'asbestos tile hazardous construction waste',
]

# =========================================================
# 🗑️ DÉCHETS MÉNAGERS
# =========================================================

keywords_menagere = [

    # --- DÉJÀ UTILISÉS (commentés) ---
    # 'mouchoirs sales poubelle',
    # 'couches usagées déchet',
    # 'verre cassé poubelle',
    # 'poussière aspirateur déchets',
    # 'boite pizza grasse poubelle',
    # 'litière chat déchet',
    # 'sac poubelle ménagère',
    # 'déchets ménagers non recyclables',
    # 'poubelle noire déchets',
    # 'déchets mélangés maison',
    # 'ordures ménagères cuisine',
    # 'déchets sales ménagers',
    # 'déchets résiduels poubelle',
    # 'emballages sales poubelle',
    # 'essuie tout sale déchets',
    # 'dirty tissues trash',
    # 'used diapers garbage',
    # 'broken glass trash',
    # 'vacuum dust waste',
    # 'dirty pizza box trash',
    # 'cat litter waste',
    # 'household garbage bag',
    # 'non recyclable household waste',
    # 'mixed trash bin',
    # 'general waste garbage',
    # 'residual waste household',
    # 'dirty packaging trash',

    # --- NOUVEAUX ---

    # Français
    'coton tige usagé poubelle déchet',
    'chewing gum mâché déchet poubelle',
    'ticket caisse reçu déchet froissé',
    'stylo bille cassé vide déchet',
    'lingette sale non recyclable déchet',
    'serviette papier sale déchet poubelle',
    'emballage chips paquet vide grasse poubelle',
    'pare brise cassé déchet ménager',
    'couche bébé souillée poubelle',
    'sac croustilles paquet vide déchet',
    'brosse à dents usée déchet',
    'cigarette mégot déchet poubelle',
    'sac aspirateur plein déchet poubelle',
    'jouet cassé plastique déchet ménager',
    'reste cire bougie déchet',
    'serpillière usée chiffon sale déchet',
    'vêtement abimé chiffon poubelle',
    'bouchon liège déchet ménager',

    # Anglais
    'used cotton swab qtip trash',
    'chewed gum waste trash',
    'crumpled receipt paper trash',
    'broken pen trash non recyclable',
    'dirty wet wipe waste',
    'soiled paper napkin trash',
    'greasy chip bag trash waste',
    'used diaper soiled garbage',
    'old toothbrush trash waste',
    'cigarette butt waste trash',
    'full vacuum cleaner bag waste',
    'broken plastic toy garbage',
    'candle wax leftover trash',
    'dirty rag mop waste',
    'torn worn out clothing trash',
    'wine cork waste trash',
    'greasy food wrapper trash',
    'non recyclable plastic waste bin',
]

# =========================================================
# Fonction scraping
# =========================================================

def scrape_images(keywords, folder, max_per_keyword=50):

    for keyword in keywords:

        print(f"\n🔎 Recherche : {keyword}")

        crawler = BingImageCrawler(
            storage={'root_dir': folder},
            downloader_threads=4,
            parser_threads=2,
        )

        try:
            crawler.crawl(
                keyword=keyword,
                filters=filters,
                max_num=max_per_keyword,
                file_idx_offset='auto'
            )

        except Exception as e:
            print(f"Erreur pour '{keyword}' : {e}")

# =========================================================
# Lancement
# =========================================================

print("\n==============================")
print("♻️ Téléchargement RECYCLABLE")
print("==============================")

scrape_images(keywords_recyclable, 'recyclable')

print("\n==============================")
print("🌱 Téléchargement COMPOST")
print("==============================")

scrape_images(keywords_compost, 'compost')

print("\n==============================")
print("☣️ Téléchargement CHIMIQUE")
print("==============================")

scrape_images(keywords_chimiques, 'dechets_chimiques')

print("\n==============================")
print("🗑️ Téléchargement MENAGERE")
print("==============================")

scrape_images(keywords_menagere, 'menagere')

print("\n✅ Dataset terminé.")
print("📁 Dossiers créés :")
print("- recyclable")
print("- compost")
print("- dechets_chimiques")
print("- menagere")