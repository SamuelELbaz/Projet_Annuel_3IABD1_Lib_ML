/* =========================================================
   ApeupreH — logique du front
   ========================================================= */

/* ---------------------------------------------------------
   1. Réglages centralisés
   --------------------------------------------------------- */

const DUREES = {
    fonduSprite:   600,   // doit rester aligné sur --duree-fondu-sprite (CSS)
    lancement:     1400,  // temps où il incante avant qu'on interroge le modèle
    retourCiel:    900,   // le ciel se recale sur le plein jour
    pointing:      1600,  // pose maintenue après le verdict, avant retour au repos
    frappe:        28,    // ms par caractère du typewriter
    pauseReplique: 420,   // silence entre deux répliques
};

const ETATS = {
    REPOS:    "repos",
    SORT:     "sort",
    POINTING: "pointing",
};

/* ---------------------------------------------------------
   2. Éléments
   --------------------------------------------------------- */

const scene         = document.getElementById("scene");
const fondNuit      = document.getElementById("fondNuit");
const sorcier       = document.getElementById("sorcier");
const invite        = document.getElementById("invite");
const champFichier  = document.getElementById("champFichier");
const offrande      = document.getElementById("offrande");
const offrandeImage = document.getElementById("offrandeImage");
const retirerBouton = document.getElementById("retirerOffrande");
const flux          = document.getElementById("flux");
const champTexte    = document.getElementById("champTexte");
const boutonEnvoyer = document.getElementById("boutonEnvoyer");

const sprites = document.querySelectorAll(".sprite");

let sortEnCours = false;
let urlOffrande = null;

/* ---------------------------------------------------------
   3. Machine à états des sprites
   --------------------------------------------------------- */

function afficherSprite(etat) {
    sprites.forEach((sprite) => {
        sprite.classList.toggle("sprite--visible", sprite.dataset.etat === etat);
    });
}

/** Petite aide : une pause, pour écrire la séquence de façon linéaire. */
const attendre = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/* ---------------------------------------------------------
   4. Le ciel
   --------------------------------------------------------- */

function declencherClignotement() {
    fondNuit.style.animation = "";
    fondNuit.style.transition = "";
    fondNuit.style.opacity = "";
    scene.classList.add("scene--sort");
}

function apaiserCiel() {
    const opaciteAtteinte = getComputedStyle(fondNuit).opacity;

    scene.classList.remove("scene--sort");
    fondNuit.style.animation = "none";
    fondNuit.style.opacity = opaciteAtteinte;

    void fondNuit.offsetWidth; // force le reflow avant la transition

    fondNuit.style.transition = `opacity ${DUREES.retourCiel}ms ease-out`;
    fondNuit.style.opacity = "0";
}

/* ---------------------------------------------------------
   5. Le dialogue

   Les répliques sont mises en file : deux appels rapprochés
   ne se marchent pas dessus.
   --------------------------------------------------------- */

let file = Promise.resolve();

function ajouterReplique(texte, auteur = "oracle") {
    file = file.then(async () => {
        const ligne = document.createElement("p");
        ligne.className = `replique replique--${auteur}`;
        flux.appendChild(ligne);

        if (auteur === "toi") {
            ligne.textContent = texte;
            flux.scrollTop = flux.scrollHeight;
            return;
        }

        ligne.classList.add("curseur");
        for (const caractere of texte) {
            ligne.textContent += caractere;
            flux.scrollTop = flux.scrollHeight;
            await attendre(DUREES.frappe);
        }
        ligne.classList.remove("curseur");
        await attendre(DUREES.pauseReplique);
    });
    return file;
}

/* ---------------------------------------------------------
   6. La prédiction

   Branchée sur l'API Flask : POST /predict, champ multipart
   "image" (même origine, Flask sert le front et l'API). Le
   back répond avec les 4 modèles d'un coup :
     { resultats: { PMC:      {classe, scores},
                    Lineaire: {classe, scores},
                    SVM:      {classe, scores},
                    RBF:      {classe, scores} } }
   où scores == { compost, dechets_chimiques, recyclable }.

   L'oracle ne montre qu'une voix : on fond les 4 en une seule
   distribution (moyenne des softmax) et on en tire l'étiquette
   gagnante + sa confiance dans [0, 1].

   Contrat rendu au reste du code : { etiquette, confiance }.
   --------------------------------------------------------- */

// De la classe brute du modèle vers quelque chose que l'oracle peut prononcer.
const ETIQUETTES_LISIBLES = {
    compost:           "du compost",
    dechets_chimiques: "des déchets chimiques",
    recyclable:        "des déchets recyclables",
};

function softmax(valeurs) {
    const max   = Math.max(...valeurs);
    const exps  = valeurs.map((v) => Math.exp(v - max));
    const somme = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / somme);
}

/*
RAPPEL LES LOULOUS

Les 4 modèles ne notent pas sur la même échelle (le SVM peut
   sortir des scores négatifs, la RBF des sommes de gaussiennes…).
   On ramène chacun à une distribution avec un softmax, puis on
   moyenne : mélange d'experts, chaque modèle vote avec une
   distribution complète plutôt qu'un simple gagnant.

*/
function agregerModeles(resultats) {
    const classes = [];
    for (const r of Object.values(resultats)) {
        if (!r) continue; // un modèle peut être "non disponible" (null)
        for (const c of Object.keys(r.scores)) {
            if (!classes.includes(c)) classes.push(c);
        }
    }

    const cumul = new Array(classes.length).fill(0);
    let nbModeles = 0;
    for (const r of Object.values(resultats)) {
        if (!r) continue;
        const scores = classes.map((c) => r.scores[c] ?? 0);
        softmax(scores).forEach((p, i) => (cumul[i] += p));
        nbModeles++;
    }
    if (nbModeles === 0) throw new Error("Aucun modèle n'a répondu.");

    const moyenne = cumul.map((v) => v / nbModeles);
    let idx = 0;
    for (let i = 1; i < moyenne.length; i++) {
        if (moyenne[i] > moyenne[idx]) idx = i;
    }

    return { etiquette: classes[idx], confiance: moyenne[idx] };
}

// mapping nom intenre du modèle -> nom lisible en txt
const NOMS_MODELES = {
    PMC:      "le PMC",
    Lineaire: "le linéaire",
    SVM:      "le SVM",
    RBF:      "la RBF",
};

const capitaliser = (s) => s.charAt(0).toUpperCase() + s.slice(1);

// Classe prédite + confiance | un model
function probaModele(r) {
    const classes = Object.keys(r.scores);
    const probs   = softmax(classes.map((c) => r.scores[c]));
    let idx = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[idx]) idx = i;
    return { etiquette: classes[idx], confiance: probs[idx] };
}

async function predire(fichier) {
    const donnees = new FormData();
    donnees.append("image", fichier);

    const reponse = await fetch("/predict", { method: "POST", body: donnees });
    if (!reponse.ok) throw new Error(`Le serveur a répondu ${reponse.status}.`);

    const data = await reponse.json();
    if (data.erreur) throw new Error(data.erreur);

    const resultats = data.resultats;

    // Une ligne par modèle classe predict + confiance %
    const parModele = [];
    for (const [cle, r] of Object.entries(resultats)) {
        if (!r) continue; // modèle non disponible (null)
        const { etiquette, confiance } = probaModele(r);
        parModele.push({
            modele:    NOMS_MODELES[cle] ?? cle,
            etiquette: ETIQUETTES_LISIBLES[etiquette] ?? etiquette,
            confiance,
        });
    }
    if (parModele.length === 0) throw new Error("Aucun modèle n'a répondu.");

    // La synthèse des 4, pour le mot de la fin.
    const v = agregerModeles(resultats);
    const verdict = {
        etiquette: ETIQUETTES_LISIBLES[v.etiquette] ?? v.etiquette,
        confiance: v.confiance,
    };

    return { parModele, verdict };
}

// Le mot de la fin -> avis du sorcier magique
function formulerVerdict({ etiquette, confiance }) {
    const pct = Math.round(confiance * 100);
    return confiance < 0.4
        ? `Moi, je penche pour ${etiquette}, mais à ${pct} % seulement. Ne mise rien là-dessus.`
        : `Tout bien pesé, je dis ${etiquette}. ${pct} % de certitude — c'est un bon jour.`;
}

/* ---------------------------------------------------------
   7. La séquence complète
   --------------------------------------------------------- */

async function lancerSort(fichier) {
    if (sortEnCours) return;
    sortEnCours = true;

    // 1. standing -> sort -> zerqzfqezf du ciel
    afficherSprite(ETATS.SORT);
    sorcier.classList.add("sorcier--incante");
    declencherClignotement();
    ajouterReplique("Ne bouge plus. Je regarde.");

    await attendre(DUREES.lancement);

    // 2. état d'attente : il incante tant que le modèle n'a pas répondu
    let vision;
    try {
        vision = await predire(fichier);
    } catch (erreur) {
        console.error(erreur);
        sorcier.classList.remove("sorcier--incante");
        apaiserCiel();
        afficherSprite(ETATS.REPOS);
        ajouterReplique("L'orbe est resté noir. L'oracle n'a pas répondu — retente ta chance.");
        sortEnCours = false;
        return;
    }

    // 3. réponse reçue -> pointing -> 4 visions + verdict
    sorcier.classList.remove("sorcier--incante");
    apaiserCiel();
    afficherSprite(ETATS.POINTING);

    ajouterReplique("Quatre regards se posent sur ton offrande. Voici ce qu'ils disent.");
    for (const v of vision.parModele) {
        const pct = Math.round(v.confiance * 100);
        ajouterReplique(`${capitaliser(v.modele)} y voit ${v.etiquette} — ${pct} %.`);
    }
    // ajouterReplique met en file : on attend la fin de TOUTE la tirade avant de bouger.
    await ajouterReplique(formulerVerdict(vision.verdict));

    await attendre(DUREES.pointing);

    // 4. retour au repos
    afficherSprite(ETATS.REPOS);
    await attendre(DUREES.fonduSprite);

    sortEnCours = false;
}

/* ---------------------------------------------------------
   8. L'offrande
   --------------------------------------------------------- */

function poserOffrande(fichier) {
    if (!fichier || !fichier.type.startsWith("image/")) {
        ajouterReplique("Ceci n'est pas une image. Mes yeux ne lisent que les images.");
        return;
    }

    if (urlOffrande) URL.revokeObjectURL(urlOffrande);
    urlOffrande = URL.createObjectURL(fichier);

    offrandeImage.src = urlOffrande;
    offrande.classList.remove("offrande--vide");

    lancerSort(fichier);
}

function reprendreOffrande() {
    if (urlOffrande) {
        URL.revokeObjectURL(urlOffrande);
        urlOffrande = null;
    }
    offrandeImage.src = "";
    offrande.classList.add("offrande--vide");
    champFichier.value = "";
}

/* ---------------------------------------------------------
   9. Écoute des événements
   --------------------------------------------------------- */

// Empêche le navigateur d'ouvrir l'image dans l'onglet
["dragenter", "dragover", "dragleave", "drop"].forEach((type) => {
    document.addEventListener(type, (e) => e.preventDefault());
});

sorcier.addEventListener("dragenter", () => sorcier.classList.add("sorcier--survol"));
sorcier.addEventListener("dragover",  () => sorcier.classList.add("sorcier--survol"));
sorcier.addEventListener("dragleave", () => sorcier.classList.remove("sorcier--survol"));

sorcier.addEventListener("drop", (e) => {
    sorcier.classList.remove("sorcier--survol");
    poserOffrande(e.dataTransfer.files[0]);
});

// Repli : clic ou clavier ouvrent le sélecteur de fichier
sorcier.addEventListener("click", () => champFichier.click());
sorcier.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        champFichier.click();
    }
});

champFichier.addEventListener("change", (e) => poserOffrande(e.target.files[0]));
retirerBouton.addEventListener("click", reprendreOffrande);

/* ---------------------------------------------------------
   10. La conversation libre
   --------------------------------------------------------- */

const REPONSES_LIBRES = [
    "Les mots, c'est ton domaine. Moi je ne lis que les images.",
    "Hmpf. Dépose une image, on verra bien.",
    "Trois oracles sur cinq mentent. Moi, je me trompe honnêtement.",
    "Ma boule n'est pas une bibliothèque. Donne-lui à voir.",
    "Pose ton image entre mes mains. Le reste n'est que bavardage.",
];

function envoyerMessage() {
    const texte = champTexte.value.trim();
    if (!texte) return;

    ajouterReplique(texte, "toi");
    champTexte.value = "";
    ajouterReplique(REPONSES_LIBRES[Math.floor(Math.random() * REPONSES_LIBRES.length)]);
}

boutonEnvoyer.addEventListener("click", envoyerMessage);
champTexte.addEventListener("keydown", (e) => {
    if (e.key === "Enter") envoyerMessage();
});

/* ---------------------------------------------------------
   11. Le lore d'ouverture
   --------------------------------------------------------- */

const LORE = [
    "Ah. Un visiteur.",
    "Je suis ApeupreH, dernier oracle de la Clairière Basse. Mon don : nommer ce que mes yeux ne voient qu'à moitié.",
    "Je me trompe deux fois sur cinq. Parfois trois. Les autres appellent ça un défaut. J'appelle ça de l'honnêteté.",
    "Dépose une image entre mes mains, et je te dirai ce que j'y devine. À peu près.",
];

LORE.forEach((ligne) => ajouterReplique(ligne));