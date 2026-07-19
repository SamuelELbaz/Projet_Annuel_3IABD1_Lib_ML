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
    pointing:      5000,  // temps où il désigne sa vision
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

   Au repos le ciel est au grand jour. Pendant le sort il
   clignote (keyframes CSS). Pour en sortir sans à-coup on
   fige l'opacité atteinte, puis on la ramène à zéro.
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

   >>> C'EST ICI QUE TU BRANCHERAS TON MODÈLE. <<<
   Remplace le corps de predire() par ton appel réseau.
   Contrat attendu : { etiquette: string, confiance: number }
   avec confiance entre 0 et 1.
   --------------------------------------------------------- */

const ETIQUETTES_FACTICES = [
    "un chat", "un chien", "une théière", "un dirigeable", "un bol de soupe",
    "une fougère", "un accordéon", "une truite", "un lampadaire", "un scarabée",
];

async function predire(fichier) {
    await attendre(1500 + Math.random() * 2200); // latence simulée

    return {
        etiquette: ETIQUETTES_FACTICES[Math.floor(Math.random() * ETIQUETTES_FACTICES.length)],
        confiance: 0.4 + Math.random() * 0.2, // 40–60 %, comme le vrai modèle
    };
}

/** Le sorcier n'annonce pas un score : il l'habite. */
function formulerVision({ etiquette, confiance }) {
    const pourcentage = Math.round(confiance * 100);

    const preambules = [
        `Je vois… ${etiquette}.`,
        `L'orbe s'éclaircit. ${etiquette.charAt(0).toUpperCase() + etiquette.slice(1)}.`,
        `Voilà. ${etiquette.charAt(0).toUpperCase() + etiquette.slice(1)}, sans le moindre doute. Enfin, presque.`,
    ];

    const aveux = confiance < 0.5
        ? `Je n'en suis sûr qu'à ${pourcentage} %. Ne mise rien là-dessus.`
        : `${pourcentage} % de certitude. C'est un bon jour.`;

    return `${preambules[Math.floor(Math.random() * preambules.length)]} ${aveux}`;
}

/* ---------------------------------------------------------
   7. La séquence complète
   --------------------------------------------------------- */

async function lancerSort(fichier) {
    if (sortEnCours) return;
    sortEnCours = true;

    // 1. standing -> sort, le ciel se met à vaciller
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

    // 3. réponse reçue -> pointing
    sorcier.classList.remove("sorcier--incante");
    apaiserCiel();
    afficherSprite(ETATS.POINTING);
    ajouterReplique(formulerVision(vision));

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
