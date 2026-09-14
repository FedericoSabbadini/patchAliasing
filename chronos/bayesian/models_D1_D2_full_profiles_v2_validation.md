# Analisi D FULL v2: correzione dei PPC e della variante D2

Notebook: `models_D1_D2_full_profiles_v2_standalone_colab.ipynb`.

## Diagnosi della run FULL restituita

Letto `Downloads/models_D1_D2_full_profiles_standalone_colab.ipynb`: tutte le nove celle di codice eseguite, nessun errore, codice identico alla v1 preparata. Run: `27c183d24ddbadc3`.

D1 non aveva perso il risultato iniziale. Tutti i fit avevano diagnostiche e LOO adeguati. Stride era primo per entrambi i generatori; aggiungere patch cambiava ELPD di circa -0.75 ±0.95 per KernelSynth e -1.09 ±0.76 per TSMixup. La factory era rimasta identica, ma era cambiato il criterio PPC.

Il notebook iniziale verificava la **media per geometria**, sulle repliche aggregate: negli output iniziali D1 aveva 30/30 strati dei generatori entro gli intervalli predittivi. La v1 standalone richiedeva invece che passassero sia media sia deviazione standard per **geometria e replica**. Nei FULL restituiti:

| D1, entrambe le griglie | Media per replica | SD per replica |
|---|---:|---:|
| KernelSynth | 45/45 | 14/45 |
| TSMixup | 33/45 | 19/45 |
| Pure | 45/45 | 15/45 |

Il confronto diretto con il vecchio `PPC_OK` era quindi scorretto. Le nuove bocciature della dispersione restano informative: una sola varianza residua descrive male profili con variabilità diversa. Per TSMixup compaiono anche differenze di livello tra repliche.

D2 v1 a larghezza 1 Hz aveva medie 45/45 e SD 14/45 per KernelSynth, medie 31/45 e SD 19/45 per TSMixup. La quadratura aveva inoltre raggiunto il limite di dimensione prima di risolvere la marginale stride: l'asse condiviso usava le zone di massa di entrambe le marginali per raffinare tutte e due le dimensioni. Il ramo patch ampio/multimodale faceva crescere la griglia anche dove stride non ne aveva bisogno.

## Correzioni D1

- Factory e prior originali conservati, verificati via AST. Nessun nuovo parametro introdotto in D1.
- Ripristinato il PPC della media per geometria con copertura minima 90%. Tutti i draw del fit contribuiscono: gli intervalli sono quantili della miscela predittiva Normale, ottenuti integrando analiticamente il rumore invece di simularne soltanto 300 realizzazioni.
- I controlli aggiuntivi per replica restano nelle tabelle e nel riepilogo, sotto `ppc_extended_ok`, separati da `ppc_original_ok`. Non sono cancellati o dichiarati superati.
- Riutilizzo dei dodici fit D1 della run FULL precedente, dopo verifica di hash input, specifica e configurazione D1, factory, floor, tolleranza, seed e completezza degli artefatti. Le copie dei posteriori e LOO vengono salvate nella nuova cartella; la sorgente non viene modificata. Se i checkpoint non sono disponibili o compatibili, il notebook lo dichiara e stima i fit mancanti.

Il PPC originale resta un controllo in-sample della media, facilitato dagli intercetti per geometria. Passarlo non dimostra che il modello descriva bene ogni replica o l'intera distribuzione delle frequenze; il riepilogo conserva questa distinzione.

## Correzioni D2

La v2 è una modifica esplicita del modello, motivata dagli errori predittivi della v1. Per ogni profilo `c=(geometria, replica)`:

`log(z_c(f)) ~ Normal(alpha_c + theta_S,c C_S(f) + theta_P,c C_P(f), sigma_c²)`.

Le spaziature restano `kappa_S fs/S` e `kappa_P fs/P`, con kappa condivisi tra profili della stessa modalità. Livello, coefficienti e rumore sono distinti per profilo. Per ciascun profilo si usano `sigma_c² ~ InvGamma(3,0.5)` e `(alpha_c,theta_S,c,theta_P,c) | sigma_c² ~ Normal(0,4 sigma_c² I)`, indipendenti tra profili. Non è un modello gerarchico con iperparametri di pooling appresi. Si integrano tutti questi parametri; le varianze non sono stimate a parte e poi trattate come note.

I due kappa hanno prior Uniform(0.25,2.0), ampliato rispetto alla v1 perché una sensibilità TSMixup accumulava massa vicino al vecchio bordo superiore. Il range resta un'assunzione e possibili modi esterni non vengono esclusi dalla sola poca massa ai bordi. La massa a priori nella ROPE [0.9,1.1] è circa 0.1143.

La quadratura usa assi separati. I controlli di sottogriglia sono effettuati un asse alla volta e si raffina soltanto quello non risolto, conservando il resto del range. I criteri numerici non sono abbassati. Si applicano un limite di dodici milioni di nodi congiunti e otto raffinamenti; un mancato superamento resta segnalato.

D2 mantiene il gate esteso su media e deviazione standard per replica, oltre a sensibilità alla larghezza, massa ai bordi, precisione numerica e probabilità di effetto medio negativo. L'effetto medio è la media dei coefficienti a pesi uguali tra profili. La sua probabilità viene stimata da 1.000 draw della posteriore marginalizzata; si riporta l'errore Monte Carlo e si salvano intervalli dei coefficienti e del rumore per profilo.

Il miglior adattamento di livelli e varianze non prova H3. Resta necessario leggere spaziature, modi alternativi e sensibilità. D2 può legittimamente non sostenere kappa_P=1 anche dopo la correzione. Le inferenze sono esplorative sugli stessi FULL e condizionate alla likelihood di indipendenza tra frequenze.

## Verifica software

- Notebook validato e codice D1 confrontato con la factory originale.
- Formula marginale per profilo confrontata con una soluzione matriciale diretta.
- Caso numerico con posteriore stride stretta e patch uniforme: viene raffinato solo l'asse stride.
- Profili sintetici con livelli e rumori diversi tra repliche, fondamentale patch rimossa e kappa generativi 0.833/1.173: recuperati circa 0.83347/1.17286, con intervalli al 95% contenenti i valori generativi, ROPE non sostenuta e quadratura verificata. Nel caso controllato passano sia i PPC di media sia quelli di SD. È una verifica software circoscritta, non un risultato FULL o una calibrazione completa.
- Caso con media corretta e SD sbagliata: il PPC originale passa e quello esteso sulla SD fallisce, come deve accadere.
- Esecuzione integrale su fixture tecnica, con catene e griglia ridotte, riutilizzando otto checkpoint D1 della precedente prova senza ricampionare né modificare i file sorgente. Tabelle, posteriori, PPC, parametri per profilo e riepilogo esportati. Grafici di posteriori e profili ispezionati visivamente.

Gli strumenti sono `output/d2_validation/build_d_full_v2.py`, `d_profile_core_v2.py`, `d_profile_d1_v2.py` e `test_d_full_v2.py`. Il notebook incorpora tutto il codice; non dipende da questi file.

## Esecuzione empirica da completare in Colab

Il notebook consegnato ha output vuoti, `REQUIRE_FULL=True` e impostazioni FULL. Il percorso predefinito dei Parquet e la sorgente dei checkpoint D1 puntano alla stessa raccolta usata dall'utente. La run v1 è preservata. La v2 salva in una nuova cartella e ricalcola D2 perché la sua specifica è cambiata.

I FULL e i posteriori D1 sono nell'altro account Google e non sono disponibili nel checkout locale. La v2 deve quindi essere eseguita in quel Colab. Non si afferma che i suoi PPC empirici siano già passati.
