# D1 e D2 sui profili FULL: ambito e verifica

Notebook: `models_D1_D2_full_profiles_standalone_colab.ipynb`.

## Uso

Caricare il notebook in Colab nell'account che contiene la raccolta FULL ed eseguire tutto su CPU. Il percorso iniziale è `/content/drive/MyDrive/patchAliasing/full/d3_15model_v1/data`. Il notebook installa la linea PyMC 5 e ArviZ 0.22 e riutilizza `02_collapse.parquet`, `02_sites.parquet` e `collection_manifest.json`.

Non sono richiesti repository, Chronos o nuove raccolte. Il preflight richiede la copertura FULL attesa, hash coerenti, normalizzazione originale e chiavi univoche. I siti salvati sono un riferimento storico; non determinano le osservazioni del nuovo D2. Gli output vengono scritti in `d_full_profiles/<run-id>`, separati dai dati. Configurazione, codice generato, versioni dei pacchetti e hash degli input determinano il run-id. Per cambiare la specifica bisogna rigenerare il notebook oppure usare un nuovo `MODEL_VERSION`; non riutilizzare manualmente checkpoint di una specifica diversa.

## D1 conservato

La factory `model_D1_sites` è identica via AST a quelle di `new_analysis_v2.ipynb` e `localisation_analysis.ipynb` forniti dall'utente. SHA-256 AST: `f450b67432016b9ca61d75ba385758e0598ef4f73e57f883bcd9166211850741`.

Sono conservati risposta logaritmica, floor 0.01, indicatori entro 1 Hz dalle griglie, intercetti per geometria, prior Normal(0,1) sui coefficienti e HalfNormal(1) sul rumore. Il notebook riproduce anche la funzione originale di distanza dalla griglia, senza importare il repository.

Le quattro versioni, entrambe le griglie, stride, patch e nessuna, sono stimate separatamente per pure, KernelSynth e TSMixup: dodici fit, ciascuno con quattro catene, 1.000 warmup e 1.000 draw per catena. La log-likelihood è utilizzata per PSIS-LOO e poi rimossa dall'oggetto da salvare; restano posteriori, diagnostiche e valori LOO puntuali. Il confronto usa differenze ELPD appaiate e relativi errori standard. Non vengono riciclati fit degli altri notebook.

Si verificano R-hat, ESS bulk/tail, divergenze, BFMI, Pareto-k e controlli predittivi di media e deviazione standard per geometria/replica. I grafici includono tracce e profili osservati con intervalli della media attesa. Le sinusoidi pure sono un riferimento di degenerazione esatta e non entrano nell'evidenza empirica sui generatori.

## D2 nuovo

La variante usa tutte le frequenze dei profili FULL di ciascun generatore. Due pettini gaussiani hanno spaziature `kappa_S * fs/S` e `kappa_P * fs/P`. I siti condivisi contribuiscono a entrambi i termini. Non vengono estratti minimi, usati `f1` o `delta_hat`, né applicate soglie 0.75/0.90 o il precedente gate di dieci siti.

I prior sono nuovi ed espliciti: kappa indipendenti Uniform(0.5,1.5), varianza InverseGamma(shape=3, scale=0.5), coefficienti e intercetti condizionatamente Normal(0,4*varianza). Il segno degli effetti non è imposto. Queste scelte non preservano i prior della vecchia regressione D2 e non devono essere descritte come un semplice cambio di campionatore.

Intercetti, coefficienti e varianza vengono integrati analiticamente con un complemento di Schur. La posteriore congiunta delle spaziature viene approssimata mediante quadratura, conservando possibili modi alternativi. Si usano 801 punti per asse, 1.601 quando necessario e fino a cinque raffinamenti locali con pesi trapezoidali non uniformi. Ogni raffinamento mantiene l'intero intervallo iniziale. Il controllo comprende sottogriglie alternate, differenze di massa ROPE, media e log integrale normalizzante, concentrazione su singoli nodi e massa ai bordi. Un risultato irrisolto resta segnalato; il codice non abbassa i criteri.

La larghezza primaria è 1 Hz; 0.5 e 2 Hz sono sensibilità obbligatorie. Il riepilogo considera anche la probabilità di coefficiente negativo e il PPC stratificato. La probabilità nella ROPE [0.9,1.1] è confrontabile con la massa a priori 0.20, ma non è un Bayes factor contro una massa puntuale a kappa=1.

## Limiti metodologici

- Il range di kappa delimita le soluzioni considerate. Poca massa ai bordi non esclude da sola modi esterni a questo range.
- La forma del pettine assume un effetto atteso comune alle armoniche del ramo. Assenze sistematiche, variazioni di profondità o forme non gaussiane possono renderla inadeguata.
- I due modelli mantengono una likelihood condizionalmente indipendente sulle frequenze. PPC stratificato e LOO puntuale non costituiscono validazione su repliche o geometrie indipendenti.
- I FULL sono gli stessi già esplorati nella diagnosi delle soglie. I risultati della variante rimangono esplorativi, anche quando i controlli interni sono favorevoli.
- Una posteriore stretta non garantisce una specifica corretta. Occorre leggere profili ricostruiti, superficie congiunta e sensibilità, oltre alla tabella conclusiva.

## Verifiche locali

Gli strumenti sono in `output/d2_validation/build_d_full.py`, `d_profile_core.py`, `d_profile_d1.py` e `test_d_full.py`; il notebook consegnato incorpora tutto il codice necessario e non dipende da questi file.

- Schema notebook, sintassi e identità AST di D1 verificati.
- La likelihood marginale accelerata è confrontata con una soluzione matriciale diretta su nove coppie di kappa; il notebook ripete questa verifica su poche righe del dataset selezionato prima dei fit.
- Test software con kappa reali 0.833 e 1.173 e fondamentale patch rimossa: medie posteriori circa 0.83309 e 1.17291, intervalli al 95% contenenti i valori generativi, massa nella ROPE praticamente nulla e controllo di quadratura superato. Questo è un controllo circoscritto del calcolo, non una calibrazione completa del metodo.
- Un profilo esattamente piatto restituisce probabilità 0.5 di coefficiente negativo, senza supporto automatico a una depressione.
- Esecuzione completa su fixture tecnica piccola con due generatori, catene brevi e quadratura ridotta: otto fit D1, sei superfici D2, posteriori, LOO, PPC, esportazioni e riepilogo. Le diagnostiche delle catene brevi non sono evidenza empirica; il riepilogo conserva `full_verified=False` e non emette supporto FULL.
- Versione finale rieseguita integralmente dopo le modifiche di presentazione. Posteriori congiunte, marginali e profili con intervalli ispezionati visivamente.
- Una seconda esecuzione in kernel pulito ha riutilizzato tutti gli otto checkpoint D1 senza campionare nuovamente; i file posteriori sono rimasti invariati. Verifica in `output/d2_validation/test_d_full_resume.py`.

I Parquet FULL dell'altro account Google non sono disponibili nel checkout. Non è stata quindi eseguita localmente l'inferenza empirica FULL. Il notebook consegnato è configurato per farla direttamente in Colab, con output vuoti per evitare di presentare la fixture tecnica come risultato.
