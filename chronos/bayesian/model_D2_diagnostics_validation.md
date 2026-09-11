# D2 standalone: verifica del notebook diagnostico

Notebook: `model_D2_diagnostics_standalone_colab.ipynb`.

## Ambito

Legge `02_collapse.parquet` e `02_sites.parquet` dalla stessa cartella, con il manifest della raccolta se disponibile. Il percorso Colab predefinito è `/content/drive/MyDrive/patchAliasing/full/d3_15model_v1/data`. Per un'altra run impostare `SOURCE_DATA_DIR` esplicitamente.

Il modello D2, i prior, la likelihood, le risposte `f1` e `delta_hat`, il floor di 1 Hz del FULL, i filtri e il gate di dieci siti sono conservati. Il notebook non contiene campionamenti, nuove raccolte o scritture sui Parquet originali. `BUILD_MODEL_GRAPH=False` evita anche la dipendenza da PyMC durante la diagnosi; l'opzione True costruisce soltanto il grafo.

Le funzioni di rilevazione e assegnazione sono incorporate. Eventuali differenze rispetto ai siti salvati vengono segnalate, senza correggere i dati. Gli hash dei due input sono controllati prima e dopo l'esecuzione. Ogni esecuzione esporta tabelle, grafici, `SUMMARY.md` e `provenance.json` in una nuova sottocartella di `d2_diagnostics`, esterna a `data`.

## Verifiche svolte l'11 settembre 2026

- Schema nbformat 4.5 e sintassi validati: 40 celle totali, di cui 20 di codice, dopo le estensioni di sensibilità e audit a 0.90.
- La factory `model_D2_movement` è identica via AST a quella di entrambi gli allegati, compresa la docstring. SHA-256 AST: `24db4c1d2a9a6a3de92aa40aced19b35f078a7c1664ca0f7795a0d7be99c0b3c`.
- Esecuzione completa in kernel Jupyter separato su una fixture sintetica: 72/72 righe dei siti ricostruite. Verificati un sito patch a 240 Hz, armonica 15, e supporto in 2/3 repliche senza gonfiare il gate. Sono test software, non osservazioni empiriche.
- Esecuzione completa sui Parquet locali `d3_15model_smoke_v2`: 120/120 righe dei siti ricostruite. Il notebook segnala correttamente smoke, griglia da 8 Hz, generatore mancante e singola replica. Questi dati non sostituiscono il FULL.
- Controllati i casi di siti condivisi esclusi, sinusoidi pure e repliche escluse dal gate, singolo sito con `delta_hat=NaN`, estremi esclusi dal rilevatore e fusione di rilevazioni vicine. Il caso P=24, S=16 restituisce distanza 64 Hz dopo l'esclusione dei siti condivisi.
- Coppia di Parquet mancante e hash incoerente con il manifest rifiutati.
- Esecuzione completa anche su curve sintetiche senza alcun sito: entrambi i rami restano non identificati, con tabelle vuote ed esportazione gestite correttamente.
- Hash degli input invariati dopo le esecuzioni. Nessun campionamento eseguito.
- Ispezione visiva dei grafici prodotti, incluso lo zoom a 240 Hz e il confronto tra input `f1` e `delta_hat`.

Gli strumenti di verifica locali sono in `output/d2_validation`, insieme ai notebook eseguiti e ai risultati dei controlli. Per isolare problemi di accesso alle dipendenze Jupyter del venv sono stati installati strumenti di validazione soltanto in quella cartella; il notebook consegnato non dipende da essa.

## Limiti e uso sui dati completi

La coppia di Parquet FULL non è presente nel checkout locale. La diagnosi empirica del conteggio storico 67/1 è stata verificata negli output restituiti dall'utente; l'estensione più recente richiede una nuova esecuzione su Drive. Il notebook consegnato ha gli output vuoti per non confondere i test con la run completa. La costruzione opzionale del grafo PyMC non è stata eseguita; l'identità del modello è verificata a livello di codice.

In Colab usare un runtime CPU ed eseguire tutte le celle. Il notebook è autonomo e monta Drive se necessario. Al termine leggere `SUMMARY.md`, verificare la coincidenza tra siti salvati e ricostruiti, quindi esaminare `patch_candidates`, `replicate_support`, `harmonic_audit`, `gap_audit` e i grafici prima di decidere qualsiasi modifica al modello o al rilevatore.

## Estensione: profondità e sensibilità della soglia

Le sezioni 12–14 esplorano otto soglie, da 0.60 a 0.95, lasciando il riferimento 0.75 e gli input di D2 invariati. Il criterio pure resta il 2% del massimo. Le varianti registrano conteggi per ramo, siti condivisi e minimi fuori griglia; non scelgono una soglia e non attivano nuovi fit.

Ogni candidato patch esclusivo viene confrontato con due controlli a ±4 Hz e, separatamente, ±6 Hz. La scelta delle frequenze dipende solo dalla geometria e dalla griglia disponibile. Coppie con controlli troppo vicini alle griglie teoriche vengono escluse e documentate. Si confrontano profondità continue al punto e quote di finestre con un minimo rilevato, distinguendo curve medie e repliche. La densità dei campioni nelle finestre non uniformi viene riportata. Le quote sui controlli non sono chiamate falsi positivi certificati e le repliche non vengono trattate come osservazioni indipendenti.

Le esecuzioni di verifica complete su fixture sintetica e Parquet smoke sono passate. Un'ulteriore fixture controllata verifica che passare da 0.75 a 0.80 possa recuperare sia minimi patch deboli sia minimi fuori griglia: il confronto continuo distingue un minimo patch più profondo dei controlli da un minimo patch di uguale profondità. Il gate originale resta invariato. Verificata anche l'invarianza delle frequenze di controllo quando si cambiano tutte le risposte mantenendo la griglia, e la gestione dell'assenza di candidati patch esclusivi.

Gli output nuovi includono `threshold_totals`, `threshold_sites`, `paired_control_audit`, `paired_control_coverage`, `paired_depth_summary`, `paired_depth_by_geometry`, `paired_depth_stability` e `paired_window_detection_rates`, oltre a due nuovi grafici e al riepilogo esteso. Lo script locale di verifica mirata è `output/d2_validation/test_sensitivity_cases.py`.

Il notebook FULL eseguito restituito dall'utente ha verificato 360/360 righe della prima diagnosi. La seconda copia restituita, `model_D2_diagnostics_standalone_colab (1).ipynb`, ha eseguito anche la sensibilità: a 0.90 compaiono 22 siti patch medi (10 KernelSynth, 12 TSMixup), contro uno a 0.75. Compaiono anche 34 minimi fuori griglia; il loro conteggio non costituisce una stima del tasso di falsi positivi. Questi risultati sostengono un approfondimento esplorativo a 0.90, senza certificarne l'ottimalità.

## Estensione: audit mirato a 0.90

Le sezioni 15–17 ricostruiscono statistiche alternative dalle rilevazioni a 0.90 già presenti nella sensibilità. Per ciascun sito patch medio mostrano frequenza, indice armonico, novità rispetto a 0.75, repliche di supporto e relative frequenze. Mantengono separate le curve medie usate nel conteggio e le repliche usate nelle risposte di D2.

Le tabelle `audit090_f1_inputs`, `audit090_gap_inputs` e `audit090_endpoints` verificano la disponibilità delle risposte originali, la presenza della fondamentale, la variabilità di `f1` e la copertura delle geometrie. `audit090_gaps` distingue le armoniche intermedie condivise escluse dal metodo dalle armoniche esclusive non rilevate. Nessuna risposta viene divisa per l'indice armonico, nessuna riga viene selezionata per forzare la previsione e nessun modello viene stimato. Il gate di riferimento a 0.75 resta invariato; il conteggio alternativo è indicato come solo controllo numerico.

Verifiche della versione finale:

- Esecuzione completa su fixture sintetica, smoke locale, fixture con depressioni deboli e fixture senza siti, con esportazione e hash degli input verificati.
- Il sito sintetico a 240 Hz conserva armonica 15 e supporto 2/3 anche nell'audit a 0.90.
- Quattro siti deboli sintetici vengono recuperati a 0.90 con supporto 3/3, ma sono armoniche 3 e 5: le sei righe patch delle repliche risultano prive della fondamentale. Le distanze di 32 Hz attraversano la quarta armonica condivisa, con previsione di base pari a 16 Hz.
- Tabelle vuote e riepilogo funzionano senza siti; nessun conteggio alternativo modifica il gate originale.
- Grafico di supporto nelle repliche ispezionato visivamente. Sintassi e identità AST di D2 rispetto a entrambi gli originali nuovamente verificate.

Gli script locali sono `output/d2_validation/validate_notebook.py`, `test_sensitivity_cases.py` e `test_audit090.py`. Questi esiti sono verifiche software; la stabilità e gli indici armonici dei 22 siti FULL non sono ancora stati calcolati con questa estensione. Per ottenerli occorre rieseguire tutte le celle nel Colab dell'account usato per le run, riutilizzando gli stessi Parquet. L'account Drive delle run è diverso da quello collegato a Codex.
