# A3 standalone: correzione del campionamento, 9 settembre 2026

Notebook: `model_A3_localisation_standalone_colab.ipynb`.
SHA-256: `21eae5d8c368e9d47a93690d8e5f1344983e4ba6be10df2ceac46d51c4e47e62`.

## Cosa cambia

Il modello scientifico A3 rimane quello consegnato. La cella che costruisce il modello è identica alla versione precedente; sono identiche anche le funzioni di generazione del recovery, la diagnostica R-hat/ESS, i contrasti decisionali e il PPC. Non sono stati modificati likelihood, prior, pesi, tolleranza in frequenza, numero di picchi, popolazione o quantità di campioni.

Il campionatore ora usa `jitter+adapt_diag`. La precedente configurazione `jitter+adapt_full` usa un aggiornamento iniziale di covarianza piena non regolarizzato che, nel controllo numerico, risultava singolare alle dimensioni del PILOT A3: 663 coordinate contro 103 punti, quindi rango al massimo 102. La selezione dell'adattamento pieno è ora rifiutata prima di campionare o ricaricare un fit. La correzione riguarda il calcolo della posteriore e non la struttura probabilistica del modello.

L'implementazione locale di `QuadPotentialDiagAdapt` e `_WeightedVariance` di PyMC 5.28.5 è stata confrontata via AST con il sorgente PyMC 6.0.1 usato nella run Colab precedente: le due classi sono identiche. La regressione numerica percorre 500 aggiornamenti su 663 coordinate e verifica varianze positive e finite e coerenza fra varianze e fattori. Questo controllo attraversa la transizione che il vecchio smoke da 15 warmup non raggiungeva.

Fonti: [PyMC init_nuts](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.init_nuts.html), [sorgente PyMC 6.0.1](https://github.com/pymc-devs/pymc/blob/v6.0.1/pymc/step_methods/hmc/quadpotential.py). La documentazione stable descrive l'API corrente; il confronto dell'algoritmo è stato fatto sul tag preciso v6.0.1. La riproduzione del problema precedente è in `output/a3_review/`.

## Uso del notebook corretto

- `RUN_MODE = "PILOT"`.
- Nuovo `RUN_ID = "model_A3_local_frequency_diag_v2"`, già impostato.
- Quattro catene CPU sequenziali, con salvataggio e verifica NetCDF dopo ogni catena completa.
- Fit primario: 2.000 warmup e 2.000 draw conservati per catena.
- Recovery: 1.500 warmup e 2.000 draw conservati per catena.
- FULL: 2.000 warmup e 3.000 draw conservati per catena, dopo un PILOT compatibile che supera tutti i gate.
- Sorgenti A1/A2/A3 compatibili riutilizzate; `ALLOW_NEW_FORECASTS = False`. La ricerca AUTO include anche la precedente run `model_A3_local_frequency_v1`.

**Questa revisione non è pronta per FULL: il recovery locale non supera ancora la convergenza.** Per riprodurre il controllo diagnostico in Colab, usare un runtime CPU e fare restart/run all. Se il setup installa l'ambiente e riavvia il runtime, fare nuovamente Run all dopo la riconnessione. Per riprendere una run della nuova revisione, mantenere RUN_ID, sorgente, codice e impostazioni invariati: le catene completate vengono ricaricate; quella interrotta ricomincia il warmup. I checkpoint prodotti con il vecchio campionatore non vengono combinati con quelli nuovi.

Per FULL cambiare solo RUN_MODE dopo un PILOT PASS, mantenendo il nuovo RUN_ID. PILOT e FULL hanno directory separate. Nessuna raccolta o posteriore storica viene cancellata.

## Recovery e gestione delle interruzioni

Il recovery moderato deve passare prima che inizi quello sparso; entrambi devono passare prima del fit empirico. Su un fallimento il notebook salva le diagnostiche, le coperture per parametro/famiglia, `recovery_status.json` e un `final_verdict.json` esplicitamente NOT REPORTABLE, poi interrompe Run all. In tale caso fit empirico e PPC empirico non sono stati eseguiti. Non è un PPC superato, né un PPC fallito. Anche avviando manualmente la cella primaria occorre un recovery superato per l'analysis fingerprint corrente.

Ogni catena conserva il suo `*.chain_XX.nc`. I report `*.sampler_health.parquet` aggiungono BFMI dell'energia, accettazione, step size e profondità/dimensione degli alberi quando disponibili. Queste informazioni aiutano a diagnosticare il campionatore; non alterano le soglie dichiarate. Le righe di avanzamento sono testo normale e includono iterazione, tempo, step size, albero e divergenze posteriori; non richiedono ipywidgets.

Tutte le catene richieste contribuiscono alle diagnostiche finali. Non vengono eliminate catene per migliorare ESS o copertura. Posteriori vuoti, parziali, non finiti o incompatibili restano rifiutati. Il vecchio argomento `idata_kwargs['log_likelihood']` è stato omesso: PyMC 5/6 non calcolano log_likelihood per default, e il controllo strutturale continua a rifiutare array delle osservazioni salvati per ogni draw.

## Modello e soglie conservati

La risposta è Bernoulli: almeno uno dei tre picchi selezionati entro ±1 Hz dalla frequenza iniettata in quel braccio. I trial con identici predittori sono aggregati con likelihood Binomiale esattamente equivalente. Non si usa una soglia di recovery dell'ampiezza.

Il predittore è `beta_c + u_f + v_b + r_cf + (gamma_c + ell_cf) L + (kappa_c + s_cf) D`, con L=1 per lock e D=-1/0/+1 per lo/lock/hi. I prior globali A2 e i tre prior locali HalfNormal(1), con effetti Normali centrati entro geometria usando i pesi dei trial lock, sono invariati. Le coordinate di riferimento a 64 Hz e la base dei contrasti locali sono invariate. La specifica completa rimane nella cella metodologica del notebook.

L'effetto per sito è `gamma_c + ell_cf`; la media pesata entro geometria è `gamma_c`; l'effetto di disegno è la media equiponderata delle 15 geometrie. L'odds ratio confronta lock con la media geometrica delle odds lo/hi.

- Supporto: P(OR < 0,8) >= 0,95.
- Equivalenza: P(1/1,1 < OR < 1,1) >= 0,95.
- Effetto opposto: P(OR > 1,25) >= 0,95; altrimenti inconclusivo.
- Convergenza: tutti i valori finiti, R-hat < 1,01, ESS bulk e tail > 400 in PILOT e > 1.000 in FULL, zero divergenze.
- Entrambi i recovery: convergenza, almeno 80% di copertura degli intervalli al 95% e copertura di gamma_design.
- PPC: almeno 90% di copertura per ciascuno dei tre controlli. Il gate fine è globale sui 615 strati; le coperture per geometria sono riportate separatamente.
- FULL: restano necessari i gate strumentali, il PILOT compatibile e le sensitività prior/link già dichiarate.

## Verifica software

Dieci test sono passati nell'ambiente locale PyMC 5.28.5 / ArviZ 0.23.4. Comprendono struttura e sintassi, identità della likelihood e delle densità nelle due coordinate, prior locale e centratura, PPC indipendente, recovery riproducibile, DataTree compatibile, interruzione/ripresa dei checkpoint, nuova regressione dell'adattamento a 663 dimensioni, calcolo del BFMI e blocco dei fit successivi dopo un recovery fallito. Sono rifiutati anche cambi del campionatore su un checkpoint già salvato.

Il piccolo test di integrazione con NUTS reale ha completato quattro fit da due catene con 15 warmup + 8 draw e otto checkpoint. Ha verificato il blocco del recovery sottodimensionato. Poi il solo harness di test ha chiamato direttamente i componenti successivi per verificare report, grafici, PPC, conversione probit e ricaricamento, lasciando i flag di recovery falsi. Il notebook consegnato non contiene un bypass del gate. Il verdetto dello smoke è NOT REPORTABLE; i pochi draw non validano la convergenza. Risultati: `output/a3/smoke_result.json`.

```powershell
& '.\.venv\Scripts\python.exe' 'chronos\bayesian\tests\test_model_A3_notebook.py' -v
& '.\.venv\Scripts\python.exe' 'chronos\bayesian\tests\test_model_A3_notebook.py' --smoke
```

## Recovery alla dimensione del PILOT

Il recovery moderato completo ha terminato quattro catene da 1.500 warmup + 2.000 draw, in 1.502 secondi locali. Risultati:

| Metrica | Valore | Soglia | Esito |
|---|---:|---:|---|
| R-hat massimo | 1,013325 | < 1,01 | fallisce |
| ESS bulk minimo | 177,30 | > 400 | fallisce |
| ESS tail minimo | 139,10 | > 400 | fallisce |
| Divergenze | 0 | 0 | passa |
| Copertura degli intervalli | 93,40% | >= 80% | passa |
| Copertura di gamma_design | sì | sì | passa |

**Il recovery moderato NON passa. La correzione del difetto della covarianza non ha risolto tutta la difficoltà di campionamento.** Il confronto con la run precedente mostra un miglioramento rispetto a R-hat 2,853 / ESS bulk 4,59 / copertura 26,85%, ma la soglia è invariata e il miglioramento non basta per autorizzare il fit empirico. Le due esecuzioni usano ambienti/hardware diversi: non è un benchmark di velocità.

La verifica separata dello scenario sparso non è completa: sono disponibili due catene, su quattro richieste, e nessun risultato combinato. La prima ha 65 divergenze su 2.000 draw. Non viene calcolato né dichiarato un PASS da questo sottoinsieme. Le divergenze della prima catena sono associate a valori bassi di tau (mediana 0,106 nei draw divergenti contro 0,359 negli altri) e in parte sigma_kappa (0,159 contro 0,290). È un indizio di ulteriore difficoltà nella parametrizzazione gerarchica; non dimostra una causa unica. Non è stata introdotta una riparametrizzazione aggiuntiva non verificata dopo la richiesta di concludere.

Artifact: `output/a3_sampler_fix/moderate_diag/result.json`, `parameter_diagnostics.csv`, `recovery_summary.csv`, `sampler_health.csv`; per lo sparso `output/a3_sampler_fix/sparse_diag/incomplete_status.json` e `first_chain_divergence_scales.csv`. Tutte le catene complete sono conservate. Alla conclusione non risultano più processi di questo harness attivi. La run moderata era stata avviata prima dell'aggiunta dei report accessori: modello, campionatore, seed, draw, warmup e target_accept sono identici; l'omissione del flag log_likelihood usa lo stesso default False verificato nel sorgente e nello smoke finale.

Il controllo usa il disegno da `chronos/bayesian/_run/full/model_A2_localisation_bernoulli_v2/data/A_counts.parquet`, con tre background per generatore. Elimina tutti gli hit empirici e li sostituisce con gli esiti dei generatori sintetici invariati. Verifica 205 siti, 570 contrasti locali, 663 coordinate campionate e 33.390 trial. Non effettua previsioni Chronos né inference empirica. I tassi sintetici sono gli stessi della run Colab precedente: 0,4274633124 nello scenario moderato e 0,0761305780 in quello sparso.

```powershell
& '.\.venv\Scripts\python.exe' 'chronos\bayesian\tests\validate_A3_recovery.py' --scenario moderate
& '.\.venv\Scripts\python.exe' 'chronos\bayesian\tests\validate_A3_recovery.py' --scenario sparse
```

Lo script limita ogni invocazione a 30 minuti durante il campionamento e conserva le catene già complete. Per un nuovo confronto dopo modifiche al codice usare `--output` con una directory nuova. I campioni non possono essere riutilizzati se il fingerprint dei parametri cambia.

## Limiti

Non è stato rieseguito il PILOT empirico corretto e non sono stati generati nuovi forecast. Setup/installazione Colab, accesso Drive e raccolta non sono stati rieseguiti. I nuovi test NUTS sono locali in PyMC 5.28.5/ArviZ 0.23.4; la compatibilità DataTree è stata verificata con il ramo di serializzazione e l'algoritmo di adattamento è stato confrontato col sorgente PyMC 6.0.1, senza installare qui una seconda distribuzione Python 3.13/PyMC 6.

Un recovery corretto non dimostra che il modello descriva bene i dati reali. Il PPC resta condizionale ai background del fit e non dimostra capacità predittiva su background esclusi. La precedente copertura PPC del 100% derivava da una run non convergente e non è promossa a risultato valido da questa correzione.

Il notebook A2 improved è invariato, SHA-256 `972b0da9d06d09ae6749d3d5b874693256f5fbe918ed321a9505f6e987be2688`. Il notebook corretto non contiene output sintetici presentati come risultati empirici.
