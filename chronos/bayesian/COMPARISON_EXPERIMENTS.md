# Confronti aggiuntivi

## Notebook

- `complex_signal_sweep.ipynb`: TSM-10 e KS-10, solo checkpoint ufficiale e retrainato 16–16; curve delle frequenze dominanti e mappe dei test binari LL/L/LH/Mid/HL/H/HH più discriminazione locale ±1 Hz.
- `solar_telemetry_comparison.ipynb`: viste esistenti più benchmark dei 16 checkpoint, integrato dal notebook Colab. Default PV_Power, 150 finestre, contesto 2048, target 64, quattro processi CPU con due thread ciascuno.
- `solar_chronos2_comparison.ipynb`: confronto separato fra Chronos-2, Bolt tiny ufficiale e retrainato 16–16. Target PV_Power (il CSV non contiene `pv_array`); tutte le altre misure sono covariate storiche di Chronos-2. Bolt resta univariato.

Le revisioni ufficiali sono fissate nei moduli. Chronos-2 usa `cross_learning=False`: una finestra non accede alle altre. Le covariate non includono misure future. Il suo miglioramento eventuale comprende sia il modello sia l'informazione aggiuntiva.

## Configurazione e artefatti

`n_draws` e `allow_overlap` sono indipendenti. La disgiunzione riguarda l'intera finestra contesto+target (2112 minuti). Il campionamento disgiunto usa un impacchettamento casuale riproducibile, con fallback a un impacchettamento massimo; non è un campionamento uniforme di tutti i sottoinsiemi disgiunti. Le finestre sovrapposte sono campionate uniformemente senza ripetere origini.

Per PV_Power si conservano la pulizia degli spike e la preparazione del solo contesto. I target mancanti notturni assumono zero; i mancanti diurni sono esclusi. G_h è selezionabile nel benchmark a 16 modelli e richiede finestre complete. Le metriche aggregano gli errori puntuali validi con uguale peso. La varianza è quella degli errori, con ddof=0.

La tabella frequenziale usa il bin dominante Hann del contesto, in cicli/minuto: cpp=fP e cps=fS. Ogni riga aggrega gli errori per modello e frequenza; le figure sovrappongono MAE e varianza con due scale verticali. Le finestre piatte non contribuiscono al solo profilo frequenziale. Il benchmark conserva RMSE, WQL (somma e media sui quantili), pinball loss, coverage, risultati per regime e passo. Gli intervalli delle differenze MAE usano un bootstrap appaiato per cluster settimanali; sono descrittivi e senza correzione per confronti multipli.

Lo sweep usa dieci background per famiglia. KS-10 è ricostruito esclusivamente dai dieci maggiori termini Fourier fra 2 e 250 Hz, non da dieci kernel. TSM-10 usa dieci componenti della miscela Light TSMixup. L'ampiezza dell'injection è 1,5 volte la componente più ampia del background standardizzato, senza rinormalizzare la miscela. La curva rossa misura il massimo FFT sul rollout mediano di 512 punti; il test di decomposition già esistente resta sul forecast diretto di 64 punti. Le mappe dei band task riportano accuratezze per frequenza da fold che escludono intere frequenze. Il test locale esclude interi background.

Ogni run salva sotto `_run/comparison/<esperimento>/<full|smoke>/<hash>/`:

- manifest con configurazione, campione, dataset, codice e identità dei modelli;
- blocchi di previsioni/quantili o rappresentazioni riprendibili e file di avanzamento;
- CSV delle prove e delle tabelle, immagini PNG e `complete.json` solo a run concluso.

La cache dei pesi Chronos-2 è interna a `_run/comparison/model_cache`. I run precedenti, incluso il vecchio solar da 50.000 origini interrotto dopo la modifica del protocollo, restano separati. Nessuna tabella parziale viene pubblicata come completa.

## Esecuzione locale

Dalla radice del repository, usare l'ambiente Python del progetto:

```powershell
.venv/Scripts/python.exe chronos/bayesian/support_scripts/run_extended_comparisons.py benchmark
.venv/Scripts/python.exe chronos/bayesian/support_scripts/run_extended_comparisons.py solar --smoke
.venv/Scripts/python.exe chronos/bayesian/support_scripts/run_extended_comparisons.py sweep --smoke
.venv/Scripts/python.exe chronos/bayesian/support_scripts/run_extended_comparisons.py chronos2 --smoke
.venv/Scripts/python.exe chronos/bayesian/support_scripts/run_extended_comparisons.py solar
.venv/Scripts/python.exe chronos/bayesian/support_scripts/run_extended_comparisons.py sweep
.venv/Scripts/python.exe chronos/bayesian/support_scripts/run_extended_comparisons.py chronos2
.venv/Scripts/python.exe -m unittest discover -s chronos/bayesian/tests -p "test_*comparison*.py" -v
```

Il parallelismo del benchmark solar è fra checkpoint su CPU. Su una singola GPU si usa un checkpoint alla volta con batch. La riesecuzione verifica e riutilizza i blocchi completi; un'identità diversa produce un'altra directory. I file atomici tollerano brevi blocchi di lettura Windows durante la sostituzione.
