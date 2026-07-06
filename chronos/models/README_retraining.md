# Chronos-Bolt Tiny — Retraining Pipeline (`chronos_bolt_pretraining_TSMIXUP.ipynb`)

## Obiettivo

Questo notebook fa **solo training puro** (nessuna valutazione/inference) di Chronos-Bolt Tiny **da zero**
(pesi random, non fine-tuning), sul corpus originale di pre-training di Chronos, variando **solo**
`input_patch_size` (P) e `input_patch_stride` (S). Serve come blocco riusabile: si lancia una volta per ogni
combinazione P/S che si vuole confrontare nello studio sull'aliasing strutturale, e produce un checkpoint
nominato in modo da poter essere confrontato con gli altri.

Non fa parte di questo notebook: la generazione dei segnali sintetici a frequenza controllata (quello è compito
di `chronos/data/synthetic/signalGenerator.py`), l'ontologia, il probing, la valutazione Bayesiana. Questo
notebook produce solo il modello allenato; il resto della pipeline sperimentale lo consuma a valle.

## Scelte fatte e perché

### 1. Model id corretto: `amazon/chronos-bolt-tiny`
La versione precedente del notebook (`toFix.ipynb`) usava `autogluon/chronos-bolt-tiny`, che non esiste su
HuggingFace. `autogluon` è solo il nome del pacchetto Python per l'inferenza zero-shot; il checkpoint reale è
sotto l'organizzazione `amazon`. Notare che qui la config serve solo come riferimento architetturale: i pesi
**non** vengono caricati, sono reinizializzati da zero (necessario perché cambiare `input_patch_size` cambia la
dimensionalità del primo blocco di embedding, che dipende da `input_patch_size * 2`).

### 2. Dataset: solo `autogluon/chronos_datasets` / `training_corpus_tsmixup_10m`
Tre alternative erano sul tavolo:
- i segnali sintetici locali del progetto (`chronos/data/synthetic/`, con frequenza iniettata nota) — scartati
  per questo notebook perché l'obiettivo qui è allenare su dati "originali Chronos", non su dati costruiti per
  il probing dell'aliasing (quelli servono altrove, per i test di ipotesi con frequenza controllata);
- entrambi i corpus ufficiali (`training_corpus_tsmixup_10m` + `training_corpus_kernel_synth_1m`) mescolati con
  probabilità fissa — scartato per semplicità: la logica di mixing tra due stream (con fallback quando uno dei
  due si esaurisce) aggiunge complessità senza portare beneficio decisivo per un modello "tiny";
- **un solo corpus, `training_corpus_tsmixup_10m`** — scelto perché è il corpus principale (10M serie, basato su
  dati reali), mentre KernelSynth (1M) è ausiliario e minoritario nel paper originale (~10%).

Il dataset è caricato in **streaming** (`streaming=True`): niente download completo di un corpus da milioni di
serie, si scarica solo ciò che serve per gli step effettivamente eseguiti.

### 3. Un solo parametro variabile per asse: P e S
`context_length`, `prediction_length`, `quantiles`, batch size, learning rate, ecc. restano fissi tra le run.
Questo è l'unico modo per attribuire una differenza di comportamento del modello a P/S e non ad altre variabili
del setup — condizione necessaria per qualunque confronto quantitativo nella relazione finale.

### 4. Warmup + scheduler LR lineare — aggiunti
Con LR costante da pesi random, i primi step di training sono spesso instabili. Se questa instabilità colpisce
in modo diverso configurazioni P/S diverse (es. patch piccoli → sequenze di patch più lunghe → gradienti con
scala diversa), rischia di confondersi con l'effetto di aliasing che si vuole isolare. Aggiungere `warmup_ratio`
+ `lr_scheduler_type="linear"` (lo stesso schema usato in `train.py` originale) costa una manciata di righe e
riduce questo rischio.

### 5. Shuffle buffer sullo stream — aggiunto
Un dataset in streaming legge le righe nell'ordine in cui compaiono negli shard Parquet. Senza un buffer di
shuffle, batch consecutivi potrebbero essere correlati (es. serie simili raggruppate nello stesso shard).
`train_stream.shuffle(buffer_size=...)` mitiga questo rischio con un costo trascurabile.

### 6. Cosa NON è stato aggiunto, e perché
Il `train.py` ufficiale di Chronos (Chronos-T5, non Bolt) usa anche `transformers.Trainer`/`TrainingArguments` e
un campionamento delle finestre pesato per lunghezza serie (`ExpectedNumInstanceSampler` di GluonTS). Entrambi
sono stati **scartati deliberatamente**:
- `Trainer`/`TrainingArguments` è solo un wrapper per uso in produzione (logging, checkpointing distribuito): non
  cambia nulla di scientificamente rilevante per confrontare P/S, aggiunge solo dipendenze e complessità;
- il campionamento pesato per lunghezza serie cambia la distribuzione dei dati visti, ma **nello stesso modo per
  ogni configurazione P/S** — quindi non aiuta a isolare l'effetto di P/S, aiuta solo a somigliare più
  fedelmente al regime ufficiale di pre-training, cosa che la vostra stessa nota metodologica esclude
  esplicitamente come obiettivo ("non reproduction ufficiale").

In sintesi: le due aggiunte fatte proteggono la comparabilità tra run; le due scartate avrebbero solo aumentato
la fedeltà alla pipeline ufficiale, senza aiutare l'obiettivo dello studio.

### 7. Nessuna valutazione in questo notebook
Il notebook salva solo la curva di training loss (diagnostica del processo di ottimizzazione, non del modello
allenato) e il checkpoint finale. Qualunque misura di qualità delle previsioni (MAE, MASE, WQL, analisi per
frequenza/cpp) è compito di un notebook di valutazione separato, a valle, comune a tutte le configurazioni P/S.

### 8. Naming dell'output
```text
outputs/chronos-bolt-tiny-retrain-p{INPUT_PATCH_SIZE}-s{INPUT_PATCH_STRIDE}/
```
Checkpoint intermedi ogni `SAVE_EVERY` step, più il modello finale nella stessa cartella. Il nome incorpora P e S
per permettere di lanciare più run senza sovrascriverle a vicenda.

## Stima dei tempi

Stima basata sulla configurazione (Chronos-Bolt tiny, ordine di grandezza 8-16M parametri, `context_length=2048`,
batch size 32, `max_steps=10 000`), **non** su un benchmark eseguito — è un ordine di grandezza per pianificare,
da correggere con il tempo/step osservato al primo checkpoint reale (dopo il primo 10% degli step, con
`SAVE_EVERY=1000`).

| Hardware | Tempo per una run (10 000 step) |
|---|---|
| Solo CPU | 3–8 ore (da evitare anche per un test) |
| GPU entry-level (T4, RTX 3060, Colab free) | 30–60 minuti |
| GPU più potente (RTX 4090, V100, A100) | 10–20 minuti |

Le diverse combinazioni P/S cambiano il numero di patch per finestra e quindi il costo per step, ma per un
modello "tiny" la differenza tra configurazioni ragionevoli è dell'ordine di decine di %, non di multipli — la
tabella sopra vale come ordine di grandezza per qualunque P/S.

**Per uno sweep di più configurazioni** (es. baseline (16,16) + alcune varianti, come nella prima fase pianificata
del progetto):

| Hardware | Sweep di 3-6 configurazioni (sequenziale) |
|---|---|
| GPU entry-level | 2–6 ore totali |
| GPU più potente | 1–2 ore totali |
| Solo CPU | impraticabile (>1 giornata) |

Un fattore aggiuntivo non incluso nella stima sopra: il tempo di **streaming del corpus** (download degli shard
Parquet di `training_corpus_tsmixup_10m` on-the-fly) dipende dalla banda della rete, non dalla GPU — con
connessione lenta può diventare il vero collo di bottiglia più del calcolo stesso.

## Come lanciare esperimenti multipli

Ripetere il notebook cambiando solo `INPUT_PATCH_SIZE` e `INPUT_PATCH_STRIDE` in testa (Sezione 1), mantenendo
fisso tutto il resto. Ogni run produce un output separato per nome; nessuna sovrascrittura tra configurazioni
diverse.
