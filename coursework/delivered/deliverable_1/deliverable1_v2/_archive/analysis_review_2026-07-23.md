> ⚠️ **ARCHIVIATO / OBSOLETO (superato).** Questa review è del 23/07/2026 e riflette uno stato
> precedente del deliverable. Da allora sono stati recepiti: reframing "ripetizione ≠ indistinguibilità",
> ipotesi H1–H3 (H4 rimossa), Probing Methodology allineata (encoder+decoder `[REG]` + `output_reg`
> + `output_head`, controlli shuffled + random-init, fasi `S_f`), banda 250 Hz, tabella patchStride.
> Tenuto solo come storia. Per lo stato corrente vedere `sections/deliverable1.tex`.

# Review tecnico ed editoriale di `deliverable1_v1`

Data della revisione: 23 luglio 2026.

## Esito in breve

Le richieste di `corrections.txt` non sono ancora state recepite completamente. Dividendo il feedback in 14 richieste verificabili, lo stato è:

- **2 implementate**;
- **7 implementate solo in parte**;
- **5 mancanti**.

I blocchi più importanti non sono grammaticali, ma scientifici:

1. la derivazione attuale dimostra la ripetizione di patch consecutive dello stesso segnale, non l'indistinguibilità tra due segnali;
2. le ipotesi H1--H3 sono ancora un placeholder;
3. la sezione bayesiana è ancora un placeholder e il tentativo nel notebook contiene 80 casi `safe` e 0 casi `blind`, quindi l'effetto non è identificabile;
4. la proposta MDL non specifica correttamente tensori, probe, codelength e controlli;
5. la procedura di training non è una riproduzione fedele di Chronos-Bolt e non contiene validation, early stopping o più seed;
6. `T=512 tokens` è errato: il checkpoint ufficiale accetta fino a 2048 **osservazioni**; con 512 osservazioni e `P=S=16` si ottengono 32 patch token più il token di regressione.

Questa analisi propone correzioni e testo inglese pronto da trasferire nel deliverable. Non modifica direttamente il `.tex`.

## 1. Stato puntuale delle corrections

| Richiesta | Stato | Evidenza nel deliverable | Cosa resta da fare |
|---|---|---|---|
| Mettere Domain prima di Purpose | **Inserita** | `sections/deliverable1.tex:3`, poi Purpose a `:10` | Nessuna modifica strutturale necessaria. |
| Ridurre e integrare Intended Audience | **Parziale** | Il contenuto è nel paragrafo a `:8` | Accorciare e correggere `individuals ... which are interested`. |
| Mettere Significance dopo il project outline | **Inserita** | Sezione a `:184` | Valutare se portarla immediatamente dopo Purpose; oggi è dopo metodologia e security. |
| Aggiungere una sezione Data dedicata | **Parziale** | `Signal Generation and Dataset` a `:130` | Centralizzare quantità, distribuzioni, split, seed, missing values, preprocessing e assenza di leakage. Oggi parte dei dati è descritta anche nelle sezioni Methodology/Evaluation. |
| Formalizzare tokenizzazione e indistinguibilità | **Parziale, ma concettualmente errata** | Operatore a `:24-38`, phase locking a `:40-121` | Definire due segnali `x,y` e la condizione `T(x)=T(y)` oppure una near-collisione `R_l(x)≈R_l(y)`. La derivazione corrente prova soltanto `x[n+S]=x[n]`. |
| Formulare H1, H2, H3 testabili | **Mancante** | Placeholder a `:124-128` | Inserire ipotesi con variabili, confronto, metrica e criterio di falsificazione. |
| Dire quali rappresentazioni vengono sondate con MDL | **Parziale** | `:171` menziona decoder; `:173` anche encoder/output | Specificare input embedding, `[REG]` dopo ogni encoder, singolo stato dopo ogni decoder e head come endpoint comportamentale. |
| Definire le metriche MDL | **Mancante** | MDL solo nominato a `:171` | Inserire online codelength in bit, bit/label e compression ratio, con protocollo prequential. |
| Confrontare i layer | **Parziale** | Promessa generica a `:173` | Stessa dimensionalità/probe/dati per tutti i layer, intervalli bootstrap e differenza lock vs controlli locali. |
| Interpretare MDL rispetto alle ipotesi | **Mancante** | Le ipotesi non esistono | Definire in anticipo quale andamento supporta o confuta H1--H3. |
| Pipeline synthetic → blind spots → PV | **Parziale** | Ordine dichiarato a `:133-135` | Specificare discovery solo su synthetic, conferma su nuovi synthetic seed, poi validazione esterna PV; il PV non deve essere usato per scegliere soglie/prior. |
| Motivare retraining e riprodurre la baseline | **Parziale** | Retraining a `:140`, baseline a `:144`; placeholder a `:148-154` | Distinguere cambio di stride, tecnicamente compatibile con i pesi, da cambio di patch size, che cambia l'input embedding. Aggiungere validation e criterio di equivalenza col checkpoint stock. |
| Provare stride su entrambi i lati di 16 | **Mancante come confronto stride-only** | Con `P=16` la tabella usa solo `S={4,8,12,16}` | Aggiungere almeno `S={8,12,16,20,24}` a `P=16`. Dichiarare che `S>P` introduce campioni non coperti, quindi è un regime con gap. |
| Descrivere i prior bayesiani | **Mancante** | `Bayesian analysis.` a `:175-177` | Definire estimand, likelihood, prior per effetti e scale, prior predictive check e sensitivity analysis. |

## 2. Correzioni del testo

### 2.1 Grammatica e lessico, nell'ordine del testo

| Punto | Testo/problema attuale | Correzione proposta |
|---|---|---|
| `deliverable1.tex:4` | `evaluate production variations ... dynamically modify the internal load, attach and detach the battery unit` | `forecast short-term variations in photovoltaic production so that the controller can adjust the load and battery dispatch, maintain local-grid stability, and detect anomalies.` |
| `:8` | `individuals ... which are interested` | `The study is intended for researchers and engineers working on Chronos-based time-series forecasting or energy management who need to understand the architecture's representational limits.` |
| `:40` | `Considering ... yielding ...` è un frammento | `Consider a continuous-time periodic signal sampled uniformly at frequency $f_s$, yielding the discrete-time sequence $x[n]$.` |
| `:133` | `To do so elementary`; `light implementation` | `To do so, elementary ...`; `a lightweight implementation`. |
| `:135` | `contains one-minute multivariate telemetry stream` | `contains a one-minute-resolution multivariate telemetry stream`. |
| `:138` | `At first`, `HuggingFace`, `tested over`, `sistematically varieted`, `strcuctural` | `Initially`, `Hugging Face`, `tested on`, `systematically varied`, `structural`; dividere il periodo in almeno due frasi. |
| `:140` | `according to the value reported`; `require a new training` | `according to the values reported`; `require retraining`. |
| `:144` | `Further possible training ...` è un frammento | `If computationally feasible, we will also train variants ...`. |
| `:146` | `fixed patch size to default value 16` | `the patch size fixed at its default value of 16`. |
| `:148` | `based over the information`; `10M data` | `based on the available information`; `10 million time series`. |
| `:156` | `Models Evaluation` | `Model Evaluation`. |
| `:157` | `After model re-training` | `After retraining the models, ...`. |
| `:159` | `sistematically varieted` | `systematically varied`. |
| `:160` | Punto e virgola prima di `to evaluate` | Usare una virgola oppure separare in due periodi. |
| `:163` | `re-trained models, to evaluate` | `retrained models to evaluate`. |
| `:165` | `Also this set ...` | `This third set will also be used ...`. |
| `:167` | `a SEED is set` | `a random seed will be fixed`. |
| `appendixA.tex:49,52,55` | `Formal description ...` sono frammenti | Iniziare con `This subsection formally describes ...`. |

Correzioni meccaniche aggiuntive:

- usare sempre `Chronos-Bolt-Tiny`, `Hugging Face`, `retraining` e `retrained`;
- aggiungere lo spazio prima delle citazioni: `framework \cite{...}.`, non `framework.\cite{...}`;
- sostituire `Security Implication` con `Security Implications`;
- in `tables/patchStride.tex:3`, la specifica `{cccl}` dichiara quattro colonne ma la tabella ne usa tre: deve essere `{ccc}`;
- eliminare tutti i marker `--- DA SISTEMARE ---`, `Ipotesi H1, H2, H3`, `Possible description ...` e `Bayesian analysis.`.

### 2.2 Forma, struttura e leggibilità

1. **Tempo verbale.** Usare il presente per fatti verificati sul modello e il futuro per il protocollo proposto. Ora presente, futuro e present perfect vengono alternati senza una regola.
2. **Apertura.** La prima frase è fuori da ogni sezione e ripete Purpose: integrarla in `Purpose and Scope`.
3. **Audience.** Una frase è sufficiente; non serve una sezione autonoma.
4. **Data.** La sezione deve contenere, in un unico punto: fonte/licenza, unità di campionamento, target, numero di serie/finestre, split, generatore sintetico, distribuzioni di frequenza/fase/ampiezza/SNR, missingness, normalizzazione e regola anti-leakage.
5. **Metodo vs risultati attesi.** Non presentare `collapse`, `data deletion` o `deception` come fatti prima dell'esperimento. Usare `potential loss of recoverable frequency information` e riservare il linguaggio di attacco a un threat model esplicito.
6. **Fonti.** Sostituire DeepWiki con la [configurazione ufficiale di Chronos-Bolt-Tiny](https://huggingface.co/amazon/chronos-bolt-tiny/blob/main/config.json), il [codice ufficiale Chronos-Bolt](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py) e la [model card](https://huggingface.co/amazon/chronos-bolt-tiny).
7. **Limite di 2000 parole.** Il conteggio grezzo attuale è circa 1617 parole nel corpo e 224 nell'appendice, equazioni escluse. Se l'appendice rientra nel limite, non c'è spazio per aggiungere integralmente i protocolli sotto: nel deliverable va tenuta la versione compatta e spostati griglie/iperparametri in una tabella o appendice.

### 2.3 Logica e correttezza scientifica

#### 2.3.1 La formulazione matematica va corretta prima di tutto

Per una sequenza finita `x[0:T-1]`, definire l'operatore raw

$$
\mathcal T_{P,S}(x)
=
\left(x[kS:kS+P-1]\right)_{k=0}^{K-1}.
$$

Sia inoltre

$$
U_{P,S}=\bigcup_{k=0}^{K-1}\{kS,\ldots,kS+P-1\}
$$

l'insieme degli indici effettivamente coperti. Allora

$$
\mathcal T_{P,S}(x)=\mathcal T_{P,S}(y)
\iff
x[n]=y[n]\quad\forall n\in U_{P,S}.
$$

Con copertura completa, la patchizzazione raw è iniettiva: due input diversi non collidono. Con `S>P` restano gap e due segnali possono differire soltanto negli indici non coperti. Con `S\le P`, salvo eventuali campioni di bordo esclusi dall'implementazione, le patch coprono la sequenza; quindi non è corretto attribuire automaticamente una collisione alla sola patchizzazione.

Per una sinusoide campionata

$$
x_f[n]=A\sin(2\pi fn/f_s+\phi),
$$

la condizione

$$
\frac{fS}{f_s}=c\in\mathbb Z
$$

implica `x_f[n+S]=x_f[n]`, cioè **ripetizione delle patch dello stesso segnale**. Non implica che due frequenze distinte producano lo stesso vettore intra-patch. La classe

$$
f_c=c\frac{f_s}{S},\qquad 1\le c<S/2,
$$

va quindi chiamata `stride-locked frequency class`, non insieme già dimostrato di segnali indistinguibili.

La collisione scientificamente interessante deve essere definita dopo preprocessing e rete. Per esempio, indicando con `R_l(x)` la rappresentazione al layer `l`:

$$
x\not\equiv y,\quad q(x)\ne q(y),\quad
\lVert R_l(x)-R_l(y)\rVert_2\le\varepsilon,
$$

dove `q` è la proprietà target (frequenza, presenza del tono o ampiezza) ed `epsilon` è fissato prima del test. In alternativa, l'indistinguibilità operativa può essere definita come maggiore codelength MDL o performance del probe vicina al controllo random-label. Questo trasforma `structural aliasing` da conclusione presupposta a ipotesi empirica falsificabile.

Altre correzioni conseguenti:

- usare `=` nelle derivazioni e riservare `:=` alle definizioni;
- esprimere sempre la frequenza anche come `f/f_s` o cycles per stride, perché `2--256 Hz` non è interpretabile senza `f_s`;
- se `f_s=512 Hz`, `256 Hz` è Nyquist e contraddice la condizione stretta `f<f_s/2`; trattarlo separatamente oppure usare `2--255 Hz`;
- nel codice/notebook usare `cps=fS/f_s`, non `cpp=fP/f_s`, quando si varia lo stride;
- distinguere allineamento con `P`, allineamento con `S`, entrambi e nessuno, così da misurare effetti separati e interazioni.

#### 2.3.2 Configurazione corretta di Chronos-Bolt-Tiny

La [configurazione ufficiale](https://huggingface.co/amazon/chronos-bolt-tiny/blob/main/config.json) riporta:

- context length massimo: 2048 osservazioni;
- `P=16`, `S=16`;
- regression token attivo;
- hidden size 256;
- quattro encoder e quattro decoder;
- horizon 64 e nove quantili da 0.1 a 0.9.

Il modello normalizza l'input, concatena valori e observation mask, proietta ogni patch da `2P` a 256 dimensioni, aggiunge il token di regressione e usa il T5 encoder. Il decoder riceve **un solo start token** e ogni blocco produce un unico stato `[B,1,256]`; la head restituisce `9 × 64` quantili, non una distribuzione normalizzata. La [model card](https://huggingface.co/amazon/chronos-bolt-tiny/blob/main/README.md) descrive infatti un forecast quantile direct multi-step e un modello da circa 9M parametri.

Se l'esperimento usa 512 punti, scrivere `an evaluation context of 512 observations`. Con `P=S=16` sono 32 patch più `[REG]`, non 512 token. Con il massimo di 2048 osservazioni sono 128 patch più `[REG]`.

#### 2.3.3 Ipotesi H1--H3 pronte da inserire

```latex
\paragraph{Hypotheses.}
\textbf{H1 (local stride-lock degradation).} With patch size held fixed, signals at
$f_k=kf_s/S$ will exhibit higher frequency-probe description length and worse
amplitude recovery than phase-, amplitude-, and noise-matched signals at
$f_k\pm\delta f_s/S$. A null or reversed local contrast will refute H1.

\textbf{H2 (stride dependence).} When $S$ is changed while $P$ is fixed, the
locations of any local degradation will move proportionally to $1/S$. An effect
that remains tied to $P$ rather than $S$ will refute the proposed stride-lock
mechanism.

\textbf{H3 (representation-to-behaviour link).} A layer-wise increase in the MDL
required to decode tone presence or frequency at stride-locked conditions will
predict a paired deterioration in forecast amplitude recovery and WQL. If
forecast degradation occurs without representational loss, or vice versa, the
claim of an internal representation bottleneck will not be supported.
```

La generalizzazione al PV va trattata come validazione esterna secondaria, non incorporata in H1 come se fosse già garantita.

## 3. Bayesian statistics: analisi del progetto e prior consigliato

### 3.1 Perché l'analisi corrente non è utilizzabile

In `chronos/ontology/pv_full_analysis.ipynb` il modello corrente usa, in sostanza,

$$
r_i\sim Normal(\alpha+\beta\,I_{blind,i},\sigma),
$$

con `alpha ~ Normal(0.5,0.2)`, `beta ~ Normal(0,0.1)` e `sigma ~ HalfNormal(0.0616)`. I risultati salvati mostrano però 80 finestre `safe` e **0 `blind`**. Di conseguenza `beta` non è identificato dalla likelihood e il posterior (`0.001 ± 0.101`) ripete quasi esattamente il prior.

Inoltre:

- `0.0616` deriva dalla deviazione standard della potenza notturna in watt, mentre la risposta è una cosine similarity adimensionale: le scale sono incompatibili;
- una likelihood normale assegna massa fuori dal supporto `[-1,1]` della similarity;
- la risposta è ridondanza delle patch raw, non una proprietà delle rappresentazioni apprese o del forecast;
- finestre vicine/sovrapposte non sono indipendenti;
- `dominant_cpp_window` usa `fP/f_s` e ignora `S`;
- un singolo offset applicato all'intera finestra non spezza la fase relativa tra patch consecutive.

Il prior non può compensare l'assenza del gruppo di confronto. Prima serve un disegno bilanciato.

### 3.2 Cosa fanno i lavori simili e cosa è trasferibile

- [Pagani et al. (2026)](https://arxiv.org/abs/2605.06361) è il lavoro più vicino: studia sinusoidi controllate in Chronos e applica online MDL ai decoder. Giustifica il probing controllato, ma **non fornisce un prior bayesiano quantitativo** da copiare.
- Nella stima bayesiana di una frequenza ignota sono comuni supporti fisicamente limitati e prior uniformi sulla frequenza; si veda [Picci e Zhu (2020)](https://arxiv.org/abs/1910.09475). Altri approcci periodici usano prior uniformi sulla fase e prior uniformi o log-uniformi sul periodo/frequenza a seconda dell'invarianza desiderata.
- Per segnali reali con rumore colorato, [Vaughan (2010)](https://arxiv.org/abs/0910.2706) mostra perché nuisance model e posterior predictive checks sono parte dell'inferenza, non dettagli opzionali.
- Il [Bayesian workflow](https://arxiv.org/abs/2011.01808), i [prior predictive checks di Stan](https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html) e la sensitivity via [power scaling](https://arxiv.org/abs/2107.14054) supportano una scelta di prior verificata sulle predizioni e non giustificata soltanto come `weak`.

Nel vostro esperimento sintetico frequenza, fase e SNR sono **fattori randomizzati del disegno**, non parametri ignoti: la frequenza non ha bisogno di un prior inferenziale. La fase va campionata `Uniform(0,2π)` nel generatore.

### 3.3 Estimand e modello raccomandati

Usare come endpoint primario l'attenuazione locale dell'ampiezza prevista, non la ridondanza raw. Per ogni configurazione `(P,S)`, armonica, fase, background e seed del modello, generare una tripletta matched:

$$
f_0=k\frac{f_s}{S},\qquad
f_-=f_0-0.25\frac{f_s}{S},\qquad
f_+=f_0+0.25\frac{f_s}{S}.
$$

Le tre serie condividono background, fase, ampiezza e SNR. Sia

$$
R=A_{pred}/A_{true}
$$

il recovery ratio stimato sul forecast. Definire

$$
d_i=\log(R_{0,i}+0.01)
-\frac{\log(R_{-,i}+0.01)+\log(R_{+,i}+0.01)}{2}.
$$

`d<0` indica un'attenuazione locale alla frequenza stride-locked, controllando il normale andamento con la frequenza.

Per rendere `R` riproducibile, usare la previsione mediana (`q=0.5`) e stimare `A_pred` con una regressione least-squares, sullo stesso horizon di 64 punti, della forma

$$
\hat y[h]=a\sin(2\pi fh/f_s)+b\cos(2\pi fh/f_s)+c,
\qquad A_{pred}=\sqrt{a^2+b^2}.
$$

Calcolare `A_true` nello stesso modo sul target futuro. Limitare `k` ai casi in cui sia `f_-` sia `f_+` rimangono strettamente tra zero e Nyquist.

Modello gerarchico robusto:

$$
d_i\sim Student\text{-}t_4(\mu_i,\sigma),
$$

$$
\mu_i=\beta_{config[i]}+u_{harmonic[i]}+u_{background[i]}+u_{seed[i]},
$$

$$
\beta_c\sim Normal(\bar\beta+\delta_O\widetilde O_c,\tau_{config}),
\qquad O_c=(P-S)/P.
$$

con


$$
u_g\sim Normal(0,\sigma_g),
\qquad g\in\{harmonic,background,seed\}.
$$

Centrare e scalare `O`, per esempio facendo corrispondere un'unità a 0.5 di overlap. Se esiste un solo checkpoint per configurazione, eliminare `u_seed` e dichiarare esplicitamente che l'inferenza è condizionata a quel checkpoint.

### 3.4 Prior principale consigliato

La scelta principale proposta è:

$$
\bar\beta,\delta_O\sim Student\text{-}t_4(0,0.5),
$$

$$
\tau_{config},\sigma,\sigma_{harmonic},\sigma_{background},\sigma_{seed}
\sim Half\text{-}Student\text{-}t_4(0,0.5).
$$

Motivazione:

- il centro zero non incorpora nel prior la conclusione che si vuole dimostrare;
- sulla scala log-ratio, `0.5` equivale a un fattore `exp(0.5)≈1.65` a una deviazione di scala: è debolmente informativo ma ammette attenuazioni forti;
- le code Student-t rendono il modello meno fragile rispetto a effetti estremi;
- le scale gerarchiche usano la stessa unità della risposta, a differenza dell'attuale prior ricavato dai watt.

Questo valore `0.5` è una **scelta metodologica calibrata sulla scala dell'effetto**, non un numero trovato in un articolo. Va dichiarato onestamente e preregistrato. Come prior più scettico usare `Student-t_4(0,0.25)`; come controllo più largo usare `Student-t_4(0,1)`. Non usare un prior solo negativo: renderebbe più facile confermare H1 incorporandola nel prior.

Se invece si mantiene la cosine similarity raw, è preferibile modellare un contrasto paired; in sensitivity si può usare `Normal(0,0.25)` sul contrasto. Una beta regression richiederebbe trasformare `(r+1)/2` e gestire esattamente gli endpoint, senza risolvere il problema più importante dell'endpoint poco legato al comportamento del modello.

### 3.5 Stima, controlli e reporting

- almeno 100 fasi indipendenti per condizione e almeno 3 seed del modello;
- 4 chain NUTS, 2000 warm-up e 2000 draw per chain, `target_accept=0.90--0.95`;
- `R-hat<1.01`, ESS bulk/tail >1000 e zero divergenze;
- prior predictive check prima di osservare i nuovi risultati;
- parameter-recovery con effetti simulati `0`, `log(0.8)`, `log(0.5)` e `log(0.1)`;
- posterior predictive check stratificato per frequenza, fase, `P`, `S` e seed;
- sensitivity alle scale `0.25, 0.5, 1.0`, alla likelihood normale vs Student-t e ai controlli locali `±0.125, ±0.25, ±0.5` cycles-per-stride;
- riportare mediana e 95% credible interval di `exp(beta)`, oltre a `P(beta<log(0.8)|D)`, cioè probabilità di almeno il 20% di attenuazione;
- ROPE per equivalenza pratica: `[-log(1.1), log(1.1)]`, da preregistrare;
- dati già visti nel notebook solo esplorativi; conferma su nuove fasi, nuovi seed e nuove frequenze.

Per frequenza ignota nel PV, separatamente dal test principale, confrontare in sensitivity

$$
f\sim LogUniform(f_s/T,f_s/2)
\quad\text{e}\quad
f\sim Uniform(f_s/T,f_s/2),
\qquad
\phi\sim Uniform(0,2\pi),
$$

modellando anche trend, ciclo diurno e rumore colorato. Non usare una classificazione deterministica `blind/safe` come unica analisi sul PV.

### 3.6 Testo Bayesian pronto da inserire

```latex
\paragraph{Bayesian analysis.}
Bayesian analysis will quantify, rather than assume, the local degradation
associated with patch phase locking. For each patch configuration and training
seed, signals at the exact stride-lock frequencies $f_k=kf_s/S$ will be paired
with signals at $f_k\pm0.25f_s/S$, using identical phase, amplitude, noise level,
and background. Letting $R$ denote forecast amplitude recovery, we define the
paired local contrast
$d=\log(R_k+0.01)-[\log(R_-+0.01)+\log(R_++0.01)]/2$.
We will model these contrasts with a hierarchical Student-$t_4$ likelihood and
configuration, harmonic, background, and training-seed effects. The population
phase-lock effect will receive the neutral weakly informative prior
$\bar\beta\sim Student\text{-}t_4(0,0.5)$; hierarchical and residual scales will
receive $Half\text{-}Student\text{-}t_4(0,0.5)$ priors. The prior is centred at
no local degradation because the literature provides no transferable
quantitative effect estimate. Frequencies are fixed by the synthetic design
and need no inferential prior; phases are independently sampled from
$Uniform(0,2\pi)$. We will report the posterior amplitude-recovery ratio, its
95\% credible interval, the probability of at least 20\% attenuation, prior and
posterior predictive checks, and sensitivity to prior scale and control-frequency
distance.
```

## 4. Paragrafo MDL: correzione e protocollo esatto

### 4.1 Problemi della versione corrente e del notebook

- `deliverable1.tex:171` dice `after each decoder block`, mentre `:173` include encoder, decoder e output: il protocollo è contraddittorio.
- Il decoder Bolt ha un solo stato per layer; non esistono patch decoder da aggregare.
- L'output è una matrice di quantili `9×64`, non una probability distribution normalizzata.
- Nel notebook `chronosBolt_layer_probing.ipynb`, l'hook su `model.patch` viene chiamato sul contesto e poi sulla mask: il secondo output sovrascrive il primo, quindi `cap["patched"]` non è affidabile come patch del segnale.
- Il notebook usa 30 coppie/60 esempi, appiattisce encoder di dimensionalità molto alta ma confronta decoder da 256 dimensioni e applica PCA variabile: il confronto tra layer non è equo.
- `SV=1-L/N` non è la compression ratio standard. [Voita e Titov (2020)](https://aclanthology.org/2020.emnlp-main.14/) propongono online/variational codelength e mostrano che MDL è più informativo della sola accuracy; [Hewitt e Liang (2019)](https://aclanthology.org/D19-1275/) motivano random-label control e selectivity.

### 4.2 Dove sondare Chronos-Bolt

| Punto | Tensore | Riduzione usata dal probe | Ruolo |
|---|---|---|---|
| A | Patch di valori + observation mask prima dell'embedding, dimensione `2P` | masked mean e deviazione standard, oppure probe separato sul raw input | Controllo input-side; non è una rappresentazione appresa. |
| B | Output dell'input patch embedding, `[B,N,256]` | masked mean dei soli patch token | Controllo dopo la proiezione locale. |
| C0--C3 | Output di ciascuno dei quattro encoder, `[B,N+1,256]` | vettore `[REG]` da 256; masked mean dei patch token come analisi secondaria | Localizza la perdita nell'encoder mantenendo 256 dimensioni. |
| C-final | Output dopo final layer norm dell'encoder | `[REG]`, 256 dimensioni | Rappresentazione consegnata al decoder. |
| D0--D3 | Output di ciascuno dei quattro decoder, `[B,1,256]` | unico stato, 256 dimensioni | Misura l'accessibilità dopo cross-attention. |
| D-final | Output dopo final layer norm del decoder | unico stato, 256 dimensioni | Input immediato alla head. |
| E | Forecast quantile, `[B,9,64]` | nessun probe principale | Endpoint comportamentale: WQL, MASE, amplitude/spectral recovery e calibrazione. |

Usare sempre il modello frozen in `eval()` e acquisire i tensori senza gradienti. Per il confronto principale non usare PCA: `[REG]` e decoder hanno già 256 dimensioni. Il masked mean dell'encoder resta un'analisi secondaria, non va mescolato al risultato `[REG]`.

### 4.3 Dataset, probe e metriche

Disegno suggerito:

- per ogni `S`, `f=(k+delta)f_s/S`, con `delta={0, ±1/8, ±1/4}`;
- almeno 500 background indipendenti per condizione `(k,delta)`;
- stessa distribuzione di fase `Uniform(0,2π)`, ampiezza e SNR `{-20,-10,0} dB` in lock e controlli;
- split per background/generator seed, non per singola finestra, così varianti matched non attraversano train/test;
- task primario bilanciato: tone present vs absent;
- task secondario con `K=8` bin di uguale ampiezza in frequenza normalizzata `f/f_s`, bilanciati e costruiti per includere lock e controlli adiacenti;
- probe identico per tutti i layer: `StandardScaler` fit soltanto sul prefisso disponibile, seguito da logistic regression multinomiale L2 con `C=1`, solver `lbfgs`, `max_iter=2000` e `tol=1e-6`; nessuna selezione di iperparametri sul blocco da codificare;
- cinque ordinamenti grouped indipendenti, con seed `13, 42, 73, 101, 137`.

Con `K` classi e confini prequential `2,4,8,16,32,64,100%`, la codelength è

$$
L_{online}=t_1\log_2K+
\sum_{i=1}^{m-1}\sum_{j=t_i+1}^{t_{i+1}}
-\log_2p_{\theta_i}(y_j\mid h_j).
$$

Riportare:

- `L_online` in bit;
- bit per label `L_online/N`;
- compression `C=N log_2(K)/L_online`, dove maggiore è meglio;
- accuracy e calibration error soltanto come metriche secondarie;
- 95% CI tramite bootstrap grouped per background;
- contrasto primario `Delta L_l = L_l(lock)-mean_delta!=0 L_l(delta)`, dove un valore positivo indica informazione meno facilmente estraibile al lock.

Controlli obbligatori: label permutate, Chronos-Bolt random-initialized, raw/FFT feature upper bound, patch-origin jitter che cambia l'origine **tra patch** e non solo l'intera finestra, stesso probe e stessi dati a ogni layer. `Extractable` non significa `causally used`: il forecast deve corroborare il probing.

### 4.4 Testo MDL pronto da inserire

```latex
\paragraph{Layer-wise MDL probing.}
For every frozen checkpoint, frequency information will be measured through
prequential Minimum Description Length (MDL) probing. For each stride $S$,
controlled paired contexts will be generated both without a tone ($y=0$) and
with an injected tone ($y=1$). Tone frequency will be parameterised as
$f=(k+\delta)f_s/S$, where $\delta\in\{0,\pm1/8,\pm1/4\}$; $\delta=0$
identifies the predicted stride-lock condition and non-zero values provide
matched local controls. We will use at least 500 independent background pairs
per $(k,\delta)$ condition, uniformly random phase, and matched SNRs of $-20$,
$-10$, and $0$ dB. A secondary balanced probe will predict the frequency bin.

Representations will be captured (i) before and after the 256-dimensional input
patch embedding; (ii) from the contextualised $[\mathrm{REG}]$ vector after
each of the four encoder blocks and the encoder final layer normalisation; and
(iii) from the single 256-dimensional state after each of the four decoder
blocks and the decoder final layer normalisation. The $9\times64$ quantile
forecast will be evaluated as a behavioural endpoint rather than described as
a token-level probability distribution.

For a $K$-class task, background-grouped examples will be divided at $2$, $4$,
$8$, $16$, $32$, $64$, and $100\%$ of the data. At every boundary an identical
L2-regularised linear probe, with preprocessing fitted only on the available
prefix, will encode the next block. We will report online codelength in bits,
bits per label, and compression $C=N\log_2K/L_{online}$, averaged over five
grouped orderings with grouped-bootstrap 95\% confidence intervals. Shuffled
labels, a randomly initialised model, patch-origin jitter, and raw/FFT features
will serve as controls. A positive
$\Delta L_l=L_l(\delta=0)-mean_{\delta\ne0}L_l(\delta)$ indicates that more data
are required to decode the label specifically at the predicted locking
condition. The first supported increase localises where information becomes
less accessible; no increase before the forecast head will fail to support the
proposed representation-bottleneck hypothesis.
```

## 5. Training procedure

### 5.1 Valutazione della procedura nella repo

Aspetti corretti:

- usa `training_corpus_tsmixup_10m` e `training_corpus_kernel_synth_1m` in rapporto 9:1;
- mantiene context 2048, horizon 64 e i nove quantili di Bolt;
- applica lo stesso data diet alle configurazioni.

Problemi:

1. `train_sweep.py` usa oggi `MAX_STEPS=100_000`, mentre il commento e `README_retraining.md` parlano di 10k; lo shuffle buffer è oggi 10k mentre la documentazione ne descrive 100k.
2. Viene usato un solo seed (`42`).
3. Non esistono validation loop, checkpoint selection o early stopping; viene salvata soltanto la training loss.
4. Dire che lo stesso under-training `si cancella` tra architetture non è valido. Con context 2048, le configurazioni hanno circa 86 patch (`24/24`), 128 (`16/16`), 255 (`16/8`) o 509 (`16/4`); costo attention e velocità di convergenza cambiano molto.
5. Cambiare soltanto `S` modifica lo step di `unfold`, non la forma dei pesi: il retraining è una scelta di adattamento/fairness, non una necessità tecnica.
6. Cambiare `P` modifica l'input residual embedding da `2P` a 256. Si può riinizializzare solo quell'embedding e trasferire encoder, decoder e head; non è obbligatorio reinizializzare tutto.
7. La recipe pubblica nel repository Chronos riguarda il Chronos-T5 originale, con tokenizzazione discreta e cross-entropy, non una recipe completa di Bolt. La [model card Bolt](https://huggingface.co/amazon/chronos-bolt-tiny/blob/main/README.md) dichiara quasi 100 miliardi di osservazioni di training ma non pubblica tutti gli iperparametri. La procedura locale non va chiamata `faithful Bolt reproduction`.
8. La tabella corrente non isola lo stride sopra il default: `(24,24)` cambia contemporaneamente `P` e `S`.

### 5.2 Confronto con modelli in letteratura

| Modello | Scala/uso | Lezione utile per questa repo |
|---|---|---|
| [Chronos](https://arxiv.org/abs/2403.07815) / recipe pubblica | Famiglia T5 per forecasting; mix real/synthetic | La recipe originale usa un budget lungo, dataset multi-dominio, batch distribuito e una valutazione su molti dataset. È un proxy, non la recipe Bolt. |
| [Chronos-Bolt-Tiny](https://huggingface.co/amazon/chronos-bolt-tiny) | 9M, patch-based, direct quantile forecasting | Il checkpoint stock deve essere la baseline primaria; quasi 100B osservazioni rendono irrealistico dichiarare equivalente un run locale corto. |
| [Moirai](https://arxiv.org/abs/2402.02592) | Moirai-Small circa 14M; universal forecasting | Usa archive multi-dominio molto ampio e valida cross-frequency/generalizzazione; sostiene validation esplicita e data diversity. |
| [Lag-Llama](https://arxiv.org/abs/2310.08278) | Piccolo FM probabilistico univariato | Mostra l'importanza di validation e fine-tuning controllato su dati target, invece di giudicare dal training loss. |
| [Tiny Time Mixers](https://arxiv.org/abs/2401.03955) | circa 1M; zero/few-shot; pretraining multi-dominio | Pretraining su dati reali, split temporali, scelta degli iperparametri su validation e fine-tuning leggero. Il paper usa 20 epoche e 4--8 ore su 6 A100 per i propri modelli, ma quei tempi/parametri non sono direttamente trasferibili a Bolt. |

Non bisogna copiare un singolo set di iperparametri da un'architettura diversa. La lezione comune è: dati diversi, validation separata, più seed, budget dichiarato e selezione del checkpoint prima del test.

### 5.3 Procedura consigliata, separata in tre esperimenti

#### Esperimento A — intervento causale frozen, principale per H1/H2

- checkpoint ufficiale `amazon/chronos-bolt-tiny`, completamente frozen;
- `P=16`, `S={8,12,16,20,24}`;
- context sia 512 sia 2048 osservazioni; horizon 64;
- identiche realizzazioni sintetiche matched per ogni stride;
- almeno 100 fasi per frequenza, SNR `{-20,-10,0} dB`, più background clean e KernelSynth;
- nessun training.

Questo isola l'intervento sullo stride dai confondenti di inizializzazione e convergenza. `S=20,24>P` crea gap non osservati e va riportato come regime distinto, non come semplice riduzione dell'overlap.

Per isolare `P`, eseguire in un'analisi secondaria `S=16`, `P={8,16,24}`. Un 3×3 factorial `P,S in {8,16,24}` è preferibile se il budget lo consente, perché stima anche l'interazione.

#### Esperimento B — adattamento warm-start, confronto trained principale

Per lo sweep stride-only, tutti i modelli partono dal checkpoint stock; anche `S=16` viene adattato per lo stesso budget.

| Parametro | Valore proposto |
|---|---|
| Seed | `13, 42, 73` (5 seed se il budget lo consente) |
| Data diet | 90% TSMixup, 10% KernelSynth, manifest di finestre fisso e identico |
| Context / horizon / output | 2048 / 64 / quantili `0.1:0.1:0.9` |
| Optimizer | AdamW, `beta=(0.9,0.999)`, `eps=1e-8`, weight decay `0.01` |
| Learning rate | peak `1e-4`, 1000 step di warm-up, cosine decay a `1e-6` |
| Budget | massimo 20,000 optimizer step |
| Effective batch | 256 finestre: per esempio micro-batch 32 × accumulation 8 su 1 GPU; ridurre micro-batch e aumentare accumulation per stride costosi |
| Precision | FP32 con TF32, uguale per tutte le configurazioni |
| Gradient clipping | norma 1.0 |
| Validation | ogni 1000 step su almeno 10,000 finestre fisse, mai usate nel training |
| Early stopping | patience 5 validation, miglioramento relativo minimo 0.2%, non prima di 5000 step |
| Checkpoint | ogni 5000 step; selezione sul WQL medio di validation |
| Metriche | WQL primaria; MASE, coverage/width 80%, amplitude e spectral recovery, phase error secondarie |

Con dataset streaming, `epoch` non è ben definita. Riportare step e finestre viste. `20,000 × 256 = 5.12M` finestre, circa 0.47 passaggi nominali su 11M serie se ogni draw fornisce una finestra; l'equivalenza è solo descrittiva.

Per `P=8` e `P=24`:

1. caricare tutti i pesi compatibili;
2. riinizializzare soltanto `input_patch_embedding`;
3. 2000 step con backbone/head frozen, LR `1e-3`;
4. 30,000 step full-model, LR `1e-4`, warm-up 1500 step, cosine decay a `1e-6`;
5. stesso effective batch, data manifest, validation e seed del confronto stride.

#### Esperimento C — sanity check from-scratch della recipe

Prima di interpretare modelli completamente reinizializzati:

- un run baseline `P=S=16`;
- 200,000 step, effective batch 256;
- AdamW, LR `1e-3`, linear decay a zero, grad clip 1.0, nessun warm-up;
- 90/10 TSMixup/KernelSynth, TF32;
- eseguire una sensitivity `weight_decay={0,0.01}` perché la documentazione locale e le fonti pubbliche non sono perfettamente allineate;
- confrontare il miglior checkpoint con il modello stock su WQL, MASE e calibrazione.

Gate proposto: chiamare la recipe `reproduced` solo se WQL e MASE aggregate sono entro il 5% relativo dallo stock e la coverage dell'intervallo 80% entro 2 punti percentuali. Altrimenti i confronti from-scratch restano esplorativi e vanno separati dalla baseline ufficiale.

### 5.4 Dataset: cosa mantenere e cosa aggiungere

Il mix TSMixup/KernelSynth è una base migliore delle sole sinusoidi e non va sostituito. Tuttavia le sole sinusoidi del notebook non bastano per addestrare un forecaster generalista.

Protocollo consigliato:

- **discovery e confronto architetturale:** training soltanto sul mix generale 90/10; synthetic phase-lock tenuto fuori dal training e usato come test controllato;
- **validazione esterna:** PV interamente held-out, split cronologico con gap almeno pari a context+horizon fra train/validation/test; Bolt resta univariato sul target `Power_W`, mentre gli altri canali servono solo a stratificare/regolare l'analisi;
- **mitigazione separata, dopo aver confermato il problema:** 90% TSMixup, 5% KernelSynth, 5% curriculum spettrale con sinusoidi, chirp e multi-tone; `delta in [-0.25,0.25]` cycles-per-stride, fase uniforme, SNR da -20 a +10 dB, trend, missingness e rumore eteroschedastico.

Non addestrare il modello principale sui blind spot prima di misurarli: potrebbe eliminare il fenomeno che l'esperimento vuole caratterizzare.

### 5.5 Tempi

Non è possibile dare ore affidabili senza GPU, precisione e throughput misurato. Il tempo deve essere stimato con un pilot di 500 step **per ogni `(P,S)`**, dopo il warm-up:

$$
T_{hours}=\frac{N_{steps}\,t_{microbatch}\,N_{accum}}{3600}.
$$

Tabella di pianificazione per throughput effettivo misurato in optimizer step/s:

| Budget | 0.5 step/s | 1 step/s | 2 step/s | 5 step/s |
|---|---:|---:|---:|---:|
| 20k step | 11.1 h | 5.6 h | 2.8 h | 1.1 h |
| 30k step | 16.7 h | 8.3 h | 4.2 h | 1.7 h |
| 100k step | 55.6 h | 27.8 h | 13.9 h | 5.6 h |
| 200k step | 111.1 h | 55.6 h | 27.8 h | 11.1 h |

Moltiplicare per seed e configurazioni; i run possono essere parallelizzati. Non usare il throughput di `S=16` per `S=4`: passando da circa 128 a circa 509 patch, la parte self-attention può crescere approssimativamente con il quadrato della lunghezza, anche se il tempo end-to-end non cresce esattamente allo stesso modo.

### 5.6 Testo Training Procedure pronto da inserire

```latex
\paragraph{Training procedure.}
The official Chronos-Bolt checkpoint will first be evaluated frozen with
$P=16$ and $S\in\{8,12,16,20,24\}$, using identical inputs across strides.
This intervention does not require retraining because changing $S$ does not
change any parameter shape; values $S>P$ will be reported separately because
they leave gaps between patches. A trained comparison will then warm-start
every stride variant, including $S=16$, from the same official checkpoint and
adapt it for at most 20,000 optimizer steps on a fixed 90:10 mixture of TSMixup
and KernelSynth. We will use seeds 13, 42, and 73, an effective batch of 256,
AdamW ($\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, weight decay 0.01),
a peak learning rate of $10^{-4}$ after 1,000 warm-up steps, cosine decay to
$10^{-6}$, and gradient clipping at 1.0. Validation will be performed every
1,000 steps on 10,000 fixed held-out windows; the checkpoint with the lowest
mean WQL will be selected, with early stopping after five evaluations without
a 0.2\% relative improvement. MASE, 80\% interval coverage and width, amplitude
recovery, and spectral recovery will be secondary metrics.

Changing $P$ alters the $2P$-dimensional input embedding. For $P\in\{8,24\}$,
compatible encoder, decoder, and forecast-head weights will be transferred,
only the input embedding will be reinitialised and trained for 2,000 frozen-
backbone steps, and the full model will then be adapted for at most 30,000
steps under the same validation protocol. Because the public training recipe
is for original Chronos rather than Chronos-Bolt, a separate $P=S=16$
from-scratch run will be treated as a sanity check, not assumed to reproduce
the official checkpoint. The synthetic phase-lock benchmark and the PV test
period will remain excluded from model selection.
```

## 6. Ordine di intervento consigliato

1. Correggere la formulazione matematica e rinominare il fenomeno in `stride locking` finché non si dimostra una collisione appresa.
2. Correggere configurazione Bolt (`2048 observations`, token count, decoder singolo, quantile head).
3. Inserire H1--H3 e rendere coerenti griglia frequenze, `P`, `S` e codice `cps=fS/f_s`.
4. Sostituire la sezione MDL col protocollo esatto sopra e correggere gli hook del notebook prima di produrre risultati.
5. Rifare il disegno bayesiano bilanciato; non interpretare il posterior attuale con 0 casi blind.
6. Separare frozen intervention, warm-start adaptation e from-scratch sanity check.
7. Solo dopo, fare la pulizia grammaticale e comprimere il testo entro il limite di parole.

## Riferimenti principali da aggiungere alla bibliografia

- [Ansari et al., *Chronos: Learning the Language of Time Series*](https://arxiv.org/abs/2403.07815).
- [Amazon, configurazione ufficiale Chronos-Bolt-Tiny](https://huggingface.co/amazon/chronos-bolt-tiny/blob/main/config.json).
- [Amazon, codice ufficiale `chronos_bolt.py`](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py).
- [Pagani et al., *Preliminary Insights in Chronos Frequency Data Understanding and Reconstruction*](https://arxiv.org/abs/2605.06361).
- [Voita e Titov, *Information-Theoretic Probing with Minimum Description Length*](https://aclanthology.org/2020.emnlp-main.14/).
- [Hewitt e Liang, *Designing and Interpreting Probes with Control Tasks*](https://aclanthology.org/D19-1275/).
- [Gelman et al., *Bayesian Workflow*](https://arxiv.org/abs/2011.01808).
- [Gabry et al., *Visualization in Bayesian Workflow*](https://arxiv.org/abs/1709.01449).
- [Kallioinen et al., *Detecting and Diagnosing Prior and Likelihood Sensitivity with Power-Scaling*](https://arxiv.org/abs/2107.14054).
- [Vaughan, *A Bayesian Test for Periodic Signals in Red Noise*](https://arxiv.org/abs/0910.2706).
- [Woo et al., *Unified Training of Universal Time Series Forecasting Transformers* (Moirai)](https://arxiv.org/abs/2402.02592).
- [Rasul et al., *Lag-Llama*](https://arxiv.org/abs/2310.08278).
- [Ekambaram et al., *Tiny Time Mixers*](https://arxiv.org/abs/2401.03955).
