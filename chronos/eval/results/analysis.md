# Analisi della Risposta in Frequenza — Patch Aliasing in Chronos-Bolt

## Modelli valutati

| Modello | P | S | Training | Ruolo |
|---------|---|---|----------|-------|
| p16-s16 (official) | 16 | 16 | 200k step, corpus completo | Ancora S=16, riferimento patch-size |
| p16-s12-seed42 | 16 | 12 | 100k step, sottoinsieme 48 shard | Asse stride |
| p16-s8-seed42 | 16 | 8 | 100k step, sottoinsieme 48 shard | Asse stride |
| p16-s4-seed42 | 16 | 4 | 100k step, sottoinsieme 48 shard | Asse stride |
| p8-s8-seed42 | 8 | 8 | 100k step, sottoinsieme 48 shard | Asse patch-size |
| p24-s24-seed42 | 24 | 24 | 100k step, sottoinsieme 48 shard | Asse patch-size |

---

## Analisi critica dei plot

### Plot 1 — Recovery vs Frequency

Il plot mostra un divario enorme tra il modello ufficiale (curva blu) e tutti i modelli riallenati. Il problema e' evidente:

- **Il modello ufficiale** mantiene recovery 0.8–1.0 fino a circa 22 Hz, oscilla tra 0.3–0.9 nella banda 30–90 Hz con picchi isolati (32, 64, 72, 86 Hz), e crolla definitivamente solo sopra 90 Hz.
- **I modelli riallenati** (arancione, verde, rosso, viola, marrone) crollano gia' a 24–28 Hz. Da li' in poi sono essenzialmente morti: recovery < 0.05 per quasi tutta la banda 30–120 Hz, con l'unica eccezione del picco a 128 Hz e qualche blip sporadico (42–44 Hz per p16-s4 e p16-s8).

**Problema critico:** Il divario non e' un effetto dell'aliasing — e' un effetto della qualita' del training. I modelli riallenati sono cosi' deboli che il loro collasso in frequenza avviene PRIMA del primo null teorico (cpp=1, cioe' 32 Hz per P=16). Non raggiungono nemmeno il limite imposto dalla geometria del patch: sono limitati dalla qualita' dell'addestramento, non dall'aliasing.

Le bande d'incertezza (aree colorate) sono ampie alle basse frequenze per i modelli riallenati, il che indica instabilita' anche nella zona dove dovrebbero funzionare bene.

### Plot 2 — Recovery vs CPP (coordinate aliasing-native)

Questo e' il plot piu' importante: se il fenomeno fosse puramente patch-aliasing, tutte le curve dovrebbero collassare sulla stessa forma quando si passa in coordinate cpp (cycles per patch).

**Non collassano.** Il modello ufficiale traccia un inviluppo completamente diverso: mantiene recovery intermittente fino a cpp ≈ 2.5, con picchi a cpp = 1, 2, 4 (interi). I modelli riallenati invece crollano a zero gia' a cpp ≈ 0.8–0.9, cioe' prima ancora del primo null teorico a cpp = 1.

Questo significa che:
- Per i modelli riallenati, il fattore limitante NON e' la geometria del patch, ma la capacita' appresa dal modello. Non possiamo separare l'effetto dell'aliasing dall'effetto dell'undertraining.
- Il modello ufficiale (200k step, dati completi) ha imparato a compensare parzialmente i null a cpp intero — dimostrando che l'aliasing del patch non e' un limite fisico invalicabile, ma un fenomeno che puo' essere attenuato da un training sufficientemente forte.

### Plot 3 — Asse stride (P=16, S variabile)

Questo plot dovrebbe rispondere alla domanda: "ridurre lo stride (piu' sovrapposizione tra patch) riempie il null dell'aliasing?"

**Il confronto e' metodologicamente viziato.** Si confronta:
- Official: 200k step, corpus completo, S=16
- Riallenati: 100k step, 48 shard, S=12/8/4

La curva blu (official, S=16) domina le altre non perche' S=16 sia migliore, ma perche' il modello e' addestrato il doppio su dati 50 volte piu' abbondanti. Qualsiasi differenza tra la curva blu e le altre e' imputabile al training, NON allo stride.

**Cio' che SI puo' concludere:** tra i modelli riallenati (arancione, verde, rosso — tutti a 100k, stessi dati, stesso regime), le curve S=12, S=8, S=4 sono praticamente sovrapposte. Lo stride non fa differenza a parita' di training. Questo risultato e' valido e difendibile.

**Cio' che NON si puo' concludere:** che S=16 sia superiore a S=4. La superiorita' dell'official e' interamente attribuibile al training.

### Plot 4 — Errore di fase vs frequenza

Il plot dell'errore di fase e' largamente inutilizzabile. Dove la recovery e' < 0.1 (cioe' quasi ovunque sopra 24 Hz per i modelli riallenati), l'errore di fase e' l'arcotangente di rumore: i valori oscillano casualmente tra 60° e 170°, senza significato fisico.

Le uniche zone interpretabili sono:
- 8–22 Hz per i modelli riallenati: errore di fase < 15°, il che indica che dove il modello riesce a recuperare ampiezza, traccia correttamente anche la fase.
- 8–30 Hz per l'ufficiale: errore di fase < 30° in media, consistente con la sua recovery elevata.

**Questo plot non aggiunge informazione al di la' di confermare che dove c'e' recovery, c'e' anche coerenza di fase. Dovrebbe essere mostrato solo nella banda di recovery significativa (> 0.3), oppure eliminato.**

---

## Valutazione critica del retrain

### Cosa e' andato storto

I modelli riallenati a 100k step con lo script originale soffrono di almeno tre problemi:

1. **Dati insufficienti.** 48 shard (40 TSMixup + 8 KernelSynth) contro il corpus completo di Chronos. Il modello ufficiale vede ordini di grandezza piu' varieta' spettrale, il che spiega perche' generalizza su una banda di frequenze molto piu' ampia.

2. **Shuffle buffer troppo piccolo.** Lo script originale usava `shuffle_buffer_size=10000`. Con soli 48 shard, la diversita' all'interno di ciascun mini-batch era limitata. Il modello ha probabilmente memorizzato pattern locali senza imparare la struttura spettrale generale.

3. **Nessuna protezione dagli spike.** La loss mostrava spike enormi (`mean_last_100 ≈ 9725`), segno di mini-batch anomali che hanno destabilizzato il training. Lo script migliorato (con spike-skip) non e' mai stato usato per queste run.

4. **100k step restano la meta' di 200k.** Anche con dati e codice perfetti, 100k step sono il 50% del budget dell'ufficiale. La differenza di recovery suggerisce che la curva di apprendimento non aveva ancora raggiunto il plateau.

### Cosa funziona nonostante tutto

- **La banda bassa (2–22 Hz)** e' genuinamente funzionale: recovery 0.85–1.0, fase corretta. I modelli hanno imparato a fare forecast di serie temporali reali — non sono rumore.
- **L'asse stride tra riallenati** e' un confronto pulito a parita' di condizioni. Il risultato (stride irrilevante) e' robusto.
- **L'asse patch-size** mostra che i null si spostano con P: p8-s8 crolla in una posizione diversa da p24-s24. Ma la profondita' dei null e' contaminata dall'undertraining — non possiamo dire quanto del collasso sia aliasing e quanto sia debolezza del modello.

### Il picco a 128 Hz

Tutti i modelli, inclusi i riallenati, mostrano un picco di recovery a 128 Hz (cpp=4 per P=16, cpp=2 per P=8, cpp=6 per P=24). Questo e' sospetto: potrebbe essere un artefatto del fatto che 128 Hz = fs/4 = 512/4 crea esattamente 4 campioni per periodo. Con 512 campioni di contesto, il segnale contiene esattamente 256 periodi completi — una configurazione degenere che potrebbe favorire il modello in modo non generalizzabile.

---

## Verdetto finale

### Cosa si puo' affermare

| Affermazione | Solidita' | Note |
|-------------|-----------|------|
| Il patch embedding crea null in frequenza | Plausibile | Visibile ma confuso dall'undertraining |
| I null si spostano con P | **Confermata** | p8 vs p16 vs p24 crollano in punti diversi |
| Lo stride non mitiga il null | **Confermata** (tra riallenati) | Confronto pulito a parita' di training |
| Il limite di banda e' fs/P | **Non dimostrabile** | I modelli crollano PRIMA di fs/P per insufficienza di training |
| Training piu' profondo compensa parzialmente l'aliasing | Plausibile | L'ufficiale sopravvive a cpp=1, i riallenati no |

### Cosa manca per rendere i risultati difendibili

1. **Un retrain alla pari dell'ufficiale.** Servirebbero modelli addestrati a 200k step sul corpus completo (o almeno su un sottoinsieme molto piu' grande) con lo script migliorato (spike-skip, validazione, shuffle buffer 100k). Solo cosi' si potrebbe isolare l'effetto della geometria del patch dall'effetto del training.

2. **Un confronto equo per l'asse stride.** Attualmente l'official (S=16) non e' confrontabile con i riallenati (S=12/8/4) perche' il regime di training e' completamente diverso. Per dimostrare che lo stride e' irrilevante anche contro un modello forte, servirebbe un p16-s16 riallenato con lo stesso protocollo degli altri — oppure riallenare i modelli stride a 200k su dati completi.

3. **Il plot dell'errore di fase va ridisegnato** mostrando solo la banda dove la recovery supera una soglia significativa (es. > 0.3), altrimenti e' rumore visivo.

### In sintesi

I risultati attuali dimostrano che (a) i null si spostano con P e (b) lo stride non aiuta a parita' di training. Ma il divario tra official e riallenati e' cosi' grande che l'ipotesi dell'aliasing strutturale resta mescolata con l'effetto dell'undertraining. I modelli riallenati non sono abbastanza forti da raggiungere il limite imposto dalla geometria del patch — crollano prima per debolezza propria. Per separare i due effetti serve un retrain con piu' dati e piu' step.
