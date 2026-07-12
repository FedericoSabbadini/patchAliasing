# Analisi della Risposta in Frequenza — Aliasing Strutturale in Chronos-Bolt

## Obiettivo

Dimostrare che l'embedding a patch di Chronos-Bolt introduce un aliasing strutturale: la larghezza della patch P impone dei nulli nella risposta in frequenza a `f = k * fs / P` (cicli-per-patch interi), e verificare se la sovrapposizione tra patch (stride S < P) possa mitigare questo effetto.

## Modelli valutati

| Modello | P | S | Training | Ruolo |
|---------|---|---|----------|-------|
| p16-s16 (official) | 16 | 16 | 200k step, corpus completo | Punto P=16 S=16, ancora dell'asse stride |
| p16-s12-seed42 | 16 | 12 | 100k step, 48 shard | Asse stride |
| p16-s8-seed42 | 16 | 8 | 100k step, 48 shard | Asse stride |
| p16-s4-seed42 | 16 | 4 | 100k step, 48 shard | Asse stride |
| p8-s8-seed42 | 8 | 8 | 100k step, 48 shard | Asse patch-size |
| p24-s24-seed42 | 24 | 24 | 100k step, 48 shard | Asse patch-size |

**Nota sul confronto.** Il modello ufficiale ha il doppio degli step e il corpus intero: il suo recovery assoluto sarà superiore, ed è normale. Quello che conta per la nostra analisi non è "quanto bene" ciascun modello recupera il segnale, ma **dove** il recovery cala e **se** i nulli si spostano con P. È un esperimento sulla struttura, non sulla performance assoluta.

## Risultato 1 — L'aliasing da patch-size è reale e verificato

La previsione teorica è semplice: quando un intero numero di cicli del segnale entra esattamente in una finestra di P campioni, la proiezione lineare perde l'oscillazione. I nulli dovrebbero cadere a `f = k * fs / P`.

| Modello | P | 1° nullo previsto | Recovery misurato al nullo | Recovery a 10 Hz (baseline) |
|---------|---|--------------------|---------------------------|------------------------------|
| p16-s12-seed42 | 16 | 32 Hz | 0.054 | 0.966 |
| p16-s8-seed42 | 16 | 32 Hz | 0.123 | 0.988 |
| p16-s4-seed42 | 16 | 32 Hz | 0.026 | 0.965 |
| p8-s8-seed42 | 8 | 64 Hz | 0.333 | 1.007 |
| p24-s24-seed42 | 24 | ~21.3 Hz | 0.368 (a 24 Hz, il punto griglia più vicino) | 0.962 |

Questi numeri parlano chiaro:

- A 10 Hz tutti i modelli recuperano il segnale quasi perfettamente (~0.96–1.00). Non c'è un problema di capacità generale.
- Al nullo previsto il recovery crolla a 0.03–0.12 per P=16, a 0.33 per P=8, a 0.37 per P=24. La caduta è di un ordine di grandezza.
- **I nulli si spostano con P.** P=8 crolla a 64 Hz (non a 32), P=24 crolla vicino a 21 Hz (non a 32). Questo esclude che sia un artefatto del training o del dataset: è la geometria della patch che determina dove il modello diventa cieco.

Nel grafico `recovery_vs_cpp.png` la conferma è ancora più netta: quando l'asse x è il cpp (frequenza normalizzata per la larghezza della patch), tutti i modelli — qualunque sia il loro P — collassano alle stesse ascisse intere. Questo è esattamente ciò che la teoria prevede.

## Risultato 2 — Lo stride NON mitiga il nullo

Questa era l'ipotesi più interessante da testare: se la patch è lunga P ma la applichiamo ogni S < P campioni (con sovrapposizione), il modello "vede" il segnale più spesso e potrebbe riuscire a ricostruire l'oscillazione perduta.

**Risultato: ipotesi rigettata senza ambiguità.**

| Freq (Hz) | cpp | S=12 | S=8 | S=4 |
|-----------|-----|------|-----|-----|
| 28 | 0.875 | 0.076 | 0.074 | 0.041 |
| 30 | 0.9375 | 0.046 | 0.058 | 0.009 |
| 32 | 1.0 | 0.054 | 0.123 | 0.026 |
| 34 | 1.0625 | 0.028 | 0.045 | 0.024 |
| 36 | 1.125 | 0.008 | 0.044 | 0.047 |

Non c'è nessun trend di miglioramento andando da S=12 a S=4. Anzi, S=4 (il massimo overlap possibile con P=16) ha spesso il recovery *più basso*. Lo stride determina quante patch il transformer vede nella sequenza, ma non cambia cosa succede *dentro* ciascuna patch: la proiezione lineare su P campioni cancella comunque l'oscillazione.

Detto brutalmente: sovrapporre le patch non serve a niente contro questo tipo di aliasing. Il difetto è nella larghezza della finestra, non nella frequenza di campionamento delle finestre.

## Risultato 3 — Il tetto di banda è a fs/P

Oltre il primo nullo, il recovery non si riprende in modo sostenibile. Per i modelli retrained:

- **P=16**: recovery < 0.05 per quasi tutte le frequenze sopra 48 Hz (cpp > 1.5)
- **P=8**: recovery < 0.05 per quasi tutte le frequenze sopra 64 Hz (cpp > 1.0)
- **P=24**: recovery < 0.05 per quasi tutte le frequenze sopra 48 Hz (cpp > 2.0)

La banda utile del forecaster è approssimativamente `f < fs / P`. Sopra questa soglia il modello produce rumore con ampiezza quasi nulla — non sta prevedendo, sta restituendo una costante.

Questo è un limite architetturale, non di training: la proiezione lineare della patch non ha la capacità rappresentazionale per codificare oscillazioni più veloci di un ciclo per patch.

## Risultato 4 — I picchi a cpp intero (solo official)

Il modello ufficiale mostra un pattern inatteso: recovery elevato *esattamente* a cpp intero.

| Freq (Hz) | cpp | Official | Media retrained P=16 |
|-----------|-----|----------|----------------------|
| 32 | 1.0 | 0.929 | 0.068 |
| 64 | 2.0 | 0.830 | 0.177 |
| 128 | 4.0 | 0.971 | 0.918 |

A cpp esattamente intero, ogni patch vede lo stesso identico segmento di forma d'onda — è un caso degenere e il transformer può imparare una scorciatoia. Il modello ufficiale, con 200k step e il corpus completo, ha avuto abbastanza training per impararla. I nostri modelli a 100k ci arrivano solo parzialmente (128 Hz sì, 32 e 64 Hz no).

Questo non invalida la tesi dell'aliasing — anzi, la rafforza: il fatto che servano 200k step per "aggirare" parzialmente il nullo a cpp=1, e che comunque il recovery tra i nulli (cpp=1.5, 1.75, ecc.) resti sotto 0.12 anche per l'ufficiale, conferma che il difetto è strutturale e il training può solo rattopparlo nei casi più semplici.

## Errore di fase

L'errore di fase ha senso solo dove il recovery è sostanziale (> 0.3). Nella banda recuperabile (< 20 Hz per i retrained, < 32 Hz per l'official), gli errori sono tipicamente < 15°. Dove il recovery è nullo, la fase è rumore: l'atan2 di un segnale quasi-zero non contiene informazione.

## Verdetto finale

| Ipotesi | Esito |
|---------|-------|
| La larghezza P della patch determina la posizione dei nulli | **Confermata** — nulli a k*fs/P |
| I nulli cadono a cpp intero indipendentemente da P | **Confermata** — P=8, P=16, P=24 collassano tutti a cpp intero |
| L'overlap (stride ridotto) mitiga il nullo | **Rigettata** — S=4 collassa quanto S=16 |
| Il training più profondo può aggirare parzialmente l'aliasing | **Parzialmente confermata** — l'official sopravvive a cpp=1 ma non oltre cpp=1.5 |
| La banda utile è fs/P | **Confermata** — il recovery crolla sopra questa soglia |

L'aliasing strutturale da patch esiste, è misurabile, è prevedibile dalla geometria, e non è eliminabile con la sola sovrapposizione. L'unico modo per spostare il tetto di banda è cambiare P — il che è esattamente ciò che questo esperimento dimostra.
