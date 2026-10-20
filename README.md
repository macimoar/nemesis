# Nemesis

Most chess engines calculate the best move. Nemesis calculates the move that beats **you**.

It studies your games and learns your playstyle: your openings, your appetite for material, how you attack, what you miss, how you play when the clock runs down. Then it plays the move most likely to make *you* go wrong. Every move comes with its reasoning:

> **a6**: 69% expected against you (44% against perfect play). Trap: I expect Bxa6 (54%) because you grab material when offered; it loses 4.9 pawns.

## How it works

1. **Opponent model.** A per-player conditional-logit model predicts P(move | position, you). Each legal move is described by about 30 features:
   - **objective quality:** centipawn loss vs Stockfish, top-3 or not, blunder
   - **style:** capture, trade, check, sacrifice, pawn storm, king attack, retreat, castling
   - **tactical blind spots:** hangs a piece, ignores a threat, walks into a fork
   - **your opening repertoire**
   - **context:** time pressure, endgame, ahead or behind

   Your weights are pulled toward a population prior, so a few games already give sensible predictions.

2. **Exploit search.** Nemesis's turns are max nodes. Your turns are *chance* nodes: it averages over the moves the model predicts you'll play, instead of assuming your best reply. Likely moves that Stockfish didn't rank get analysed explicitly, because that's where blunders hide. Leaves are Stockfish evals converted to expected score.

3. **Risk dial** (0–100):
   - **What it controls:** how unsound a move may be (15cp below best at 0, up to 300cp at 100), and how much "expected score against you" outweighs "expected score against perfect play".
   - **At 0** Nemesis plays sound moves but still picks the lines you handle badly. **At 100** it's a hustler.
   - **When it's already winning,** the window shrinks: it converts instead of gambling.

4. **Repertoire steering.** In the opening it heads for positions where you have historically scored badly.

5. **It keeps learning.** Every game you play against it, in the web app, the terminal or a UCI GUI, is saved. Retraining weights recent games, and games against Nemesis, more heavily.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
# plus Stockfish: apt install stockfish / brew install stockfish (or set STOCKFISH_PATH)
```

## Use

```bash
nemesis import lichess  <username> --max 300     # or: chesscom <username>, or: pgn <username> --file games.pgn
nemesis train <username>                         # Stockfish-analyses your moves and fits your model
nemesis profile <username>                       # what Nemesis learned about you
nemesis play <username> --color black --risk 0.8 # play in the terminal
nemesis serve                                    # web app at http://127.0.0.1:8000
nemesis uci                                      # UCI engine for Arena / Cute Chess / En Croissant
```

`train` prints how well the model predicts the moves in your newest games, which are held out. It reports top-1 and top-3 accuracy and compares them with a generic player model and with "you'll play Stockfish's move".

Once several players are imported, `nemesis train --population` fits the shared prior on all of them. Each personal model starts from that prior.

**UCI options:**

| Option | Meaning |
|---|---|
| `Player` | Whose model to use |
| `Risk` | 0–100 |
| `Depth` | Stockfish depth |
| `Deep` | Look two of your moves ahead (slower) |
| `Learn` | Save finished games for retraining |
| `StockfishPath` | Path to the Stockfish binary |

The database lives at `~/.nemesis/nemesis.db`. Override it with `--db` or `NEMESIS_DB`.

## Does exploitation pay?

`scripts/simulate.py` generates games for synthetic players with known styles. `scripts/benchmark.py` then plays Nemesis (risk 100) and plain Stockfish at the same depth against the same player, using the same seeds. Results after training on 30 games of each:

| Simulated opponent | Plain Stockfish (depth 8) | Nemesis (depth 8, risk 100) |
|---|---|---|
| **Grabber** (captures whenever it can) | 10/10 wins, 22.6 moves on average | 10/10 wins, **16.8 moves** on average |
| **Solid** (one of Stockfish's top 2 moves) | 9/10 wins (95%), 37.9 moves | **10/10 wins**, 33.9 moves |

The model learns the grabber's habit: its capture weight comes out at +3.9 above the prior, and it predicts 40% of held-out moves exactly vs 25% for "Stockfish's top move".

## Layout

```
nemesis/
  ingest/       lichess.py, chesscom.py, pgn.py
  analysis/     stockfish.py (MultiPV analysis + persistent eval cache)
  features.py   per-move feature vectors
  profile/      move_model.py (the opponent model), style.py (profile), repertoire.py (opening tree)
  search.py     exploit search + risk dial + explanations
  engine.py     Nemesis = model + opening tree + search
  learn.py      training pipeline; recording games played against Nemesis
  uci.py, cli.py, web/
scripts/        simulate.py, benchmark.py
tests/
```

## Ideas for next steps

- Use a Maia network as the population prior for stronger move prediction.
- Add motif detectors: pins, skewers, back-rank weaknesses, discovered attacks.
- Model time usage, so Nemesis plays fast in positions where you burn clock.
- Add online play through the Lichess Bot API, so it can challenge you where you already play.
