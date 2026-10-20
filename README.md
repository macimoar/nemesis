# Nemesis

Most chess engines calculate the best move. Nemesis calculates the move that beats **you**.

It studies your games and learns your playstyle: your openings, your appetite for material, how you attack, what you miss, how you play when the clock runs down. Then it plays the move most likely to make *you* go wrong. It also talks to you, in one of two commentary modes.

**X-Ray** gives the full breakdown. Nemesis never tells you which move it expects from you, so it doesn't bias your thinking. It reacts after you've moved.

> **Nemesis plays d6**
> - **My move:** d6 (pawn to d6).
> - **The position:** You're slightly better (about 0.6 pawns ahead for you).
> - **Why this move:** It's as good as the engine's top choice. Nf6 is 0.25 pawns better, too small to matter, and this one suits how you play.
> - **My chances:** About 72% against you, compared with 44% if you play perfectly.
> - **Hint:** I've set something up. Look carefully before you move.
>
> *(you play Bxf7+)*
> - **Gotcha:** That was the trap. I played d6 hoping you'd play bishop takes the pawn on f7 with check, because you grab material when offered and play checks reflexively. It costs you about 4.0 pawns.

**Poker Face** gives nothing away: no numbers, no moves, just in-character one-liners ("Go on, then." … "And there it is.").

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

### Windows

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

1. Download Stockfish from https://stockfishchess.org/download/. If you're not sure your CPU supports AVX2, take the AVX2 build. If Stockfish crashes on startup, switch to the "compatible" build.
2. Point Nemesis at the `.exe`. Nemesis also accepts the unzipped folder and strips quotes, but the plain path is safest:
   ```powershell
   setx STOCKFISH_PATH C:\tools\stockfish\stockfish-windows-x86-64-avx2.exe
   ```
   `setx` only affects **new** terminals, so open a new one and re-activate the venv.
3. Run `pytest -q`. All tests should pass. If Stockfish can't be found, the engine test is skipped and says why.

## Use

```bash
nemesis import lichess  <username> --max 300     # or: chesscom <username>, or: pgn <username> --file games.pgn
nemesis train <username>                         # Stockfish-analyses your moves and fits your model
nemesis profile <username>                       # what Nemesis learned about you
nemesis play <username> --color black --risk 0.8 # play in the terminal (--commentary xray|poker, --elo N)
nemesis games <username>                         # list your games
nemesis review <username> [--game ID]            # review a game (default: the latest)
nemesis ratings <username>                       # your rating and Nemesis's, per risk tier
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
| `Commentary` | `X-Ray` or `Poker Face`; sent as `info string` lines |
| `Elo` | Nemesis's strength; 0 = match me (your rating + 50) |
| `StockfishPath` | Path to the Stockfish binary |

The database lives at `~/.nemesis/nemesis.db`. Override it with `--db` or `NEMESIS_DB`.

## Strength and ratings

Nemesis is meant to be a challenge you can beat, not a wall.

- **Strength.** By default it plays at **your rating + 50** ("Match me"), so winning takes good play. You can also pick a fixed strength from 800 to 2400, where 2400 is full strength. Weaker targets don't play random moves: Nemesis still uses everything it knows about you, but samples its move instead of always taking the top one, and now and then makes a human-style slip. In X-Ray mode it admits it when a move isn't its best.
- **Your rating** starts from the rating recorded in your imported Chess.com/Lichess games. If none of them record one, it starts at 1200. After that, only games against Nemesis move it.
- **Nemesis's ratings** are tracked separately for each risk tier: Sound, Steering, Trapper and Hustler. Each tier starts at your rating + 50 and then earns its rating from results, so you can see which version of Nemesis actually beats you. Ratings marked `?` are provisional: they have fewer than 10 games behind them.
- **Calibration.** The strength curve is fitted with `scripts/calibrate_strength.py`, which plays Nemesis at a target Elo against Stockfish's own Elo-limited mode. Stockfish's scale is calibrated for engines and maps only roughly onto online ratings. Your results-based ratings correct for that over time.
  - **Current fit**, over 16 games per target at depth 10: at a 1400 target, Nemesis scored 62% against Stockfish at 1400, about 90 Elo strong. At a 2000 target, it scored 47% against Stockfish at 2000, about 20 Elo weak.
  - Stockfish can't play below 1320, so the 800–1100 settings are extrapolated.

## Game review

Any game can be reviewed: games played here and imported ones. In the web app, open the **Games** tab, or press **Review this game** when a game ends.

- **Accuracy** for both sides, using the Lichess formula.
- **Move labels**: Brilliant, Best, Excellent, Good, Inaccuracy, Mistake, Miss, Blunder. A move's label depends on how much it lowered the mover's winning chances.
- **Replay** on the board with ◀ ▶ (or the arrow keys), and a clickable evaluation graph.
- **Key moments**: the turning points in plain English, with the better move.
- **Nemesis's side**, for games against Nemesis:
  - the traps it set, and which ones you fell for;
  - how often it predicted your exact move;
  - the strength it played at.

  All of this is saved while the game is played, so retraining later doesn't change it.

## Does exploitation pay?

`scripts/simulate.py` generates games for synthetic players with known styles. `scripts/benchmark.py` then plays Nemesis (risk 100, full strength) and plain Stockfish at the same depth against the same player, using the same seeds. Results after training on 30 games of each:

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
  search.py     exploit search + risk dial
  strength.py   playing-strength limit (target Elo)
  commentary.py what Nemesis says (X-Ray / Poker Face)
  ratings.py    Elo for you and for each Nemesis risk tier
  review.py     game review: accuracy, labels, key moments, Nemesis's side
  engine.py     Nemesis = model + opening tree + search
  learn.py      training pipeline; recording games played against Nemesis
  uci.py, cli.py, web/
scripts/        simulate.py, benchmark.py, calibrate_strength.py
tests/
```

## Ideas for next steps

- Use a Maia network as the population prior for stronger move prediction.
- Add motif detectors: pins, skewers, back-rank weaknesses, discovered attacks.
- Model time usage, so Nemesis plays fast in positions where you burn clock.
- Add online play through the Lichess Bot API, so it can challenge you where you already play.
