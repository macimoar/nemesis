# Nemesis

Most chess engines calculate the best move. Nemesis calculates the move that beats **you**.

It studies your games and learns your playstyle: your openings, your appetite for material, how you attack, what you miss, how you play when the clock runs down. Then it plays the move most likely to make *you* go wrong. It also talks to you, in one of two commentary modes.

In every mode, the **opening is named as you play it** (from the Lichess opening list, by position, so transpositions are named correctly). **X-Ray** gives the full breakdown. Nemesis never tells you which move it expects from you, so it doesn't bias your thinking. It reacts after you've moved. When you make a mistake (a pawn or more), it tells you the better move and, when it's clear, what went wrong. X-Ray also shows the game's evaluation live: an **eval bar** beside the board, a **graph** of the game so far with your mistakes marked, and each of your moves tagged with how much it shifted the evaluation (−1.8, +0.2…). In untimed games (and in timed ones once they're over) you can press **Explore this mistake**: a side board shows how your move gets punished and what follows the better move, step by step, and you can try moves of your own from that position, with Stockfish answering.

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

## Quick start (no coding needed)

1. Go to the [**Releases**](../../releases) page and download the zip for your computer:
   - **Windows:** `Nemesis-Windows.zip`
   - **Mac** (M1/M2/M3 or later): `Nemesis-Mac-AppleSilicon.zip`
   - **Older Intel Mac:** `Nemesis-Mac-Intel.zip`
   - **Linux:** `Nemesis-Linux.zip`
2. Unzip it, then open **Nemesis**:
   - **Windows:** double-click `Nemesis.exe`. If you see "Windows protected your PC", click **More info → Run anyway**. This warning appears because the app isn't code-signed.
   - **Mac:** right-click `Nemesis`, choose **Open**, then **Open** again. You only need to do this the first time.
3. Your browser opens to Nemesis. Type your Chess.com or Lichess username, and it studies your games. This takes a few minutes the first time.
4. Play. Keep the small Nemesis window open while you play, and close it when you're done.

Everything stays on your computer, in a `.nemesis` folder in your home folder. Stockfish is included, so there's nothing else to install.

## How it works

1. **Opponent model.** A per-player conditional-logit model predicts P(move | position, you). Each legal move is described by about 30 features:
   - **objective quality:** centipawn loss vs Stockfish, top-3 or not, blunder
   - **style:** capture, trade, check, sacrifice, pawn storm, king attack, retreat, castling
   - **tactical blind spots:** hangs a piece, ignores a threat, walks into a fork, pin, skewer, discovered attack or back-rank mate
   - **your opening repertoire**
   - **how likely a human at your rating is to play it,** from [Maia-2](https://github.com/CSSLab/maia2)
   - **context:** time pressure, endgame, ahead or behind

   Your weights start from "a typical human at your rating" and adjust as your games come in, so predictions are good from your first game.

2. **Exploit search.** Nemesis's turns are max nodes. Your turns are *chance* nodes: it averages over the moves the model predicts you'll play, instead of assuming your best reply. Likely moves that Stockfish didn't rank get analysed explicitly, because that's where blunders hide. Leaves are Stockfish evals converted to expected score.

3. **Risk dial** (0–100):
   - **What it controls:** how unsound a move may be (15cp below best at 0, up to 300cp at 100), and how much "expected score against you" outweighs "expected score against perfect play".
   - **At 0** Nemesis plays sound moves but still picks the lines you handle badly. **At 100** it's a hustler.
   - **When it's already winning,** the window shrinks: it converts instead of gambling.

4. **Repertoire steering.** In the opening it heads for positions where you have historically scored badly.

5. **It keeps learning.** Every game you play against it, in the web app, the terminal or a UCI GUI, is saved. Retraining weights recent games, and games against Nemesis, more heavily.

## Running from source (for developers)

You need Python 3.10 or newer. Stockfish is downloaded for you on first run.

```bash
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
nemesis                       # starts the app and opens your browser
```

- **Stockfish:** if it isn't already installed, Nemesis downloads the right official build for your computer into `~/.nemesis/stockfish/` on first launch. It falls back to a compatible build if your CPU can't run the fast one. Run `nemesis install-stockfish` to do this up front. To use your own copy instead, set `STOCKFISH_PATH` to the executable.
- **Tests:** `pytest -q`.
- **Building the app:** `python -m nemesis.analysis.install --dest packaging/stockfish`, then `pyinstaller packaging/nemesis.spec`. The `Build app` GitHub workflow does this for Windows, macOS and Linux.

## Use

```bash
nemesis import lichess  <username> --max 300     # or: chesscom <username>, or: pgn <username> --file games.pgn
nemesis train <username>                         # Stockfish-analyses your moves and fits your model
nemesis profile <username>                       # what Nemesis learned about you
nemesis play <username> --color black --risk 0.8 # play in the terminal (--commentary xray|poker, --elo N)
nemesis games <username>                         # list your games
nemesis review <username> [--game ID]            # review a game (default: the latest)
nemesis ratings <username>                       # your rating and Nemesis's, per risk tier
nemesis                                          # the app: starts the server and opens your browser
nemesis serve                                    # just the server (8000, or the next free port)
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

## Clocks and "does it work on you?"

- **Time controls:** untimed, or bullet, blitz and rapid presets from 1 minute to 15 | 10. Nemesis plays on its own clock: when it's short on time it thinks less deeply, so it plays faster and a little weaker rather than ever losing on time. Running out of time loses, unless your opponent has nothing left to checkmate with, which makes it a draw. A game where nobody moved is aborted and isn't rated.
- **Ratings per time control**, like Chess.com: bullet, blitz, rapid, classical and untimed. Each starts from your imported rating for that time control, if your imported games include one. Ratings from before v0.2 carry over as your untimed rating. The database is backed up automatically before it's upgraded.
- **Does it work on you?** The Profile tab shows:
  - how often Nemesis guessed your exact move, per game;
  - how often you fell for its traps;
  - how that compares with your mistake rate on other moves.

  It's measured from what Nemesis predicted *during* each game, so retraining doesn't rewrite the history.

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

## Puzzles from your mistakes

Open the **Puzzles** tab to drill the positions where you went wrong in your own games.

- **Where they come from:** every position where your move cost at least 1.5 pawns and one move was clearly best (at least a pawn better than anything else). Nemesis finds them each time it studies your games. If you imported games before puzzles existed, press **Retrain** once.
- **Hardest first:** your move model estimates how likely *you* are to find each move, and the ones you'd most likely miss come first. Puzzles you miss come back after the new ones.
- **Three tries** per puzzle. A wrong move is shown, then taken back, with how many tries are left; the solution stays hidden until you solve it, run out of tries or press **Show solution**.
- **Two hints:** the first names the theme (a fork, a pin, a skewer, a discovered attack, a back-rank mate, a threat…) without giving the move; the second tells you what you played in the real game and why it was wrong.
- **After you answer**, Nemesis shows the solution and what follows on the board, and reminds you what you played in the real game and what it cost.

## Opening repair

The **Openings** tab finds your habits: moves in your first 12 that you've played in at least two games and that cost you at least half a pawn each time.

- **Worst openings first,** by your score in the games that reached each habit.
- **One spot per habit:** if a habit leads to more trouble later in the same line, only the first move is drilled, since the rest only happens because of it.
- **Play through your line:** the board replays your usual moves up to the habit, then you find a better move. Any move about as good as the best one counts.
- **Learned** means a good move three times in a row, at least an hour apart, so it tests memory rather than repetition. Spots you just got right rest in the meantime; ones you miss come straight back.

## Doppelgänger mode

Pick **Opponent → Doppelgänger of …** to play against a copy of anyone Nemesis has studied: a friend, a rival you're about to face, or yourself. Import their games first, like your own.

- **Their openings:** in positions they've reached before, it plays what they played, as often as they played it.
- **Their habits:** after that, it picks each move with the probability that player would, according to their move model. It grabs, trades, attacks and blunders about as they do. There's no search and no traps, and its slips aren't announced.
- **X-Ray** shows how likely they were to play each option; **Poker Face** keeps quiet.
- Practice games aren't rated, but Nemesis still learns from your moves in them.

## The human-move model (Maia-2)

[Maia-2](https://github.com/CSSLab/maia2) is a neural network from the University of Toronto, trained on millions of Lichess games to predict the move a human at a given rating will actually play (not the best move). Nemesis uses it as the starting point of your move model; your personal weights then learn what's different about you.

- **It downloads once** (about 43 MB per time control, rapid and blitz) the first time Nemesis starts, into `~/.nemesis/maia2/`, from this project's GitHub releases. Until it arrives, or if it can't, Nemesis works exactly as before.
- **It runs locally** with onnxruntime: nothing about your games leaves your computer.
- **Ratings:** Maia-2 thinks in Lichess ratings; Chess.com ratings are shifted onto that scale.
- **Openings:** Maia-2 wasn't trained on the first 10 half-moves of games, so there its predictions get a smaller, separate weight, and your own repertoire does most of the work.

**Does it help?** Measured on one real player's 74 Chess.com games (rated about 850 to 1150), training on the older games and predicting the 568 moves of the newest 19 games, which were never trained on:

| Predicting the exact move | top-1 | top-3 | log loss (lower is better) |
|---|---|---|---|
| Stockfish's best move | 39.6% | | |
| Before: generic starting model | 42.6% | 69.2% | 2.28 |
| Before: personal model | 46.1% | 73.1% | 1.83 |
| Maia-2 alone, at your rating | 48.6% | 75.2% | 1.65 |
| After: starting model with Maia-2 (no games yet) | 55.6% | 82.2% | 1.60 |
| After: personal model with Maia-2 | 55.6% | 82.4% | 1.41 |

Personalising on top of Maia-2 doesn't change the single top guess much here, but it makes the probabilities sharper (log loss 1.60 → 1.41), and the exploit search works with those probabilities, not just the top guess.

That's one player, so treat it as a strong first sign, not a guarantee (about ±2% from sample size alone). To measure it on your own games: `python scripts/evaluate_maia.py --player you your-games.pgn`.

### A profile per time control

Most people play differently in bullet than in daily games, but a separate model per time control only helps if it predicts those games better. So for each time control where you have at least 10 games, Nemesis fits a profile that starts from your main one, tests both on that time control's newest games, and uses the time-control profile only if it wins. The Profile tab's **By time control** section shows which profile is used for each, and how your play really differs between them (only differences bigger than chance, by a permutation test over your games).

On the test player above, rapid earned its own profile (log loss 1.39 → 1.37 on its newest games) and daily games didn't; the profile notes that they lose 0.99 pawns a move and blunder 11% of the time in rapid, against 0.61 and 5% in daily games.

Maia-2 is MIT-licensed (Copyright (c) 2024 CSSLab); the licence is downloaded alongside the model. `scripts/convert_maia2.py` and the "Maia-2 models" workflow convert the official models to ONNX and check the copies against the originals.

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
  analysis/     stockfish.py (MultiPV analysis + eval cache), install.py (automatic Stockfish download)
  features.py   per-move feature vectors
  profile/      move_model.py (the opponent model), style.py (profile), repertoire.py (opening tree)
  search.py     exploit search + risk dial
  strength.py   playing-strength limit (target Elo)
  commentary.py what Nemesis says (X-Ray / Poker Face)
  ratings.py    Elo for you and for each Nemesis risk tier
  review.py     game review: accuracy, labels, key moments, Nemesis's side
  puzzles.py    puzzles mined from your mistakes
  repairs.py    opening repair: your costly opening habits, drilled with spacing
  doppelganger.py  play like any imported player
  maia.py       Maia-2 human-move model: ONNX inference, download on first use
  openings.py   opening names by position (data/openings.json, from the Lichess list, CC0)
  profile/timecontrols.py  a profile per time control, used only where it predicts better
  engine.py     Nemesis = model + opening tree + search
  launcher.py   double-click entry: free port, reuse a running copy, open the browser
  learn.py      training pipeline; recording games played against Nemesis
  uci.py, cli.py, web/
scripts/        simulate.py, benchmark.py, calibrate_strength.py
packaging/      PyInstaller spec, smoke test, per-OS "READ ME FIRST" notes
tests/
```

## Roadmap

What comes next, and what would make it v1.0: see [ROADMAP.md](ROADMAP.md).
