# Nemesis roadmap

Nemesis learns how you play, then plays the move most likely to beat *you*.
This is the plan from here: first make sure that idea works on real players,
then turn what Nemesis knows about you into training, then make the model sharper.

Shipped so far (v0.1.x): personal move model, exploit search with a risk dial,
X-Ray / Poker Face commentary, beatable strength ("match me"), ratings per risk
tier, game review, and a download-and-run app for Windows, macOS and Linux.

## v0.2 — Make it real

Close the gaps between what's built and how you actually play. Done: v0.2.0 shipped
the clock and the "does it work on you?" stats, v0.2.1 and v0.2.2 the rest.

- ✅ **Time controls and a clock.** Bullet, blitz and rapid in the web app, with
  ratings kept per time control. Nemesis has its own clock and budgets its thinking
  so it never loses on time. Clock times are saved with each game, so training
  learns how you play under time pressure.
- ✅ **Board basics:** promotion picker, drag-and-drop moves, move sounds, draw
  offers (Nemesis decides from its chances against you), and PGN download.
- ✅ **Faster learning:** each game's features are cached, so after a game only the
  new one is computed. Retraining after a game takes about 1.5 seconds instead of
  growing with every game you've played.
- ✅ **"Does it work on you?"** The profile shows how often Nemesis guessed your
  exact move, game by game, and whether you fall for its traps more often than you
  make mistakes on other moves. This is the real test of the whole idea, and it
  decides the order of everything below.

## v0.3 — From nemesis to coach

Nemesis knows your weaknesses, so it can drill them.

- ✅ **Puzzles from your own mistakes.** Positions from your games where you missed a
  fork, ignored a threat or hung a piece, replayed as "find the move" exercises,
  hardest-for-you first.
- ✅ **Opening repair.** For the openings where you score worst, play through the
  lines where you usually go wrong until you stop going wrong.
- ✅ **Doppelgänger mode.** The model that predicts your moves can also play like any
  imported player. Import a friend's or rival's games and practise against their style.

## v0.4 — A sharper model

- ✅ **More tactical patterns:** pins, skewers, back-rank weaknesses and discovered
  attacks, alongside forks and hanging pieces, for finer-grained blind spots.
- **A stronger human-move base:** start from a Maia network (trained on millions of
  human games per rating level) instead of the generic prior, so predictions are
  good even for new players.
- **A profile per time control:** most people play differently in bullet and rapid.

## Later

- **Nemesis on Lichess** as a bot account, so it can challenge you where you play.
- **Update checks** inside the app.
- **Code signing** to remove the Windows/macOS security warnings (an ongoing
  certificate cost; worth it once other people use Nemesis).

## What makes it v1.0

v1.0 means the core promise is proven and the app is something you'd hand to a
friend without caveats. All of these, not a feature count:

1. **The idea is proven on real people, not just simulations.**
   - Across several real players (not only one), the personal model predicts their
     moves clearly better than both the generic model and "they'll play Stockfish's move".
   - Exploitation measurably helps: at the same strength, Nemesis with its risk
     dial on scores better against you than Nemesis with it off.
2. **"Match me" really is a fair fight.** With real human results, players win
   roughly 40–50% of games at the default setting, and ratings settle instead of
   drifting. The strength table below 1320 is fitted from those results, not extrapolated.
3. **Updating never loses your data.** The database has versioned migrations, so
   any new version opens an old `.nemesis` folder safely, and you can export/back up
   your games, profile and ratings.
4. **It installs and runs without warnings or workarounds.** Signed apps, built-in
   update checks, and no terminal knowledge needed for anything.
5. **The everyday experience is complete.** Clocks, a full-featured board, game
   review, and at least the first coaching feature (puzzles from your mistakes).
6. **Usable by everyone.** Playable by keyboard, works with a screen reader for
   moves and commentary, and move labels don't rely on colour alone.
7. **Clear about your data.** A short plain-English statement of what's stored,
   where (only on your computer), and what is ever sent anywhere (game downloads
   from Chess.com/Lichess, Stockfish download).

The order above is a plan, not a promise: the v0.2 "does it work on you?" stats
decide what comes next. If prediction on real games is weak, the v0.4 model work
moves ahead of coaching.
