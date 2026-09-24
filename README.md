# Pioneer

**A conversational AI terminal with branchable history and explicit decision checks.**

OpenAI writes the conversational answer. Pioneer stores each turn on a branch, checks proposed records and calculations, and can compare the finished answer with earlier user or AI statements. It shows those checks separately. You can inspect the source and make the final judgment.

Pioneer also models the value of waiting for information, the cost of delay, and what can be recovered if an action is reversed. Its history lives in a local `.pioneer/` directory; it does not use Git internally.

## Get started

### Windows executable

Download [Pioneer.exe from the latest release](https://github.com/Virgo-3/Pioneer/releases/latest) and double click it. Python is not required. On first launch, Pioneer creates a workspace at `%LOCALAPPDATA%\Pioneer\workspace` and asks for an OpenAI API key. You can copy the key and press Enter to read it from the clipboard, choose `H` for hidden typing, or choose `O` for offline tools. If you choose to save the key, Windows encrypts it for your user account.

The executable also accepts CLI commands, for example `Pioneer.exe --help`. A saved key is available to its `chat` and `ask` commands; an `OPENAI_API_KEY` environment variable takes precedence.

### Python

Python 3.10 or newer is required when running from source. From this repository:

```sh
python -m pip install -e .
pioneer init
pioneer chat
```

Set `OPENAI_API_KEY` in your environment before `chat` or `ask`. The `.env.example` file lists supported variable names, but Pioneer does not load it automatically. To work in another directory, put `--repo PATH` before the command, such as `pioneer --repo my-workspace init`.

Once in chat, talk normally about a choice, a goal, or a result. Use `/help` for the everyday controls and `/help advanced` for exact record commands. To run one turn without entering chat:

```sh
pioneer ask "Should I launch now or run a pilot first?"
```

## What happens during a conversation

1. Pioneer sends OpenAI up to 20 complete recent turns, capped at 24,000 characters, along with relevant working context and saved decision records. A direct request to recall older history can also include bounded, speaker-labeled quotations from this branch.
2. OpenAI supplies an answer and proposed state updates. Pioneer displays and stores the answer **as OpenAI wrote it**. Pioneer checks any proposed target, reported outcome, forecast, or numerical decision case and prints the result as a separate `Pioneer check:` line. A rejected proposal is not saved, even if OpenAI's answer describes it as settled.
3. When a relevant earlier statement is found, a separate OpenAI review compares it with the *finished* answer. Pioneer shows an `OpenAI challenge:` only after confirming that its cited quote, speaker, and commit match this branch. `/source ID` displays both sides of the cited turn.

The review is an additional billable API call when it runs. Its usage appears in `/usage`. If it fails, the original answer remains available and Pioneer reports that the review was unavailable.

**A citation proves what was said, not what is true.** Retrieval uses bounded excerpts and word overlap, so it can miss a relevant claim; the reviewer can also misread a change of mind. Pioneer does not silently rewrite OpenAI's answer or decide whether you must accept a challenge.

## Explore alternatives without losing the original

In chat:

```text
/branch pilot-first
/switch pilot-first
```

`/branch` copies the current point in the conversation; `/switch` moves to it. You can return with `/switch main`. Each branch has its own later turns and records. `/branches` lists them, `/log` shows recent saved turns, and `/source ID` opens a cited turn on the active branch.

`/reset main` restarts `main` while saving its previous head on a recovery branch. The CLI also provides `pioneer branch`, `pioneer switch`, `pioneer log`, `pioneer show`, `pioneer rewind`, and `pioneer reset` for exact history control.

## Compare acting, waiting, and reversing

The decision engine accepts explicit states, their probabilities, actions, payoffs, and costs. An optional wait scenario supplies signal likelihoods, delay cost, and information cost. An action outcome can include the payoff after undoing it and the cost of undoing it. Pioneer compares the best move now with waiting and choosing again after a signal.

Try the included case without an API key:

```sh
pioneer decide examples/launch-decision.json --no-save
```

Remove `--no-save` to save the checked analysis in the current workspace. You can also discuss a numerical case in chat; Pioneer only saves a computed result when its inputs pass local validation and trace to the current decision conversation. OpenAI may still offer an opinion in its own words, which is shown separately from the checked calculation.

## Measure what happened

Pioneer keeps desired outcomes distinct from reported actual outcomes. In chat, you can say, for example, “I want at least 100 weekly active users by October 31,” and later report the measured result at that checkpoint. A `Pioneer check:` line confirms whether a target or result was saved. You can inspect or enter them directly:

```sh
pioneer target "Pilot" "weekly active users" --desired 100 --by "2026-10-31" --unit users
pioneer targets
pioneer observe TARGET_ID 92 --at "2026-10-31"
pioneer calibration
```

`TARGET_ID` can be a unique prefix from `pioneer targets`. Calibration compares the desired value with the user-reported actual value; Pioneer does not independently verify the report or infer why a gap occurred.

Probability forecasts use a separate record and score:

```sh
pioneer forecast 70% "Pilot reaches 100 users" --by "2026-10-31"
pioneer forecasts
pioneer resolve FORECAST_ID yes
pioneer forecast-accuracy
```

The forecast accuracy report scores saved probability estimates against user-reported outcomes. It is feedback on those estimates, not model retraining. Use `pioneer forecast-accuracy --source user` to inspect user-entered forecasts separately.

## Optional Jev attention

Set `TYPESAFE_API_KEY` to let Pioneer consult System-One Jev during decision conversations. Jev supplies narrow attention signals about timing, reversibility, missing information, and assumptions in recent user context. Those signals can inform the conversation; they are not outcome probabilities or commands to choose an action. Ordinary chat and the local decision engine work without Jev. `pioneer triage "your proposed action"` is an explicit Jev check.

## Usage, storage, and integrity

`pioneer usage` shows recorded OpenAI and Jev tokens by provider and model, including retrospective review calls. For an estimated cost, copy `examples/prices.example.json`, enter your actual per-million-token prices, and run `pioneer usage --prices PATH`. The example prices are zero placeholders.

Pioneer stores content-addressed objects, branch references, and a usage ledger under `.pioneer/` in the workspace. Run `pioneer verify` to check object hashes, references, and ledger links. This detects corruption or changes to recorded objects; it does not verify the truth of conversation content or user-reported results.

## Useful commands

| In chat | Purpose |
| --- | --- |
| `/help`, `/help advanced` | Show conversational and exact controls |
| `/branch NAME`, `/switch NAME`, `/branches` | Explore and navigate branches |
| `/log`, `/source ID` | Inspect history and a cited turn |
| `/clear`, `/reset [BRANCH]` | Clear the display or restart a branch with recovery history |
| `/usage` | Inspect API token usage |
| `/targets`, `/calibration` | Inspect desired and actual outcomes |
| `/forecasts`, `/forecast-accuracy` | Inspect and score forecasts |
| `/analysis`, `/context` | Inspect the last checked calculation or working context |

Run `pioneer --help` for the complete CLI reference.

## Develop

```sh
python -m unittest discover -s tests -q
```

The tagged Windows release workflow runs the tests, builds a one-file `Pioneer.exe` with PyInstaller, smoke tests startup, and publishes the executable on the [Releases page](https://github.com/Virgo-3/Pioneer/releases).
