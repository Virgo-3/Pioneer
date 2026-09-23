# Pioneer

**A branchable conversational AI terminal for reasoning through decisions — without losing the paths you didn't take.**

Pioneer pairs conversational AI with Git-like conversation history and an explicit decision-analysis engine. Fork a conversation to explore an alternative, weigh acting now against waiting for more information, model what's reversible, track API spend, and verify that your local history hasn't been tampered with.

> Pioneer stores its own content-addressed history in `.pioneer/`. It does not depend on Git.

## Why Pioneer

Most AI chat tools give you one linear thread and no way to compare options side by side. Pioneer is built for decisions, not just conversation:

- **Fork instead of overwrite.** Explore "what if I waited" or "what if I optimized for downside instead" on a separate branch — your original reasoning stays intact.
- **Checked calculations.** Pioneer accepts a conversational calculation only when its numbers trace to the current decision conversation. If a number matters and you haven't supplied it, Pioneer asks instead of calculating from a guess.
- **Reversibility is explicit.** Undoing a decision often has its own cost and its own recovered value. Pioneer models that directly rather than folding it into a vague "risk" label.
- **Everything is checkable.** Every stored turn is SHA-256 addressed, so you can verify your history hasn't been corrupted or altered.

## Features

- **Branchable conversations** — fork, explore, and return to the original path later
- **Persistent working context** — goal, options, known facts, uncertainties, and provisional view, tracked automatically
- **History-aware challenge** — retrieve relevant older user statements from the active branch and surface a cited tension when a current plan materially conflicts with one
- **Explicit decision analysis** — compare actions using states, probabilities, payoffs, costs, and reversible outcomes
- **Value of information** — weigh acting now against waiting for a signal
- **Reversibility modeling** — account for what's recoverable if an action is later undone
- **Optional Jev decision attention** — use TypeSafe System One to help Pioneer focus on timing, reversibility, useful information, and tensions with earlier claims
- **Forecast feedback** — save specific yes/no forecasts with a date or resolution condition, record outcomes you report, compare the forecasts with those reports, and bring relevant history into later estimates
- **Content-addressed history** — SHA-256-hashed objects, checkable for corruption or tampering
- **Usage ledger** — track OpenAI/Jev token usage, with optional cost estimation
- **Local decision engine** — analyze structured decision cases with no API call at all

For conversation, Pioneer sends up to the latest 20 complete turns, capped at 24,000 characters of prior messages, plus its compact working context to OpenAI. It searches older user turns on the active branch and includes up to five relevant quotations in the same request. When asked to recall earlier history, it can also retrieve old turns without a topic keyword. If the model identifies a material conflict, Pioneer checks that the quoted text and commit really exist in the retrieved history before showing the challenge. Omitted or older quotations do not supply numbers to the decision engine. The search uses word overlap, so it can miss a relevant statement phrased differently; the model can also misjudge whether a change of view is a conflict.

## Requirements

- Python **3.10+** for installation from source; the Windows executable includes Python
- An OpenAI API key (for `chat` and `ask`)
- Optionally, a TypeSafe API key (for Jev decision attention)

Pioneer has no third-party Python package dependencies.

## Installation

```bash
git clone https://github.com/Virgo-3/Pioneer.git
cd Pioneer
python -m pip install -e .
```

This installs the `pioneer` command. You can also run it as a module:

```bash
python -m pioneer
```

## Configuration

Pioneer reads configuration from environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"

# Optional: enables Jev's decision attention inside conversation
export TYPESAFE_API_KEY="your-typesafe-api-key"

# Optional model overrides
export PIONEER_OPENAI_MODEL="your-openai-model"
export PIONEER_JEV_MODEL="your-jev-model"
```

`.env.example` is included as a reference, but Pioneer does **not** load `.env` files automatically — export these yourself or use your preferred environment manager.

## Windows executable

Download `Pioneer.exe` from the [latest release](https://github.com/Virgo-3/Pioneer/releases/latest) and double click it. The first launch creates a workspace under `%LOCALAPPDATA%\Pioneer\workspace`. Copy your OpenAI API key, return to Pioneer, and press Enter: it reads the key from the clipboard without displaying it. You can also paste an `sk-` key at the first prompt, choose `H` to type with hidden input, `V` to paste into a visible field, or `O` for offline decision tools. A directly pasted key may be visible in the terminal. If you choose to save the key, Windows encrypts it for your user account under `%APPDATA%\Pioneer\openai-key.dpapi`. Later `Pioneer.exe chat` and `Pioneer.exe ask` commands use that saved key too. An existing `OPENAI_API_KEY` takes precedence. Python is not needed to run the executable.

Run `Pioneer.exe --help` in a terminal for the full CLI. With arguments, it behaves like the Python CLI and uses the current directory unless you pass `--repo PATH`. Optional Jev decision attention uses `TYPESAFE_API_KEY` from your environment. The Windows build is made by [GitHub Actions](https://github.com/Virgo-3/Pioneer/actions/workflows/windows-exe.yml), which also provides a downloadable build artifact on every push.

## Python quick start

```bash
mkdir my-decision
cd my-decision
pioneer init
```

Start an interactive conversation:

```bash
pioneer chat
```

```text
Pioneer | Talk through a choice, or type /help for commands.
On main | new conversation

You: I'm deciding whether to launch now or run a smaller pilot first.
```

Every turn is saved into `.pioneer/`.

For a single non-interactive turn:

```bash
pioneer ask "Should I launch now or run a pilot first?"
```

Use a specific model for a session or a single request:

```bash
pioneer chat --model MODEL
pioneer ask "Help me think through this decision" --model MODEL
```

## Branching conversations

A Pioneer conversation is a history that can branch.

```bash
pioneer branch conservative-plan
pioneer switch conservative-plan
pioneer ask "Suppose I optimize for minimizing downside instead."
pioneer switch main
```

List branches:

```bash
pioneer branch
```

Pioneer can also branch automatically when a turn explicitly explores an alternative or counterfactual — the original conversation is left untouched.

### Rewinding

```bash
pioneer log
pioneer rewind <commit-id>
```

Before rewinding, Pioneer creates a rescue branch pointing at the previous head, so the displaced history stays accessible.

To start over on `main`, type `/reset main` in chat or run `pioneer reset main`. Pioneer moves the old `main` history to a named recovery branch and returns `main` to its starting point. API usage records remain in the ledger. Type `/clear` in chat to clear only the display and keep the conversation.

## Interactive commands

Inside `pioneer chat`, type `/help` to see these:

| Command | Description |
| --- | --- |
| `/status` | Show the current branch and topic |
| `/context` | Show the saved working context |
| `/branches` | List conversations and their topics |
| `/branch NAME` | Copy the current conversation to a new branch |
| `/switch NAME` | Continue on another branch |
| `/clear` | Clear the terminal display without changing history |
| `/reset [BRANCH]` | Restart `main` or a named branch; preserve old history on a recovery branch |
| `/log` | Show recent saved turns |
| `/analysis` | Show the latest complete decision calculation |
| `/usage` | Show recorded API token usage |
| `/forecast 70% \| EVENT \| DEADLINE [\| TOPIC]` | Save a forecast you supply |
| `/forecasts` | List unresolved forecasts on this branch |
| `/resolve ID yes\|no` | Report whether an event happened, or correct an earlier report |
| `/calibration [pioneer\|user\|all]` | Compare forecasts with reported outcomes |
| `/decide FILE` | Analyze a structured decision case |
| `/triage ACTION` | Ask Jev to assess an action |
| `/model MODEL` | Change the OpenAI model for this session |
| `/exit` | Leave the interactive terminal |

## Decision analysis

Pioneer's decision engine is deterministic: give it explicit quantities, and it will calculate rather than guess.

A case defines:

- mutually exclusive states and prior probabilities
- available actions
- the payoff of each action in each state
- optional action costs
- optional reversal values and reversal costs
- optionally, an informative signal and the cost of waiting for it

Example:

```json
{
  "title": "Ship now or pilot first",
  "units": "utility points",
  "states": {
    "strong_demand": 0.4,
    "weak_demand": 0.6
  },
  "actions": {
    "ship": {
      "cost": 2,
      "outcomes": {
        "strong_demand": { "payoff": 16 },
        "weak_demand": { "payoff": -10, "undo": -4, "undo_cost": 2 }
      }
    },
    "pilot": {
      "cost": 1,
      "outcomes": {
        "strong_demand": { "payoff": 7 },
        "weak_demand": { "payoff": -1 }
      }
    }
  },
  "wait": {
    "delay_cost": 1,
    "information_cost": 0.5,
    "signals": {
      "positive": { "strong_demand": 0.8, "weak_demand": 0.2 },
      "negative": { "strong_demand": 0.2, "weak_demand": 0.8 }
    }
  }
}
```

```bash
pioneer decide decision.json
```

Calculate without saving to the workspace:

```bash
pioneer decide decision.json --no-save
```

A full example lives at `examples/launch-decision.json`.

### Acting now vs. waiting

When a case includes a `wait` section, Pioneer computes both:

1. the expected value of the best action available now, and
2. the expected value of waiting for the signal and choosing afterward.

The output reports the **value of information**, waiting costs, the preferred action under each possible signal, and the maximum combined waiting cost before acting immediately wins out.

The calculation assumes the same actions remain available after waiting. It treats reversal as a choice after the true outcome is known. Cases with different timing or reversal rules need a different model. Misspelled or unsupported case fields are rejected rather than ignored.

### Reversible actions

An outcome can specify what remains after reversal:

```json
{ "payoff": -10, "undo": -3, "undo_cost": 1 }
```

Reversal is treated as an **option**, not an automatic penalty — Pioneer only uses it when reversing beats leaving the outcome in place.

All payoffs, costs, undo values, and delay costs must share the same units.

## Numerical guardrails

When Pioneer builds a decision case from conversation, it checks that every number traces back to something you actually said. It will not silently invent:

- state probabilities
- payoffs
- signal accuracy
- delay or information costs
- do-nothing values
- reversal values

If a calculation is missing a material assumption, Pioneer returns to the conversation to ask rather than calculating from a fabricated number.

## Optional Jev decision layer

Set `TYPESAFE_API_KEY` to let Pioneer consult Jev when its local routing detects a decision or a substantive follow-up to one. Ordinary chat does not require a Jev call. The routing is deliberately conservative, so it can miss an implicit decision; Pioneer still uses its normal conversational decision guidance in that case.

Jev sees the current message, recent user messages, the active branch's working decision context when the topic continues, a few relevant older user statements, and the previous Jev assessment for that goal. It answers narrow [typed yes/no questions](https://api.typesafe.ai/docs) about whether a choice is in play, whether timing and reversal matter, whether a missing fact could change the choice, and whether prior user statements suggest an assumption to check. Pioneer turns the results into at most two attention cues for OpenAI's single natural reply and saves the assessment with the turn. `/context` shows those decision checks without displaying raw scores.

Jev's values are judgments about the situation described, not outcome probabilities or action recommendations. They never enter the numerical decision case. If Jev is unavailable, Pioneer continues with OpenAI; conversation and local decision analysis also work without a Jev key.

The separate `triage` command remains available when you explicitly want to inspect Jev's four basic scores for an action:

```bash
pioneer triage "Launch the product tomorrow"
```

That manual diagnostic does not replace the conversational decision layer.

## Learning from forecasts

When you explicitly ask Pioneer for the probability of a specific yes/no event with a clear date or resolution condition, it can give a subjective estimate and save it as a forecast. For example: “What are the chances we launch by Friday?” Pioneer records its own forecast only when its visible reply states the same percentage. If the event or resolution condition is unclear, it should ask rather than record a vague prediction. Forecast probabilities are separate from the numbers you supply to the decision engine.

Later, tell Pioneer an unambiguous result such as “We launched Friday.” It can link that user-reported outcome to an open forecast when the event and its deadline are clear. An ambiguous report stays unscored until you clarify it. You can correct an earlier report conversationally (for example, “Actually, we did not launch by Friday”) or inspect and record it directly:

```text
/forecasts
/resolve FORECAST_ID yes
/calibration
```

You can also track your own estimate with `/forecast 70% | We launch | Friday | Launch timing`, or from a shell:

```bash
pioneer forecast 70% "We launch" --by Friday --topic "Launch timing"
pioneer forecasts
pioneer resolve FORECAST_ID yes
pioneer calibration --source user --topic "Launch timing"
```

The forecast ID is a saved commit ID; a unique short prefix is enough for `/resolve`. Each branch sees only forecasts and outcomes in its own ancestry. Recording the opposite result later corrects an outcome while preserving the earlier report.

Calibration reports show the number of resolved forecasts, average predicted probability, user-reported event rate, a Brier score, and reported rates across five confidence ranges. The Brier score compares predictions with those reports; lower is better, and zero means perfect agreement with the recorded results. Pioneer does not independently verify outcomes. After at least five resolved **Pioneer** forecasts on the same topic, Pioneer supplies that history as limited evidence during later turns on that decision. You can ask “How calibrated are you?” for the branch-wide record. Small or selected samples can mislead, so Pioneer does not silently rewrite a new estimate or your decision-case inputs. This is local feedback, not model retraining. Jev's attention scores are not outcome forecasts and are not calibrated by this report.

## History and storage

Each workspace has a `.pioneer/` directory:

```text
.pioneer/
├── HEAD
├── objects/
├── refs/
└── usage.jsonl
```

Turns, decisions, and notes are stored as immutable JSON objects. Each object's ID is the SHA-256 hash of its own contents, and branches are lightweight refs pointing at those objects — giving Pioneer Git-like properties without depending on Git internals:

```text
main
  │
  A ── B ── C
       │
       └── D ── E   explore-alternative
```

Different branches can hold different histories from the same starting point.

## Inspecting history

```bash
pioneer log                    # recent commits
pioneer log --ref alternative  # another branch
pioneer log --limit 5          # limit output

pioneer show <commit-id>       # full JSON for a commit
pioneer show                   # ...or just the current head

pioneer status                 # current position
```

## Integrity verification

```bash
pioneer verify
```

Checks stored object hashes, branch references, history links, and usage-ledger references. A successful run reports how many objects, branches, and usage entries were verified.

## API usage

```bash
pioneer usage
```

To estimate cost, supply a JSON price card (price per million tokens):

```json
{
  "my-openai-model": {
    "input_per_million": 1.0,
    "output_per_million": 4.0
  }
}
```

```bash
pioneer usage --prices prices.json
```

A template is at `examples/prices.example.json`.

## CLI reference

```text
pioneer [--repo DIR] init
pioneer [--repo DIR] chat [--model MODEL]
pioneer [--repo DIR] ask TEXT... [--model MODEL]

pioneer [--repo DIR] branch [NAME] [--from REF]
pioneer [--repo DIR] switch NAME
pioneer [--repo DIR] rewind REF
pioneer [--repo DIR] reset [BRANCH]

pioneer [--repo DIR] log [--ref REF] [--limit N]
pioneer [--repo DIR] show [REF]
pioneer [--repo DIR] status

pioneer [--repo DIR] decide FILE [--no-save]
pioneer [--repo DIR] triage TEXT...
pioneer [--repo DIR] forecast PROBABILITY EVENT... --by DEADLINE [--topic TOPIC]
pioneer [--repo DIR] forecasts
pioneer [--repo DIR] resolve ID yes|no
pioneer [--repo DIR] calibration [--topic TOPIC] [--source pioneer|user|all]

pioneer [--repo DIR] usage [--prices FILE]
pioneer [--repo DIR] verify
```

`--repo` operates on a workspace outside the current directory:

```bash
pioneer --repo ~/decisions/product-launch status
```

## Project structure

```text
Pioneer/
├── pioneer/
│   ├── cli.py         # CLI and interactive terminal
│   ├── calibration.py # forecasts, outcomes, and reliability scoring
│   ├── decision.py    # deterministic decision engine
│   ├── history.py     # bounded recall of earlier branch turns
│   ├── jev.py         # optional decision attention routing
│   ├── pipeline.py    # conversational turn pipeline
│   ├── providers.py   # OpenAI and Jev HTTP adapters
│   └── state.py       # content-addressed workspace storage
├── examples/
│   ├── launch-decision.json
│   └── prices.example.json
├── tests/
│   ├── test_cli.py
│   ├── test_pioneer.py
│   └── test_state_integrity.py
└── pyproject.toml
```

## Development

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

CI runs the test suite on Python 3.10 and Python 3.13.

## Design principles

**Conversation should be forkable.** Exploring an alternative shouldn't overwrite the reasoning that led to your current view.

**Calculations should be inspectable.** A numerical recommendation should come from an explicit case, not a hidden assumption.

**Uncertainty should stay visible.** Pioneer keeps user-supplied facts, unresolved assumptions, qualitative reasoning, and deterministic calculations distinct from one another.

**Waiting is an action.** When information might arrive later, the real comparison often isn't action A vs. action B — it's whether learning before acting is worth what it costs.

**Reversibility matters.** The ability to undo a decision changes its downside. That should be represented explicitly, not gestured at.

## License

No license file is currently included in this repository.
