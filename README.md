# Pioneer

**A branchable conversational AI terminal for reasoning through decisions — without losing the paths you didn't take.**

Pioneer pairs conversational AI with Git-like conversation history and an explicit decision-analysis engine. Fork a conversation to explore an alternative, weigh acting now against waiting for more information, model what's reversible, track API spend, and verify that your local history hasn't been tampered with.

> Pioneer stores its own content-addressed history in `.pioneer/`. It does not depend on Git.

## Why Pioneer

Most AI chat tools give you one linear thread and no way to compare options side by side. Pioneer is built for decisions, not just conversation:

- **Fork instead of overwrite.** Explore "what if I waited" or "what if I optimized for downside instead" on a separate branch — your original reasoning stays intact.
- **No invented numbers.** Pioneer never fabricates probabilities, payoffs, or costs to force a calculation. If a number matters and you haven't supplied it, Pioneer asks instead of guessing.
- **Reversibility is explicit.** Undoing a decision often has its own cost and its own recovered value. Pioneer models that directly rather than folding it into a vague "risk" label.
- **Everything is checkable.** Every stored turn is SHA-256 addressed, so you can verify your history hasn't been corrupted or altered.

## Features

- **Branchable conversations** — fork, explore, and return to the original path later
- **Persistent working context** — goal, options, known facts, uncertainties, and provisional view, tracked automatically
- **Explicit decision analysis** — compare actions using states, probabilities, payoffs, costs, and reversible outcomes
- **Value of information** — weigh acting now against waiting for a signal
- **Reversibility modeling** — account for what's recoverable if an action is later undone
- **Optional Jev triage** — use TypeSafe System One to flag whether a decision is time-sensitive, hard to reverse, or missing key information
- **Content-addressed history** — SHA-256-hashed objects, checkable for corruption or tampering
- **Usage ledger** — track OpenAI/Jev token usage, with optional cost estimation
- **Local decision engine** — analyze structured decision cases with no API call at all

## Requirements

- Python **3.10+** for installation from source; the Windows executable includes Python
- An OpenAI API key (for `chat` and `ask`)
- Optionally, a TypeSafe API key (for Jev triage)

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

# Optional: enables Jev triage
export TYPESAFE_API_KEY="your-typesafe-api-key"

# Optional model overrides
export PIONEER_OPENAI_MODEL="your-openai-model"
export PIONEER_JEV_MODEL="your-jev-model"
```

`.env.example` is included as a reference, but Pioneer does **not** load `.env` files automatically — export these yourself or use your preferred environment manager.

## Windows executable

Download `Pioneer.exe` from the [latest release](https://github.com/Virgo-3/Pioneer/releases/latest) and double click it. The first launch creates a workspace under `%LOCALAPPDATA%\Pioneer\workspace`. Copy your OpenAI API key, return to Pioneer, and press Enter: it reads the key from the clipboard without displaying it. You can also choose `H` to type with hidden input, `V` to paste into a visible field, or `O` for offline decision tools. If you choose to save the key, Windows encrypts it for your user account under `%APPDATA%\Pioneer\openai-key.dpapi`. An existing `OPENAI_API_KEY` takes precedence. Python is not needed to run the executable.

Run `Pioneer.exe --help` in a terminal for the full CLI. With arguments, it behaves like the Python CLI and uses the current directory unless you pass `--repo PATH`. The optional Jev triage still uses `TYPESAFE_API_KEY` from your environment. The Windows build is made by [GitHub Actions](https://github.com/Virgo-3/Pioneer/actions/workflows/windows-exe.yml), which also provides a downloadable build artifact on every push.

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

## Interactive commands

Inside `pioneer chat`, type `/help` to see these:

| Command | Description |
| --- | --- |
| `/status` | Show the current branch and topic |
| `/context` | Show the saved working context |
| `/branches` | List conversations and their topics |
| `/branch NAME` | Copy the current conversation to a new branch |
| `/switch NAME` | Continue on another branch |
| `/log` | Show recent saved turns |
| `/analysis` | Show the latest complete decision calculation |
| `/usage` | Show recorded API token usage |
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

## Jev triage

With `TYPESAFE_API_KEY` set, Pioneer can use TypeSafe System One / Jev as an additional routing signal:

```bash
pioneer triage "Launch the product tomorrow"
```

Jev scores four questions — decision request, time sensitive, hard to reverse, missing information. These are judgments about how the decision is *described*, not outcome probabilities, and they are never inserted into numerical decision cases.

Conversation and decision analysis both work fine without Jev.

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

pioneer [--repo DIR] log [--ref REF] [--limit N]
pioneer [--repo DIR] show [REF]
pioneer [--repo DIR] status

pioneer [--repo DIR] decide FILE [--no-save]
pioneer [--repo DIR] triage TEXT...

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
│   ├── decision.py    # deterministic decision engine
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
