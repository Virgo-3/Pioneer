# Pioneer

Pioneer is a conversational AI terminal with branchable local history, API token accounting, and a transparent decision engine. OpenAI handles conversation. TypeSafe AI's Jev can provide optional, narrow judgments about urgency, reversibility, and missing information. The act-versus-wait recommendation is calculated locally from assumptions you can inspect and edit.

## Quick start

Python 3.10 or newer is required. No runtime packages are needed.

```sh
python -m pioneer init
python -m pioneer chat
```

Set `OPENAI_API_KEY` in your environment to enable chat. Set `TYPESAFE_API_KEY` to enable Jev triage. Pioneer never saves keys in the workspace. You can select models with `PIONEER_OPENAI_MODEL` and `PIONEER_JEV_MODEL` (defaults: `gpt-6-astra` and `jev-latest`), or use `pioneer chat --model MODEL`.

You can also install the command locally with `python -m pip install -e .`, then use `pioneer` instead of `python -m pioneer`. Start it from the directory whose history you want to keep, or pass `--repo PATH` before a command.

## Conversation and branches

```sh
python -m pioneer ask "What are our options?"
python -m pioneer branch cautious
python -m pioneer switch cautious
python -m pioneer ask "Explore the smallest reversible experiment."
python -m pioneer log
python -m pioneer branch
python -m pioneer status
python -m pioneer verify
```

In `chat`, type a message normally. Slash commands include `/branch NAME`, `/branches`, `/switch NAME`, `/log`, `/status`, `/usage`, `/decide FILE`, `/triage ACTION`, `/model MODEL`, and `/exit`.

Each turn is an immutable, hash-verified object under `.pioneer/objects/`. Branches are pointers to those objects. `pioneer branch NAME --from REF` creates a branch at an existing branch or full commit ID. `pioneer rewind REF` moves the current branch and first creates a named rescue branch at its old head. Chat history sent to OpenAI comes from the active branch only. Decision records and triage notes stay in the history but do not become model chat messages.

`.pioneer/` is excluded from the project's Git repository by default because it may contain private conversation content. It can be backed up separately. The app's history is content-addressed, but it is not a replacement for Git version control of source files.

## Usage accounting

```sh
python -m pioneer usage
python -m pioneer usage --prices examples/prices.example.json
```

The append-only `.pioneer/usage.jsonl` records provider-returned input and output tokens once per successful call. Usage stays in the ledger after a rewind or branch change, so branch totals cannot make spent API calls disappear. If the branch changes while a request is in flight, Pioneer records its usage as orphaned and asks you to retry. Dollar cost is shown only when you supply a JSON price card with `input_per_million` and `output_per_million` for the exact returned model name. Update that card from your provider's current rates; estimates do not include discounts, taxes, or provider adjustments.

## Decide under uncertainty

```sh
python -m pioneer decide examples/launch-decision.json
python -m pioneer decide examples/launch-decision.json --no-save
python -m pioneer triage "Launch the new feature to all users tomorrow"
```

The decision case has mutually exclusive `states` with priors summing to 1. Each `action` gives a payoff in each state and may have an upfront `cost`. An outcome can include an `undo` payoff and `undo_cost`; Pioneer chooses reversal in that state only if its net payoff is better. Include a `hold` action if doing nothing should remain available.

Optional `wait.signals` describes `P(signal | state)` for each state, with likelihoods summing to 1 across signals. After each signal, Pioneer applies Bayes' rule and picks the best available action. Its value of information is the expected best payoff after the signal minus the best payoff now. The value of waiting subtracts `delay_cost` and `information_cost`. All payoffs and costs must use the same unit. If waiting wins, the engine recommends waiting; ties favor acting now. Its output includes the break-even total wait cost and the planned action for each signal.

The tool does not infer true state probabilities or execute a recommended action. Jev triage probabilities describe the proposed action text; they are **not** fed into the decision case as outcome probabilities. Use data, domain judgment, and sensitivity checks to set the priors, payoffs, and signal likelihoods. Saved analyses are commits on the current branch, so you can branch and compare changed assumptions without overwriting the earlier case.

## API contracts

- OpenAI conversation uses the [Responses API](https://developers.openai.com/api/docs/guides/text) with `store: false` and replays the active branch's user and assistant messages. Token counts come from the response's `usage` object.
- Jev triage uses TypeSafe's [System One API](https://docs.typesafe.ai/api) with three `noul` questions in one request. The result includes named probabilities and token usage. Jev is optional; local decision analysis works without either API key.

## Tests

```sh
python -m unittest discover -s tests -v
```

Tests use mocked provider responses, so they require no keys or paid API calls.
