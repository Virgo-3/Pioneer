# Pioneer

Pioneer is a conversational AI terminal with branchable local history, API token accounting, and a transparent decision engine. Talk through a choice in ordinary language: Pioneer can offer a provisional view, ask a focused follow-up when it matters, or calculate act versus wait from explicit numbers. Each turn saves the conversation, working context, analysis, and provider usage together.

## Quick start

Python 3.10 or newer is required. No runtime packages are needed.

```sh
python -m pioneer init
python -m pioneer chat
```

Set `OPENAI_API_KEY` in your environment to enable chat. Set `TYPESAFE_API_KEY` to include Jev triage in each conversation turn. Pioneer never saves keys in the workspace. You can select models with `PIONEER_OPENAI_MODEL` and `PIONEER_JEV_MODEL` (defaults: `gpt-6-astra` and `jev-latest`), or use `pioneer chat --model MODEL`. A complete decision case pasted as JSON into chat runs locally, even without API keys.

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

In `chat`, type a message normally. Replies appear as a simple `You:` / `Pioneer:` exchange; `/status` and `/usage` show the saved details when you want them. Pioneer uses a branch-local working context to remember the goal, options, known facts, open questions, and provisional view. It should answer when it can, ask a pivotal question when the answer could change advice, and group closely related questions when that is easier. It does not demand a numerical matrix for a qualitative conversation. If you clearly ask to explore an alternative, such as “What if we launch anyway?”, it saves the turn on a new `explore-...` branch and enters it while preserving the original. Slash commands include `/branch NAME`, `/branches`, `/switch NAME`, `/log`, `/status`, `/context`, `/usage`, `/analysis`, `/decide FILE`, `/triage ACTION`, `/model MODEL`, and `/exit`.

For example, a discussion can start with “Should we launch or run a pilot?” Pioneer may explain why a pilot is easier to reverse and ask how long it would take. If you later provide probabilities, payoffs, and the quality and cost of a future signal, it can calculate the choice. Its reply is a short explanation; `/analysis` shows the full saved case and calculation. These are model-guided conversation choices, so the wording will vary.

Each turn is an immutable, hash-verified object under `.pioneer/objects/`. Branches are pointers to those objects. `pioneer branch NAME --from REF` creates a branch at an existing branch or full commit ID. `pioneer rewind REF` moves the current branch and first creates a named rescue branch at its old head. Chat sends the latest 20 turns from the active branch plus its compact working context to OpenAI, keeping long sessions bounded. If a numerical detail is older than those turns, restate it before asking for a calculation. Decision records and triage notes stay in the history but do not become model chat messages.

`.pioneer/` is excluded from the project's Git repository by default because it may contain private conversation content. It can be backed up separately. The app's history is content-addressed, but it is not a replacement for Git version control of source files.

## Usage accounting

```sh
python -m pioneer usage
python -m pioneer usage --prices examples/prices.example.json
```

The append-only `.pioneer/usage.jsonl` records provider-returned input and output tokens. A turn using both providers creates two ledger entries linked to one conversation commit. If an API returns usage but an unusable reply, Pioneer saves an incomplete-turn note with that usage. Usage stays in the ledger after a rewind or branch change, so branch totals cannot make spent API calls disappear. If the branch changes while a request is in flight, Pioneer records its usage as orphaned and asks you to retry. Dollar cost is shown only when you supply a JSON price card with `input_per_million` and `output_per_million` for the exact returned model name. Update that card from your provider's current rates; estimates do not include discounts, taxes, or provider adjustments.

## Decide under uncertainty

```sh
python -m pioneer decide examples/launch-decision.json
python -m pioneer decide examples/launch-decision.json --no-save
python -m pioneer triage "Launch the new feature to all users tomorrow"
```

The decision case has mutually exclusive `states` with priors summing to 1. Each `action` gives a payoff in each state and may have an upfront `cost`. An outcome can include an `undo` payoff and `undo_cost`; Pioneer chooses reversal in that state only if its net payoff is better. Include a `hold` action if doing nothing should remain available.

Optional `wait.signals` describes `P(signal | state)` for each state, with likelihoods summing to 1 across signals. After each signal, Pioneer applies Bayes' rule and picks the best available action. Its value of information is the expected best payoff after the signal minus the best payoff now. The value of waiting subtracts `delay_cost` and `information_cost`. All payoffs and costs must use the same unit. If waiting wins, the engine recommends waiting; ties favor acting now. Its output includes the break-even total wait cost and the planned action for each signal.

The tool does not infer true state probabilities or execute a recommended action. Jev triage probabilities describe the proposed action text and working context; they are **not** fed into the decision case as outcome probabilities. OpenAI can draft a case from the current decision conversation, but Pioneer calculates only if every numeric input appears in a recent user message and the case passes mathematical validation. It carries a request to compare waiting or reversal across follow-up turns, rather than dropping that part of the choice. Without complete numbers it can still discuss a provisional choice and its conditions. Review the saved assumptions with `/analysis`: matching numbers alone cannot prove the model assigned them to the right states or actions. Saved analyses are part of the conversation commit on the current branch, so you can compare changed assumptions without overwriting the earlier case.

## API contracts

- OpenAI conversation uses the [Responses API](https://developers.openai.com/api/docs/guides/text) with [structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs), `store: false`, and the active branch's user and assistant messages. Token counts come from the response's `usage` object.
- Jev triage uses TypeSafe's [System One API](https://docs.typesafe.ai/api) with four `noul` questions in one request. The result includes named probabilities and token usage. Jev is optional; local decision analysis works without either API key.

## Tests

```sh
python -m unittest discover -s tests -v
```

Tests use mocked provider responses, so they require no keys or paid API calls.
