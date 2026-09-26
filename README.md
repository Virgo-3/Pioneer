# Dao

**A branchable record for reviewing claims and decisions.**

Dao helps you open a question, collect evidence, and write a reasoned finding in ordinary language. Each case, evidence record, and finding is an immutable entry in the local branch history. A later finding can supersede an earlier one without erasing it. Dao does not require a formal logic language or a theorem prover.

You decide what a record means. A linked quote proves that particular words appear in a saved conversation turn; it does not prove that the words are true. Dao keeps first-hand records, source quotations, calculations, and conclusions distinct.

## Get started

Python 3.10 or newer is required when running from source:

```sh
python -m pip install -e .
dao init
dao case open "Was the pilot ready to launch?"
```

Use `dao --repo PATH` before a command to work in another directory. `dao chat` opens the interactive terminal; `dao ask "question"` runs one OpenAI conversation turn when `OPENAI_API_KEY` is set. The record commands work offline.

### Review a question from records

```sh
dao case open "Was the pilot ready to launch?"
dao case list
dao evidence CASE_ID "18 of 20 participants completed the pilot"
dao evidence CASE_ID "Two participants could not sign in"
dao case show CASE_ID
dao finding CASE_ID disputed "Completion was strong, but access failures remain" --support FIRST_EVIDENCE_ID --oppose SECOND_EVIDENCE_ID
dao case show CASE_ID
```

Use full IDs or unique prefixes of at least eight hexadecimal characters. `dao case show` prints them. A finding is `supported`, `refuted`, `disputed`, or `unresolved`. A supported finding needs supporting evidence, a refuted finding needs opposing evidence, and a disputed finding needs both. An unresolved finding can state what remains unknown.

To revise a conclusion, add another finding with `--supersedes CURRENT_FINDING_ID`. The previous finding stays visible in `dao case show`. Support and opposition refer to evidence on the active branch. A branch cannot cite evidence that only exists on another branch.

You can link evidence to exact words in a saved conversation turn:

```sh
dao evidence CASE_ID "The earlier reply claimed the pilot was ready" --source TURN_ID --role assistant --quote "the pilot was ready"
```

Dao checks that the cited turn is on the active branch and that the exact quote appears under the named speaker. Uncited evidence is labeled as a user record. Neither form receives an automatic truth verdict.

In `dao chat`, use `/case QUESTION`, `/cases`, `/review CASE_ID`, and `/evidence CASE_ID | TEXT`. The extended form `/evidence CASE_ID | TEXT | TURN_ID | user|assistant | EXACT_QUOTE` links a quote. Use `/finding CASE_ID | STATUS | REASON | SUPPORT_IDS | OPPOSE_IDS | SUPERSEDES_ID`; separate multiple IDs with commas and leave unused trailing fields out. `/help` lists the other controls.

## Conversation and branches

OpenAI writes the conversational answer. Dao stores it as written, then separately checks proposed goal, forecast, outcome, and numerical decision records. When a relevant earlier statement is found, a separate review may raise a source-linked challenge. The challenge is a question to examine, not an adjudicated finding. Use `/source TURN_ID` to inspect both sides of that saved turn, then add any relevant evidence and your own finding to a case.

```text
/branch pilot-first
/switch pilot-first
```

A branch copies the current history and gets its own later records. `/branches`, `/log`, and `/source ID` help inspect it. `/reset main` preserves the former head on a recovery branch. The CLI also has `dao branch`, `dao switch`, `dao log`, `dao show`, `dao rewind`, and `dao reset`.

Conversation normally uses one OpenAI call to write the reply and another to propose structured state updates. A relevant history review can make a third call. Token counts appear in `dao usage`. Dao never silently rewrites the authored answer after a check.

## Other decision tools

`dao decide examples/launch-decision.json --no-save` compares explicit actions, state probabilities, payoffs, reversal costs, and the option to wait for more information. Remove `--no-save` to keep the calculation in branch history. The calculation is an input to review, not a factual finding about the world.

`dao target`, `dao predict`, `dao observe`, and `dao outcomes` follow a desired outcome from goal through forecast to reported result. `dao forecast`, `dao resolve`, and `dao forecast-accuracy` score standalone predictions. These tools use user-entered outcomes; Dao does not independently establish that an event happened.

Set `TYPESAFE_API_KEY` to enable optional Jev attention signals during decision conversations. They concern timing, reversibility, missing information, and assumptions; they are not measured outcome probabilities. `dao triage "proposed action"` runs an explicit check.

## Windows executable

Download [Dao.exe from this repository's latest release](https://github.com/Virgo-3/Pioneer/releases/latest) and double-click it. Python is not needed for the executable. On first launch, Dao creates a workspace at `%LOCALAPPDATA%\Dao\workspace` and offers OpenAI key setup or offline tools. A saved key is protected for your Windows account. `Dao.exe --help` lists CLI commands.

Existing `%LOCALAPPDATA%\Pioneer\workspace` installations and `.pioneer/` workspaces remain readable in place. Dao creates `.dao/` for new workspaces. It also reads an existing saved key from the former Pioneer location and accepts the old `PIONEER_OPENAI_MODEL` and `PIONEER_JEV_MODEL` environment variables when the Dao equivalents are not set.

## Storage and verification

Dao stores content-addressed objects, branch pointers, and a usage ledger under `.dao/` in new workspaces. Run `dao verify` to check object hashes, refs, and ledger links. This detects storage corruption or altered records; it does not verify the truth of a claim. Branch history is local to the workspace and does not use Git internally.

To estimate API cost, copy `examples/prices.example.json`, supply your current prices, and run `dao usage --prices PATH`. The example prices are zero placeholders.

## Develop

```sh
python -m unittest discover -s tests -q
```

The tagged Windows release workflow runs the tests, builds `Dao.exe` with PyInstaller, smoke-tests startup, and publishes it on this repository's [Releases page](https://github.com/Virgo-3/Pioneer/releases). The application is named Dao; its GitHub repository remains `Virgo-3/Pioneer`.
