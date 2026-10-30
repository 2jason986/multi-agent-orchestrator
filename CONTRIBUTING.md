# Contributing

Thanks for taking a look. This project is small on purpose, so the contribution
process is short.

## Development setup

```bash
git clone https://github.com/2jason986/multi-agent-orchestrator.git
cd multi-agent-orchestrator
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Before you open a pull request

```bash
python -m unittest discover -s tests -t . -v
ruff check .
python -m orchestra.cli demo
```

All three must pass. The demo is included because a green test suite has never
proved that the pipeline still produces a sensible reply.

## What makes a good change

- Keep the runtime dependency-free. Anything that needs a third-party package
  belongs behind an optional extra and must not be imported at module scope.
- New agent roles need an output contract documented in `README.md` and a test
  that fails when the contract is violated.
- Bug fixes should come with a test that fails on the old code. This repository
  runs 51 tests in a few seconds; there is no excuse for an untested path.
- Changes to prompts are changes to behaviour. Describe what you measured, not
  what you felt.

## Commit messages

Conventional Commits, subject in the imperative mood:

```
feat(orchestrator): escalate when the revision budget is exhausted
fix(knowledge): keep CJK bigrams for single-character runs
docs(readme): document the mock provider
```

## Branch and review

- Branch from `main`, name it `feat/...`, `fix/...` or `docs/...`.
- `main` is protected: a pull request plus a green CI run is required.
- Squash merges keep the history linear.

## Reporting bugs

Open an issue with the bug template. Include the output of `python --version`,
the command you ran, and the trace JSON if the pipeline behaved unexpectedly.

## Security issues

Do not open a public issue. See [SECURITY.md](SECURITY.md).
