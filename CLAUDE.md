# Claude Code Instructions

Follow `AGENT.md` as your primary workflow. Every task defaults to the Primary Loop defined there.

## Iteration Workflow

When asked to do iterations (e.g. "do 25 iterations"):

1. Read `AGENT.md`, `TODO.md`, and `results.tsv` before starting.
2. Execute the Primary Loop from `AGENT.md` one iteration at a time.
3. Each iteration must be one commit with: test, implementation, verification, `TODO.md` update (if coverage changed), and `results.tsv` row.
4. Do not batch multiple iterations into one commit.
5. Do not skip verification steps.

## Verification

Run tests from the repo root:

```
python3 -m pytest tests/ -x -v          # Python-side tests
make test-<unit>                         # BSV unit tests
python3 scripts/test_cosim.py            # end-to-end co-sim
```

The tinygrad submodule is at `tinygrad/`. When importing tinygrad in tests, use `PYTHONPATH=tinygrad` or `sys.path.insert(0, "tinygrad")`.

## Submodule Commits

The tinygrad backend lives in the submodule at `tinygrad/tinygrad/runtime/ops_tinytpu.py`. To commit submodule changes:

1. Commit inside the submodule first: `cd tinygrad && git add ... && git commit ...`
2. Then commit the submodule pointer in the parent repo: `cd .. && git add tinygrad && git commit ...`

## Commit Messages

- Do not add `Co-Authored-By` footers.
- Use the format from `AGENT.md`: `subsystem: short description` (e.g. `tinytpu: lower elementwise add through vpu sequence`).

## Results Tracking

- Append one row to `results.tsv` per iteration (tab-separated, schema in `AGENT.md`).
- Use the short commit hash after the commit is created — amend or append in a follow-up.
