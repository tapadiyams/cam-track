<!--
Authored by: Shubham Tapadiya
Created: 2026-09-02
Updated: 2026-09-02
-->
# Troubleshooting log

A running log of real issues hit while setting up and running this project
locally, and how they were resolved -- kept as I go so the debugging isn't
lost, and so it doubles as a record of the kind of problems a multi-service
pipeline like this actually produces in practice. New entries get appended
at the bottom as they come up; nothing here gets deleted once fixed, since
the point is to document what happened, not just the current state.

This file is for issues in the *application* (running the pipeline,
tests, dependencies). GitHub/`git push` issues live in `GIT.md` instead,
next to the publishing steps they belong to -- not duplicated here.

## Corrupted / wedged `.venv`

**Symptom:** `python3 -m venv .venv` left a partially-populated
`.venv/bin/` (missing `pip`), and later, `rm -rf .venv` (even with `sudo`)
hung indefinitely instead of completing.

**Cause:** the venv creation was interrupted partway through, and the
resulting directory ended up with a stale file handle/lock on macOS that
made deletion hang in uninterruptible I/O wait.

**Fix:** stop fighting the stuck directory -- leave it alone rather than
retrying deletes, and create a fresh venv under a different name instead
(`.venv2`). `scripts/setup.sh` now does this automatically: if `.venv`
exists but is missing `bin/activate` (a sign it's broken), it picks
`.venv2`, then `.venv3`, etc.

## zsh does not treat a trailing `#` as a comment interactively

**Symptom:** copy-pasting a command like
`cp .env.example .env  # adjust if needed; defaults work with the compose stack`
produced `cp: needed is not a directory` followed by the macOS `defaults`
command's usage text being printed.

**Cause:** unlike bash, an interactive zsh session does not treat `#` as a
comment marker by default (`INTERACTIVE_COMMENTS` is off). The whole line
is parsed literally, and `;` still splits it into two commands regardless.
Here that produced `cp .env.example .env # adjust if needed` (extra
"arguments" to `cp`) followed by `defaults work with the compose stack`
(the real macOS `defaults` binary, called with garbage arguments).

**Fix:** every copy-pasteable command block in `README.md` had its inline
`# comment` removed; explanations moved into the surrounding prose
instead. Rule of thumb: don't put `#` comments inline in commands meant to
be pasted into an interactive shell.

## Tests fail with `ModuleNotFoundError` after setup

**Symptom:** `pytest` failed collecting several test files with
`ModuleNotFoundError: No module named 'scipy'` / `'pydantic_settings'` /
`'cv2'`, even though `requirements.txt` lists all three.

**Cause:** the earlier `pip install -r requirements.txt -r
requirements-dev.txt` step didn't fully succeed (large packages like
`opencv-python-headless` are a common place for that to time out), so only
some packages actually landed in the venv.

**Fix:** re-run `pip install -r requirements.txt -r requirements-dev.txt`
and read the actual output for errors, rather than assuming it worked.

## `ModuleNotFoundError: No module named 'ultralytics'` when exporting a model

**Symptom:** `python3 scripts/export_onnx.py ...` failed even after
installing `requirements.txt` and `requirements-dev.txt`.

**Cause:** `ultralytics` (and `onnxruntime`, `confluent-kafka`) live in
`requirements-ml.txt`, deliberately kept separate from `requirements.txt`
-- production only needs the exported `.onnx` file, not the full PyTorch
training stack (see `docs/decisions/0001-detector-choice.md`).

**Fix:** `pip install ultralytics` (or `pip install -r
requirements-ml.txt` for everything, including Kafka support) before
running the export script.

**Related gotcha:** this error can also happen even after installing the
right package, if the virtualenv isn't actually active -- check the shell
prompt for the `(.venv2)` (or similar) prefix before assuming an install
didn't work. Running `pip install X` while the venv is inactive installs
`X` into some other Python entirely.

## `confluent-kafka` fails to build on macOS

**Cause:** `confluent-kafka` (in `requirements-ml.txt`, only needed if
`STREAM_BACKEND=kafka`) needs the `librdkafka` C library to compile, which
isn't there by default on macOS.

**Fix:** don't `pip install -r requirements-ml.txt` directly -- run
`./scripts/install_ml_deps.sh` instead. It checks for `librdkafka` first
and installs it via Homebrew (or apt on Linux) automatically before
running pip, so the confusing mid-build compiler error never happens in
the first place.

## `caplog.records[-1]` raises `IndexError` in a logging test

**Symptom:** `test_log_with_context_omits_unset_fields` failed with
`IndexError: list index out of range`.

**Cause:** `configure_logging()` sets `logger.propagate = False` on
purpose (so a service's own handler doesn't also fire through the root
logger and double-print every line). pytest's `caplog` fixture only
captures records that reach the *root* logger, regardless of the
`logger=` passed to `caplog.at_level(...)` -- so with propagation off,
`caplog.records` stayed empty.

**Fix:** the test now temporarily flips `logger.propagate = True` for the
duration of the assertion, then restores it, so it can observe records via
`caplog` without changing the production default.

## `redis.exceptions.ConnectionError: ... Connection refused` running a service directly

**Symptom:** `python -m src.ingestion.main` (or any other service run
directly, not via Docker Compose) immediately throws a Redis connection
error on `localhost:6379` in every ingestion thread.

**Cause:** the "running one service directly" path (README.md) only
starts the Python service itself -- unlike `./scripts/run_demo.sh`, it
does not also start Redis or TimescaleDB. With neither running, every
service that depends on them fails immediately.

**Fix:** start the infra pieces first, via Docker, then run the Python
service(s):

```bash
docker compose up -d redis timescaledb
```

Or use `./scripts/run_demo.sh` instead, which starts everything (infra
and services) together.

## `./scripts/run_demo.sh: line 17: docker: command not found`

**Cause:** Docker Desktop isn't installed (or isn't running -- on macOS,
being installed is not enough; the app itself has to be open for the
`docker` CLI and daemon to work).

**Fix:** `./scripts/run_demo.sh` now checks for this itself before doing
anything else -- it prints exactly what's wrong (not installed vs.
installed-but-not-running) and, if Homebrew is available, installs Docker
Desktop for you, rather than failing with a bare "command not found" three
steps into starting the stack. It also checks `.env`, the exported model,
and that every camera source file referenced in `configs/cameras.yaml`
actually exists, for the same reason: fail with a specific, actionable
message up front instead of a cryptic error mid-startup.

For GitHub `git push` errors (`403 Permission denied`, "Updates were
rejected (fetch first)", PAT scopes), see `GIT.md` -- that's where the
full publishing workflow and its troubleshooting live together.
