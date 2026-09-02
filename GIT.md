<!--
Authored by: Shubham Tapadiya
Created: 2026-09-02
Updated: 2026-09-02
-->
# Publishing this repo to GitHub

Steps to push this project to `https://github.com/tapadiyams/cam-track`.

## 1. Create the repo on GitHub

Go to GitHub -> New repository -> name it `cam-track`. Leave "Add a
README", ".gitignore", and "license" **unchecked** -- this project already
ships those, and an auto-created README would conflict with the local one
on the first push.

## 2. Initialize and push from your project folder

Run these from the root of your local `cam-track` project folder (the one
containing this file):

```bash
git init
git add .
git commit -m "Initial commit: cam-track pipeline scaffold"
git branch -M main
git remote add origin https://github.com/tapadiyams/cam-track.git
git push -u origin main
```

## 3. Before you push, double-check what's staged

```bash
git status
```

`.gitignore` already excludes `.venv*/`, `.env`, `data/`, model weights
(`models/*.onnx`, `models/*.pt`), sample videos, and `.DS_Store` -- but
since this is a **public** repo, skim the output anyway for anything that
shouldn't be there: local absolute paths, IDE config (`.idea/`,
`.vscode/`), credentials, or large binary files that slipped past the
ignore rules.

## 4. Authentication

- **HTTPS** (the URL used above): GitHub no longer accepts your account
  password for `git push` over HTTPS. Use a Personal Access Token instead,
  entered in place of the password when prompted.
- **SSH** (no password prompt once set up): use
  `git@github.com:tapadiyams/cam-track.git` as the remote instead, provided
  you already have an SSH key added to your GitHub account
  (Settings -> SSH and GPG keys).

### Creating a Personal Access Token (classic)

GitHub -> Settings -> Developer settings -> Personal access tokens ->
Tokens (classic) -> Generate new token (classic). Scopes needed for this
project:

- **`repo`** (the top-level box) -- required for `git push` to any repo,
  public or private.
- **`workflow`** -- required specifically because this project has
  `.github/workflows/ci.yml`; GitHub rejects a push touching files under
  `.github/workflows/` unless this scope is granted, even with `repo`
  checked.

Leave every other scope unchecked -- none of them (packages, org admin,
gist, notifications, etc.) are needed just to push this project. Set an
expiration, click "Generate token", and copy it immediately -- it's shown
only once. Use it as the password when Git prompts for one; the username
is your GitHub username.

## 5. "Updates were rejected (fetch first)" on the first push

If `git push -u origin main` fails with something like:

```
! [rejected]        main -> main (fetch first)
error: failed to push some refs to '...'
hint: Updates were rejected because the remote contains work that you do
hint: not have locally.
```

it means the remote already has a commit your local repo doesn't --
almost always because "Add a README" (or `.gitignore`/license) got
checked when creating the repo on GitHub, despite step 1 above. Check
what's actually there first:

```bash
git fetch origin
git log origin/main --oneline
```

If it's just a single auto-generated commit (e.g. a bare README), pick
one:

**Option A -- integrate it, then push (safe default):**

```bash
git pull origin main --allow-unrelated-histories --no-rebase
```

This may open an editor for a merge commit message -- save and close it
(in vim: `Esc` then `:wq` and Enter). If `README.md` conflicts, open it,
resolve the content, then:

```bash
git add README.md
git commit
git push -u origin main
```

**Option B -- your local repo is authoritative, remote content is
disposable:**

```bash
git push -u origin main --force
```

Only use `--force` if you're certain nothing on the remote is worth
keeping -- it overwrites the remote branch with your local history.

## 6. Updating the repo later

```bash
git add .
git commit -m "Describe what changed"
git push
```
