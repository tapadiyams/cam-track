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
  password for `git push` over HTTPS. Use a Personal Access Token instead
  (GitHub -> Settings -> Developer settings -> Personal access tokens),
  entered in place of the password when prompted.
- **SSH** (no password prompt once set up): use
  `git@github.com:tapadiyams/cam-track.git` as the remote instead, provided
  you already have an SSH key added to your GitHub account
  (Settings -> SSH and GPG keys).

## 5. Updating the repo later

```bash
git add .
git commit -m "Describe what changed"
git push
```
