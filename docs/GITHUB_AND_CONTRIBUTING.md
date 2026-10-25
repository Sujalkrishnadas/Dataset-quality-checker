# Publishing on GitHub, and contributing to other projects

## Part A: Publish your own repository

1. Create a free account at https://github.com.
2. Install Git for Windows (https://git-scm.com/download/win), then set your identity once:
   ```powershell
   git config --global user.name "Sujal Krishna Das"
   git config --global user.email "you@example.com"
   ```
3. Edit `pyproject.toml`, `LICENSE` and `README.md`: replace `Sujal Krishna Das` and `Sujalkrishnadas`.
4. Run the tests one last time: `pytest`
5. In the project folder:
   ```powershell
   git init -b main
   git add .
   git status        # check that .venv and reports contents are NOT listed
   git commit -m "Initial commit: dataset quality checker"
   ```
6. On GitHub click **New repository**, name it `dataset-quality-checker`, leave it empty (no README/license), create it.
7. Connect and push (use the URL GitHub shows you):
   ```powershell
   git remote add origin https://github.com/Sujalkrishnadas/dataset-quality-checker.git
   git push -u origin main
   ```
   A browser window may ask you to sign in.
8. Open the **Actions** tab. The workflow in `.github/workflows/tests.yml` runs the tests on Windows and Linux with Python 3.9, 3.11 and 3.12. Green check = passing.
9. Optional: add topics (`computer-vision`, `yolo`, `data-quality`) and tag a release: `git tag v0.1.0` then `git push --tags`.

## Part B: Contributing to someone else's project

Your own repository and a contribution to another project are different things. In your repo you decide everything. In someone else's repo the maintainers decide, and a pull request is a *proposal* they may accept, change or decline. Nothing here is guaranteed to be accepted.

1. **Find candidate projects.** Search GitHub for topics you now understand (`yolo`, `dataset`, `data-validation`, `computer-vision`). Prefer projects that are active (recent commits, replies to issues), have a `CONTRIBUTING.md`, and use Python.
2. **Read the rules.** Read `README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` and the license. Note how to set up, run tests, and format code.
3. **Find an issue.** Open the Issues tab and filter by labels such as `good first issue` or `help wanted`. Choose one that is open, unassigned, and has no linked pull request.
4. **Claim it politely.** Comment that you would like to work on it and briefly say how you plan to approach it. Wait for a maintainer to respond if the guidelines ask for it.
5. **Fork and clone.** Click **Fork**, then `git clone` your fork. Add the original as `upstream`: `git remote add upstream <original-url>`.
6. **Reproduce first.** Set up the environment as documented and make the bug happen yourself. If you cannot reproduce it, say so on the issue instead of guessing.
7. **Branch and fix.** `git switch -c fix/short-description`. Keep the change small and focused. Follow the project's style.
8. **Add a test** that fails before your change and passes after it, then run the project's full test suite and any linters.
9. **Commit and push** to your fork, following the project's commit message style.
10. **Open the pull request.** Describe the problem, link the issue (`Fixes #123`), explain your approach, and mention how you tested it.
11. **Respond to review.** Maintainers may ask for changes. That is normal. Reply politely and push updates to the same branch.

Tip: your first contributions can be documentation fixes or tests; maintainers often welcome them and they teach the workflow with low risk.
