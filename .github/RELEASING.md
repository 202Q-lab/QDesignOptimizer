# Releasing `qdesignoptimizer` to PyPI

This checklist walks you through publishing a new version of the package to
[PyPI](https://pypi.org/project/qdesignoptimizer/) so anyone can install it
with `pip install qdesignoptimizer`.  
Work through the steps **in order** — each one catches problems before they
reach the next.

> [!NOTE]
> Before following this checklist for the **first time**, also work through
> the one-off prerequisite in
> [Known blockers for the first PyPI release](#known-blockers-for-the-first-pypi-release)
> at the bottom of this file.

---

## 1 · Merge all open PRs and confirm CI is green

**Why:** PyPI releases are permanent — you can never overwrite a published
version.  Publishing from a dirty or broken state means shipping bugs that
can't be patched without bumping the version again.

- Merge (or close) all open pull requests into `main`.
- Check the **Actions** tab and make sure every workflow on `main` shows a
  green ✅ before continuing.

---

## 2 · Update `CHANGELOG.md` and bump the version

**Why:** Users and future maintainers rely on the changelog to understand what
changed. The version number in `pyproject.toml` is what ends up on PyPI — it
must match the tag you will create in step 6.

- In [`CHANGELOG.md`](../CHANGELOG.md) add (or finalize the draft of) a
  section for the new version, e.g.:
  ```markdown
  ## [0.2.0] - YYYY-MM-DD
  ### Added
  - ...
  ### Fixed
  - ...
  ```
- In [`pyproject.toml`](../pyproject.toml), set the same version number:
  ```toml
  [project]
  version = "0.2.0"
  ```
- Commit both files together: `git commit -m "Release 0.2.0"` and push to
  `main`.

---

## 3 · Run the unit tests locally

**Why:** The CI analysis workflow runs linting and type checks; the unit tests
exercise actual logic.  Running them locally before publishing catches any
environment-specific breakage early.

```bash
conda activate qdo202q
pytest tests/
```

The test suite is intentionally lightweight — it uses `unittest.mock` so no
HFSS or hardware is required. All four test modules should pass:

| File | What it tests |
|---|---|
| `tests/test_utils/test_names_parameters.py` | Parameter and mode name utilities |
| `tests/test_utils/test_utils.py` | Geometry helpers, unit parsing |
| `tests/test_sim_capacitance_matrix.py` | `CapacitanceMatrixStudy` (mocked) |
| `tests/test_sim_plot_progress.py` | Optimizer plots (mocked matplotlib) |

Fix any failures before proceeding.

---

## 4 · Smoke-test the tutorials

**Why:** Tutorials are the first thing new users run. A broken import or a
missing dependency that slipped through is very visible here.

- Open each notebook in [`tutorials/`](../tutorials/).
- Run at minimum the first two cells (imports + design setup) of each.
- If a notebook raises an `ImportError` or crashes, fix it before publishing.

---

## 5 · Test-publish to TestPyPI

**Why:** TestPyPI is a separate, throwaway index specifically for testing the
publish pipeline. This step proves the wheel builds correctly and installs
cleanly *before* anything touches the real PyPI.

1. Go to **GitHub → Actions → "Publish to TestPyPI"**.
2. Click **"Run workflow"** → **"Run workflow"** (no inputs needed — it runs
   on `main` automatically).
3. Wait for the green ✅.
4. Verify the install in a clean environment:
   ```bash
   pip install -i https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ \
       qdesignoptimizer
   python -c "import qdesignoptimizer; print('OK')"
   ```

If the install or the import fails, diagnose and fix before moving on.  
The secrets `TEST_PYPI_TOKEN` and `PYPI_TOKEN` are already configured in the
repository — you do not need to touch them.

---

## 6 · Publish the real release

**Why:** The production publish workflow
([`publish_release.yml`](workflows/publish_release.yml)) is triggered
automatically when you publish a **GitHub Release** — this is intentional so
that the git tag and the PyPI version are always in sync.

1. On GitHub, go to **Releases → "Draft a new release"**.
2. In **"Choose a tag"**, type `v0.2.0` (must match the version in
   `pyproject.toml`) and select **"Create new tag on publish"**.
3. Set the release title to `v0.2.0`.
4. Paste the relevant section from `CHANGELOG.md` into the description.
5. Click **"Publish release"**.

This triggers `publish_release.yml` which runs `poetry publish --build` using
`PYPI_TOKEN`.  Watch the **Actions** tab for the green ✅.

---

## 7 · Verify the live release

```bash
# In a clean environment (e.g. a fresh conda env):
pip install qdesignoptimizer
pip install --no-deps quantum-metal   # not on PyPI; must be installed separately
python -c "import qdesignoptimizer; print('OK')"
```

Also check the project page directly:
**https://pypi.org/project/qdesignoptimizer/**

Confirm the new version number appears and the description renders correctly.

---

## Quick reference — what triggers what

| Action | Workflow triggered | Target |
|---|---|---|
| Manual "Run workflow" click | `publish_test_release.yml` | TestPyPI |
| Publishing a GitHub Release | `publish_release.yml` | PyPI ✅ |

---

## Who has the tokens?

The repository secrets `PYPI_TOKEN` and `TEST_PYPI_TOKEN` are scoped to the
`202Q-lab/QDesignOptimizer` repository.  If you ever need to rotate them (e.g.
because a token expires or a maintainer leaves):

1. Log in to https://pypi.org with the 202Q-lab account.
2. Go to **Account settings → API tokens** and create a new token scoped to
   this project.
3. In the GitHub repo go to **Settings → Secrets and variables → Actions** and
   update `PYPI_TOKEN` (or `TEST_PYPI_TOKEN` for the test index).

---


