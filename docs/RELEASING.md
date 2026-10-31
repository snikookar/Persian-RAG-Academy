# Releasing parsrag

Releases are published by [`.github/workflows/release.yml`](../.github/workflows/release.yml)
with **PyPI Trusted Publishing**: GitHub Actions proves its identity to PyPI with a short-lived
OIDC token, so no API token is ever created, copied or stored as a secret.

| Trigger | Publishes to | GitHub environment |
|---|---|---|
| Manual run of the *Release* workflow (Actions tab → Release → Run workflow) | TestPyPI | `testpypi` |
| Publishing a GitHub release | PyPI | `pypi` |

Both paths first build the sdist and wheel, run `scripts/check_dist.py` (only `src/parsrag`,
`README.md`, `LICENSE`, `pyproject.toml` and `CHANGELOG.md` may ship), `twine check --strict`,
and install the wheel in a clean virtual environment. A release whose tag does not match the
version in `pyproject.toml` (`v0.1.0` ↔ `0.1.0`) fails before anything is uploaded.

## One-time setup

Do this once, as the maintainer. Nothing here can be done from a pull request.

### 1. TestPyPI (test.pypi.org)

1. Create an account on <https://test.pypi.org/account/register/> and enable two-factor
   authentication (required for publishing).
2. Open <https://test.pypi.org/manage/account/publishing/> and, under
   **Add a new pending publisher**, fill in:
   - PyPI project name: `parsrag`
   - Owner: `snikookar`
   - Repository name: `Persian-RAG-Academy`
   - Workflow name: `release.yml`
   - Environment name: `testpypi`
3. Save. The project is created on TestPyPI by the first successful upload.

### 2. PyPI (pypi.org)

1. Create an account on <https://pypi.org/account/register/> and enable two-factor
   authentication.
2. Check that the name `parsrag` is still free: <https://pypi.org/project/parsrag/> should be a 404.
3. Open <https://pypi.org/manage/account/publishing/> and add a **pending publisher** with:
   - PyPI project name: `parsrag`
   - Owner: `snikookar`
   - Repository name: `Persian-RAG-Academy`
   - Workflow name: `release.yml`
   - Environment name: `pypi`

A pending publisher does not reserve the name; publish the first release soon after creating it.

### 3. GitHub (repository settings)

1. **Settings → Environments → New environment** → `testpypi`. No secrets are needed.
2. **Settings → Environments → New environment** → `pypi`. Recommended protection:
   - *Required reviewers*: yourself, so every PyPI upload waits for a manual approval;
   - *Deployment branches and tags*: **Selected branches and tags** → add the tag rule `v*`.
3. **Settings → Pages → Build and deployment → Source: GitHub Actions** (for the docs site,
   published by [`.github/workflows/docs.yml`](../.github/workflows/docs.yml)).
4. Optional: **Settings → General → Social preview** → upload `docs/images/social-preview.png`
   (regenerate with `python scripts/make_social_preview.py`).

## Every release

1. Make sure `main` is green (CI, Docs) and the eval gate passes.
2. Update the version in **two** places — `pyproject.toml` (`version`) and
   `src/parsrag/__init__.py` (`__version__`) — and in `CHANGELOG.md` move the *Unreleased*
   entries under a heading `## [X.Y.Z] - YYYY-MM-DD` (for 0.1.0, replace "not yet released"
   with the date) and point its link at the tag. Merge that PR.
3. Dry run on TestPyPI: **Actions → Release → Run workflow** on `main`. Then, in a clean
   virtual environment:

   ```bash
   pip install --index-url https://test.pypi.org/simple/ \
               --extra-index-url https://pypi.org/simple/ parsrag
   python -c "import parsrag; print(parsrag.__version__)"
   parsrag --help
   ```

   TestPyPI does not accept the same version twice; for another dry run of the same code,
   bump to a pre-release such as `0.1.1rc1`.
4. Create the release: **Releases → Draft a new release**, tag `vX.Y.Z` on `main`, title
   `vX.Y.Z`, paste the changelog section, **Publish release**. The workflow builds, checks
   and — after approval in the `pypi` environment — uploads to PyPI.
5. Check <https://pypi.org/project/parsrag/>: the README renders, images load, links point to
   GitHub, and `pip install parsrag` works.

PyPI never allows re-uploading a version, even after deleting it. If something is wrong, fix it
and release the next patch version.

## Building locally (no upload)

```bash
pip install build twine
python -m build
python scripts/check_dist.py
twine check --strict dist/*
```
