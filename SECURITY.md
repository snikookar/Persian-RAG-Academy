# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | ✅ |
| older development snapshots | ❌ |

## Reporting a vulnerability

**Please do not open a public issue for security problems.** Report them privately with
GitHub's [private vulnerability reporting](https://github.com/snikookar/Persian-RAG-Academy/security/advisories/new)
(*Security → Report a vulnerability*). Include the affected version or commit, how to reproduce
it, and the impact you expect.

You can expect an acknowledgement within 7 days and an assessment within 30 days. Fixes are
released as a new patch version and credited in the advisory, unless you prefer to stay anonymous.

## Scope

In scope: the `parsrag` package (`src/parsrag/`), the API service and web UI (`parsrag serve`,
the `Dockerfile` and `docker-compose.yml`), and the repository's CI workflows.

Things worth knowing when you deploy `parsrag`:

- **The API has no authentication.** `parsrag serve` and the Docker image are meant to run behind
  your own gateway or on a private network. Rate limiting is per API key or IP, not access control.
- **Uploaded documents are untrusted input.** File size and type are limited
  (`PARSRAG_MAX_UPLOAD_MB`), but PDF/DOCX/HTML parsing relies on third-party libraries
  (`pypdf`, `python-docx`, `beautifulsoup4`); keep them up to date.
- **Prompt injection inside documents** is a known, unsolved risk for every RAG system: a
  document can contain instructions aimed at the model. Grounded generation, citation
  validation and abstention (lesson 09) reduce the damage but do not prevent it. Do not give the
  model tools or permissions that a malicious document could abuse.
- **API keys** are read from environment variables (`api_key_env` in `parsrag.toml`), never from
  config files; do not commit `.env` files. Logs record a hash of each question, not its text.
- The `docker-compose.yml` database password is a local-development default. Change it for any
  shared deployment.

Out of scope: the example fixtures, the course text, and issues that require an already
compromised machine or a malicious configuration file.
