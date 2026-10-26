# Qanun Masr: staged candidate + manual promotion

This bundle replaces `.github/workflows/cd.yml` and adds:

- `.github/scripts/integration-test.sh`: disposable integration environment for MongoDB, Qdrant, MinIO, and Ollama.
- `.github/workflows/release.yml`: manually promotes a tested immutable image SHA to the `latest` GHCR tag.

## Intended flow

1. CI continues to validate pull requests before merge.
2. After a commit reaches `main`, CD builds one image and runs a fast HTTP smoke test plus the full startup integration test in disposable containers.
3. Only if both tests pass, the image is pushed to GHCR using immutable `sha-<full-commit-sha>` and review-friendly `candidate-<short-sha>` tags. It does **not** change `latest`.
4. Review the staging job logs and its summary in GitHub Actions.
5. After approval, manually run **Actions -> Promote Tested Image -> Run workflow** on `main`, and enter the full 40-character SHA from the successful candidate run. The workflow promotes that exact image to `latest` in GHCR.

## Install

Extract the ZIP into the root of the repository, preserving the `.github` directory structure. Review the resulting diff before committing. Do not copy any `.env` or production secrets into these files.

## GitHub setup

- The staging test intentionally uses only disposable CI credentials and small Ollama models; it does not use `ENV_APP` or production API keys.
- In **Settings -> Environments**, create an environment named `production` if you want GitHub's environment gate. Configure required reviewers there if you want an additional approval step. If you are the only reviewer, check the self-review setting so it does not block your own release.
- The manual workflow promotes the container image in GHCR. It does not deploy to a live VM/server because this repository does not yet define a production host/deployment target.

## Notes

The staging job validates app startup, MongoDB, MinIO bucket initialization, Qdrant collection initialization, and Ollama model availability/warmup. Langfuse is initialized with test-only credentials and a non-routable local endpoint, so this does not validate a real Langfuse server or ingestion delivery. Full request-level RAG quality/evaluation should be added as a separate test once representative test documents and expected behavior are selected.
