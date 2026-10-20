# Docker Setup

Run all commands from the repository root:

```bash
cd ~/projects/OrangeHub/qanun-masr-rag
```

## Start the infrastructure

Start the standard services:

```bash
docker compose -f docker/docker-compose.yml up -d
```

Check status:

```bash
docker compose -f docker/docker-compose.yml ps
```

View logs for a service, for example MinIO:

```bash
docker compose -f docker/docker-compose.yml logs -f minio
```

## DVC dataset versioning with MinIO

MinIO stores application uploads. DVC is separate and is used for datasets that need version history, not for every API upload.

Install DVC in the main project and initialize it once:

```bash
uv add --dev "dvc[s3]"
uv run dvc init
```

If DVC is already initialized, do not run `dvc init` again.

Configure a remote under a separate prefix in the existing MinIO bucket:

```bash
uv run dvc remote add -d minio-dvc s3://qanun-masr-assets/dvc-cache
uv run dvc remote modify minio-dvc endpointurl http://localhost:9091
uv run dvc remote modify --local minio-dvc access_key_id "<MINIO_ACCESS_KEY>"
uv run dvc remote modify --local minio-dvc secret_access_key "<MINIO_SECRET_KEY>"
```

Replace the placeholders with the MinIO credentials. The `--local` settings are stored in `.dvc/config.local`; do not commit that file or the credentials. If the remote already exists, do not add it again.

Track and push a dataset when needed. Replace the example path with the actual dataset file:

```bash
uv run dvc add path/to/dataset.jsonl
git add path/to/dataset.jsonl.dvc .gitignore
git commit -m "Track dataset with DVC"
uv run dvc push
```

Restore tracked data on another checkout with:

```bash
uv run dvc pull
```
