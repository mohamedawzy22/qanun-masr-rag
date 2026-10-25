# ==========================================================
# Qanun Masr RAG - Docker Development Commands
# ==========================================================

.DEFAULT_GOAL := help

COMPOSE = docker compose --env-file .env -f docker/docker-compose.yml
API_URL ?= http://127.0.0.1:8001/api/v1/health

.PHONY: help config build up start stop down status logs logs-api health restart-api api load-test stop-load-test logs-load-test

help:
	@echo ""
	@echo "Qanun Masr RAG - Available commands"
	@echo "----------------------------------"
	@echo "make help        Show available commands"
	@echo "make config      Validate Docker Compose configuration"
	@echo "make build       Build the RAG API image"
	@echo "make up          Build and start the complete stack"
	@echo "make start       Start the stack without forcing a rebuild"
	@echo "make stop        Stop containers and preserve them"
	@echo "make down        Stop and remove containers and networks"
	@echo "make status      Show container status"
	@echo "make logs        Follow logs from all services"
	@echo "make logs-api    Follow logs from the RAG API"
	@echo "make load-test      Start optional Locust load testing"
	@echo "make stop-load-test Stop Locust"
	@echo "make logs-load-test Follow Locust logs"
	@echo "make health      Check the API health endpoint"
	@echo "make restart-api Restart the RAG API container"
	@echo "make api         Build and start the API with its dependencies"
	@echo ""

config:
	$(COMPOSE) config -q
	@echo "Docker Compose configuration is valid."

build: config
	$(COMPOSE) build rag-api

up: config
	$(COMPOSE) up -d --build
	@echo ""
	@echo "Docker services started. Check status with: make status"
	@echo "Check API health with: make health"

start: config
	$(COMPOSE) up -d
	@echo "Docker services started."

stop:
	$(COMPOSE) stop

down:
	$(COMPOSE) down
	@echo "Containers and networks removed. Named volumes are preserved."

status:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs -f --tail=100

logs-api:
	$(COMPOSE) logs -f --tail=100 rag-api

health:
	@curl --fail --silent --show-error --max-time 5 "$(API_URL)"

restart-api:
	$(COMPOSE) restart rag-api

api: config
	$(COMPOSE) up -d --build rag-api
	@echo "RAG API deployment command completed. Check with: make health"

load-test:
	$(COMPOSE) --profile loadtest config -q
	$(COMPOSE) --profile loadtest up -d locust
	@echo "Locust UI: http://localhost:8089"
	@echo "Start conservatively, e.g. 1 user at 1 user/second."

stop-load-test:
	$(COMPOSE) --profile loadtest stop locust

logs-load-test:
	$(COMPOSE) --profile loadtest logs -f --tail=100 locust
