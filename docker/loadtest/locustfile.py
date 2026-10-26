import os
import random

from locust import HttpUser, between, task

PROJECT_ID = os.getenv("LOCUST_PROJECT_ID", "1").strip()

QUESTIONS = (
    "ما أثر الظروف الاستثنائية على الالتزامات التعاقدية؟",
    "ما شروط صحة العقد وفقًا للقانون المدني المصري؟",
    "What are the legal effects of a valid contract?",
    "How does Egyptian Civil Code Article 147 address exceptional events?",
)


class QanunMasrRAGUser(HttpUser):
    """Load-test the health, search, and answer-generation endpoints."""

    wait_time = between(1, 3)

    @task(1)
    def health(self):
        with self.client.get(
            "/api/v1/health",
            name="GET /api/v1/health",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Unexpected HTTP status: {response.status_code}")
                return

            try:
                payload = response.json()
            except ValueError:
                response.failure("Health endpoint returned invalid JSON")
                return

            if payload.get("API_Name") != "rag-system":
                response.failure("Unexpected health response")
            else:
                response.success()

    @task(1)
    def search(self):
        with self.client.post(
            f"/api/v1/rag/search/{PROJECT_ID}",
            json={
                "query": random.choice(QUESTIONS),
                "limit": 5,
            },
            name="POST /api/v1/rag/search/[project_id]",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Unexpected HTTP status: {response.status_code}")
                return

            try:
                payload = response.json()
            except ValueError:
                response.failure("Search endpoint returned invalid JSON")
                return

            if not isinstance(payload.get("results"), list):
                response.failure("Search response has no results list")
            else:
                response.success()

    @task(1)
    def generate(self):
        with self.client.post(
            f"/api/v1/rag/generate/{PROJECT_ID}",
            json={
                "query": random.choice(QUESTIONS),
                "limit": 5,
            },
            name="POST /api/v1/rag/generate/[project_id]",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Unexpected HTTP status: {response.status_code}")
                return

            try:
                payload = response.json()
            except ValueError:
                response.failure("Generate endpoint returned invalid JSON")
                return

            if (
                not isinstance(payload.get("answer"), str)
                or not payload["answer"].strip()
            ):
                response.failure("Generate response has no answer")
            else:
                response.success()
