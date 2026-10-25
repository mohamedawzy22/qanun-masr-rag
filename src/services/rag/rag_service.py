import asyncio
import json
import re
import time
from dataclasses import dataclass

from langfuse import Langfuse, propagate_attributes

from services.llm.chat_interface import ChatModel
from services.prompts.rag.legal_rag import (
    document_prompt,
    footer_prompt,
    system_prompt,
)
from services.rag.indexing_service import IndexingService
from services.rag.online_evaluator import OnlineRAGEvaluator
from services.rag.retrieval_service import RetrievalService
from utils.logger import get_logger
from utils.metrics import (
    GENERATION_DURATION,
    GENERATION_FAILURES,
    GENERATION_NO_CONTEXT,
    GENERATION_REQUESTS,
    LLM_GENERATION_DURATION,
    PROMPT_BUILD_DURATION,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class EvaluationJob:
    trace_id: str
    project_id: str
    query: str
    answer: str
    results: list[dict]


class RAGService:
    def __init__(
        self,
        chat_model: ChatModel,
        retrieval_service: RetrievalService,
        indexing_service: IndexingService,
        langfuse: Langfuse,
        online_evaluator: OnlineRAGEvaluator | None = None,
        evaluation_interval_seconds: int = 300,
    ):
        self.chat_model = chat_model
        self.retrieval_service = retrieval_service
        self.indexing_service = indexing_service
        self.langfuse = langfuse
        self.online_evaluator = online_evaluator
        self.evaluation_interval_seconds = max(1, int(evaluation_interval_seconds))
        self._pending_evaluations: list[EvaluationJob] = []
        self._evaluation_worker_task: asyncio.Task[None] | None = None
        self._evaluation_stop_event = asyncio.Event()

        logger.info(
            "RAG service initialized successfully",
        )

    async def index(
        self,
        project_id: str,
        batch_size: int = 32,
        do_reset: bool = False,
    ) -> dict:
        return await self.indexing_service.index(
            project_id=project_id,
            batch_size=batch_size,
            do_reset=do_reset,
        )

    async def search(
        self,
        project_id: str,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        return await self.retrieval_service.search(
            project_id=project_id,
            query=query,
            limit=limit,
        )

    @staticmethod
    def _query_language(query: str) -> str:
        """Detect the dominant language of the user's question."""
        arabic_count = sum(
            char.isalpha() and "\u0600" <= char <= "\u06ff" for char in query
        )
        latin_count = sum(char.isascii() and char.isalpha() for char in query)
        return "ar" if arabic_count > latin_count else "en"

    @staticmethod
    def _format_prompt_document(
        result: dict,
        language: str,
    ) -> str:
        """Format bilingual legal evidence without mixing answer languages."""

        is_arabic = language == "ar"
        lines = []

        article_number = result.get("article_number")
        source_page = result.get("source_page")
        citation = result.get("citation")

        if article_number is not None:
            lines.append(
                f"المادة {article_number} من القانون المدني المصري"
                if is_arabic
                else f"Egyptian Civil Code, Article {article_number}"
            )
        elif citation:
            lines.append(f"Citation: {citation}")

        if source_page is not None:
            lines.append(
                f"صفحة PDF: {source_page}" if is_arabic else f"PDF page: {source_page}"
            )

        if result.get("is_repealed") is True:
            lines.append(
                "حالة المصدر: المادة مسجلة باعتبارها ملغاة."
                if is_arabic
                else "Source status: this article is marked as repealed."
            )

        text_ar = (result.get("text_ar") or "").strip()
        text_en = (result.get("text_en") or "").strip()

        # Prefer the readable companion text when available.
        if text_en:
            if is_arabic:
                lines.append(
                    "النص الإنجليزي المقابل للمادة، ويُستخدم لفهم المعنى "
                    "فقط. أجب بالعربية، وحافظ على النفي والشروط والاستثناءات."
                )
            else:
                lines.append("Corresponding English legal text:")
            lines.append(f"{text_en}")
        elif text_ar:
            arabic_label = (
                "النص العربي المستخرج، وقد يحتوي على أخطاء OCR. لا تنقله حرفيًا."
                if is_arabic
                else "Arabic source text, which may contain OCR errors."
            )
            lines.append(f"{arabic_label}:\n{text_ar}")

        # Fallback only when the separated language fields are unavailable.
        if not text_ar and not text_en:
            fallback_text = (result.get("text") or "").strip()

            if fallback_text:
                lines.append(
                    "مقتطف مستخرج، وقد يحتوي على أكثر من لغة:"
                    if is_arabic
                    else "Extracted excerpt, which may contain multiple languages:"
                )
                lines.append(fallback_text)

        return "\n".join(lines)

    @staticmethod
    def _build_prompt(
        query: str,
        results: list[dict],
    ) -> tuple[str, str, list[dict]]:
        language = RAGService._query_language(query)

        response_language = (
            "Arabic (Modern Standard Arabic)" if language == "ar" else "English"
        )

        system_text = system_prompt.substitute(
            response_language=response_language,
        )

        metadata_fields = (
            "article_number",
            "book",
            "chapter",
            "section",
            "topic",
            "text_ar",
            "text_en",
            "is_repealed",
            "source_page",
            "citation",
            "source_file",
        )

        documents = []

        for result in results:
            document = {
                "text": result.get("text", ""),
                "score": result.get("score", 0.0),
            }

            for field in metadata_fields:
                document[field] = result.get(field)

            documents.append(document)

        context = "\n\n".join(
            document_prompt.substitute(
                doc_num=index,
                chunk_text=RAGService._format_prompt_document(
                    result,
                    language,
                ),
            )
            for index, result in enumerate(results, start=1)
        )

        footer_text = footer_prompt.substitute(
            query=query,
            response_language=response_language,
        )

        user_prompt = "\n\n".join(part for part in (context, footer_text) if part)

        return system_text, user_prompt, documents

    def _schedule_evaluation(
        self,
        *,
        project_id: str,
        query: str,
        answer: str,
        results: list[dict],
    ) -> None:
        """Queue an evaluation without delaying the API response."""

        if self.online_evaluator is None:
            logger.error(
                "Cannot queue evaluation: evaluator is unavailable | project_id=%s",
                project_id,
            )
            return

        trace_id = self.langfuse.get_current_trace_id()

        if not trace_id:
            logger.error(
                "Cannot queue evaluation without an active trace | project_id=%s",
                project_id,
            )
            return

        self._pending_evaluations.append(
            EvaluationJob(
                trace_id=trace_id,
                project_id=project_id,
                query=query,
                answer=answer,
                results=results,
            )
        )

        if self._evaluation_worker_task is None or self._evaluation_worker_task.done():
            self._evaluation_stop_event.clear()
            self._evaluation_worker_task = asyncio.create_task(
                self._run_evaluation_worker(),
                name="rag-online-evaluation-worker",
            )

        logger.debug(
            "Online evaluation queued: project_id=%s | trace_id=%s | pending=%s",
            project_id,
            trace_id,
            len(self._pending_evaluations),
        )

    async def _run_evaluation_worker(self) -> None:
        """Evaluate queued answers periodically in the background."""

        logger.info(
            "Periodic online evaluation worker started: interval_seconds=%s",
            self.evaluation_interval_seconds,
        )

        try:
            while True:
                # Wait for the next scheduled batch, unless shutdown was requested.
                if not self._evaluation_stop_event.is_set():
                    try:
                        await asyncio.wait_for(
                            self._evaluation_stop_event.wait(),
                            timeout=self.evaluation_interval_seconds,
                        )
                    except TimeoutError:
                        pass

                batch = self._pending_evaluations
                self._pending_evaluations = []

                if batch:
                    logger.info(
                        "Starting online evaluation batch: count=%s",
                        len(batch),
                    )

                    for job in batch:
                        try:
                            await self._evaluate_and_score(
                                trace_id=job.trace_id,
                                project_id=job.project_id,
                                query=job.query,
                                answer=job.answer,
                                results=job.results,
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            logger.exception(
                                "Unexpected evaluation batch item failure: "
                                "project_id=%s | trace_id=%s",
                                job.project_id,
                                job.trace_id,
                            )

                    logger.info(
                        "Online evaluation batch completed: count=%s",
                        len(batch),
                    )

                # During shutdown, keep draining jobs that arrived while a batch ran.
                if (
                    self._evaluation_stop_event.is_set()
                    and not self._pending_evaluations
                ):
                    break

        except asyncio.CancelledError:
            logger.warning("Periodic online evaluation worker cancelled")
            raise
        except Exception:
            logger.exception("Periodic online evaluation worker failed")
        finally:
            if self._evaluation_worker_task is asyncio.current_task():
                self._evaluation_worker_task = None

    async def _evaluate_and_score(
        self,
        *,
        trace_id: str,
        project_id: str,
        query: str,
        answer: str,
        results: list[dict],
    ) -> None:
        """Evaluate one generated answer and attach scores to its trace."""

        evaluator = self.online_evaluator

        if evaluator is None:
            logger.error(
                "Online evaluator unavailable: project_id=%s",
                project_id,
            )
            return

        try:
            language = self._query_language(query)
            contexts = [
                self._format_prompt_document(result, language) for result in results
            ]

            try:
                scores, errors = await evaluator.evaluate(
                    question=query,
                    answer=answer,
                    contexts=contexts,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(
                    "Online evaluation execution failed: project_id=%s | trace_id=%s",
                    project_id,
                    trace_id,
                )
                scores = {}
                errors = {
                    "evaluation": f"{type(exc).__name__}: {exc}",
                }

            score_write_errors: dict[str, str] = {}

            for name, score in scores.items():
                try:
                    self.langfuse.create_score(
                        trace_id=trace_id,
                        name=name,
                        value=float(score),
                        data_type="NUMERIC",
                        comment="Online Ragas evaluation",
                        metadata={
                            "evaluator": "ragas",
                            "project_id": project_id,
                            "judge_model": evaluator.judge_model,
                            "embedding_model": evaluator.embedding_model,
                        },
                    )
                except Exception as exc:
                    score_write_errors[name] = type(exc).__name__

                    logger.exception(
                        "Failed to record online evaluation score: "
                        "metric=%s | project_id=%s | trace_id=%s",
                        name,
                        project_id,
                        trace_id,
                    )

            required_metrics = {
                "faithfulness",
                "answer_relevancy",
            }
            missing_metrics = sorted(required_metrics - scores.keys())

            if not scores:
                status = "failed"
            elif missing_metrics or errors or score_write_errors:
                status = "partial"
            else:
                status = "completed"

            # Record evaluation status separately from numeric metric scores.
            try:
                self.langfuse.create_score(
                    trace_id=trace_id,
                    name="online_evaluation_status",
                    value=status,
                    data_type="CATEGORICAL",
                    comment="Online evaluation execution status",
                    metadata={
                        "project_id": project_id,
                        "missing_required_metrics": missing_metrics,
                        "failed_metrics": sorted(errors),
                        "score_write_failures": sorted(score_write_errors),
                    },
                )
            except Exception:
                logger.exception(
                    "Failed to record evaluation status: project_id=%s | trace_id=%s",
                    project_id,
                    trace_id,
                )

            if scores:
                logger.info(
                    "Online RAG evaluation finished: "
                    "project_id=%s | trace_id=%s | status=%s | scores=%s",
                    project_id,
                    trace_id,
                    status,
                    {name: round(value, 4) for name, value in scores.items()},
                )
            else:
                logger.warning(
                    "Online RAG evaluation produced no scores: "
                    "project_id=%s | trace_id=%s | status=%s",
                    project_id,
                    trace_id,
                    status,
                )

            for name, error in errors.items():
                logger.warning(
                    "Online RAG metric failed: metric=%s | project_id=%s | error=%s",
                    name,
                    project_id,
                    error,
                )

        except asyncio.CancelledError:
            logger.warning(
                "Online evaluation cancelled during shutdown: "
                "project_id=%s | trace_id=%s",
                project_id,
                trace_id,
            )
            raise
        except Exception:
            logger.exception(
                "Unexpected online evaluation error: project_id=%s | trace_id=%s",
                project_id,
                trace_id,
            )

    async def generate(
        self,
        project_id: str,
        query: str,
        limit: int = 5,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> dict:
        """
        Retrieve relevant legal articles and generate an answer
        with structured source metadata.
        """

        logger.info(
            "Starting generation: project_id=%s | limit=%s",
            project_id,
            limit,
        )

        GENERATION_REQUESTS.inc()
        generation_start_time = time.perf_counter()

        trace_attributes = {
            key: value
            for key, value in {
                "user_id": user_id,
                "session_id": session_id,
            }.items()
            if value is not None
        }

        with (
            propagate_attributes(**trace_attributes),
            self.langfuse.start_as_current_observation(
                as_type="chain",
                name="rag.generate",
                input={
                    "query": query,
                    "limit": limit,
                },
                metadata={
                    "project_id": project_id,
                },
            ) as rag_observation,
        ):
            try:
                # --------------------------------------------
                # Retrieve relevant legal articles
                # --------------------------------------------
                # Use vector-search results directly. Avoid an extra LLM
                # call for evidence selection before answer generation.
                results = await self.retrieval_service.search(
                    project_id=project_id,
                    query=query,
                    limit=limit,
                )

                # --------------------------------------------
                # Build the prompt
                # --------------------------------------------
                prompt_start_time = time.perf_counter()

                try:
                    with self.langfuse.start_as_current_observation(
                        as_type="span",
                        name="rag.prompt-construction",
                        input={
                            "query": query,
                            "document_count": len(results),
                        },
                        metadata={
                            "project_id": project_id,
                        },
                    ) as prompt_observation:
                        try:
                            system_prompt_text, user_prompt, _ = self._build_prompt(
                                query=query,
                                results=results,
                            )

                            prompt_observation.update(
                                output={
                                    "prompt_length": len(user_prompt),
                                    "system_prompt_length": len(system_prompt_text),
                                    "document_count": len(results),
                                },
                            )

                        except Exception as exc:
                            prompt_observation.update(
                                level="ERROR",
                                status_message=type(exc).__name__,
                            )
                            raise
                finally:
                    PROMPT_BUILD_DURATION.observe(
                        time.perf_counter() - prompt_start_time,
                    )

                # --------------------------------------------
                # No relevant context
                # --------------------------------------------

                if not results:
                    GENERATION_NO_CONTEXT.inc()

                    logger.info(
                        "No relevant documents found: project_id=%s",
                        project_id,
                    )

                    answer = (
                        "لم أجد في النصوص القانونية المسترجعة "
                        "ما يكفي للإجابة عن السؤال."
                        if self._query_language(query) == "ar"
                        else "The retrieved legal sources do not contain "
                        "enough information to answer this question."
                    )

                    response = {
                        "answer": answer,
                        "sources": [],
                    }

                    rag_observation.update(
                        output={
                            "answer": answer,
                            "source_count": 0,
                        },
                    )

                    self._schedule_evaluation(
                        project_id=project_id,
                        query=query,
                        answer=answer,
                        results=results,
                    )

                    return response

                # --------------------------------------------
                # Generate the answer
                # --------------------------------------------
                llm_start_time = time.perf_counter()

                try:
                    generate_validated = getattr(
                        self.chat_model,
                        "generate_validated",
                        None,
                    )

                    if callable(generate_validated):
                        response_text = await generate_validated(
                            prompt=user_prompt,
                            system_prompt=system_prompt_text,
                        )
                    else:
                        response_text = await self.chat_model.generate(
                            prompt=user_prompt,
                            system_prompt=system_prompt_text,
                        )
                finally:
                    LLM_GENERATION_DURATION.observe(
                        time.perf_counter() - llm_start_time,
                    )

                # --------------------------------------------
                # Build structured source objects
                # --------------------------------------------
                sources = self._build_sources(results, answer=response_text)
                # --------------------------------------------
                # Final API response
                # --------------------------------------------
                response = {
                    "answer": response_text,
                    "sources": sources,
                }

                rag_observation.update(
                    output={
                        "answer": response_text,
                        "source_count": len(sources),
                    },
                )

                self._schedule_evaluation(
                    project_id=project_id,
                    query=query,
                    answer=response_text,
                    results=results,
                )

                logger.info(
                    "Generation completed successfully: project_id=%s | sources=%s",
                    project_id,
                    len(sources),
                )

                return response

            except Exception as exc:
                GENERATION_FAILURES.inc()

                try:
                    rag_observation.update(
                        level="ERROR",
                        status_message=type(exc).__name__,
                    )
                except Exception:
                    logger.exception(
                        "Failed to update Langfuse observation after generation error",
                    )

                logger.exception(
                    "Generation failed: project_id=%s",
                    project_id,
                )

                raise

            finally:
                GENERATION_DURATION.observe(
                    time.perf_counter() - generation_start_time,
                )

    async def close(
        self,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Drain pending evaluations before closing their dependencies."""

        task = self._evaluation_worker_task

        if task is None:
            return

        logger.info(
            "Stopping online evaluation worker: pending=%s",
            len(self._pending_evaluations),
        )

        # Wake the worker so it can flush queued evaluations without waiting
        # for the full periodic interval.
        self._evaluation_stop_event.set()

        _, pending_tasks = await asyncio.wait(
            {task},
            timeout=timeout_seconds,
        )

        if pending_tasks:
            logger.warning(
                "Online evaluation worker exceeded shutdown timeout; cancelling it"
            )
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        if self._pending_evaluations:
            logger.warning(
                "Some evaluations did not finish during shutdown: pending=%s",
                len(self._pending_evaluations),
            )

        if self._evaluation_worker_task is task:
            self._evaluation_worker_task = None

    async def _select_relevant_results(
        self,
        query: str,
        results: list[dict],
        limit: int,
    ) -> list[dict]:
        """Select relevant legal evidence before generating the answer."""

        if len(results) <= limit:
            return results

        candidates = []
        for index, result in enumerate(results, start=1):
            candidates.append(
                {
                    "id": index,
                    "article_number": result.get("article_number"),
                    "topic": result.get("topic"),
                    "arabic_text": (result.get("text_ar") or "")[:1200],
                    "english_text": (result.get("text_en") or "")[:1200],
                }
            )

        system_text = (
            "You are a legal evidence relevance selector for questions "
            "about the Egyptian Civil Code. Do not answer the legal question. "
            "Assess each candidate by its actual statutory content, not by "
            "its similarity score, article number, or broad topic label. "
            "Select only sources that directly help answer the user's "
            "question. Reject sources that discuss a different legal issue, "
            "even if they share words with the question. Do not invent "
            "missing rules. Return at most the requested number of candidate "
            "IDs, ordered by relevance, as valid JSON only in this format: "
            '{"selected_ids": [1, 3]}. '
            "Use only IDs present in the candidate list. If none is relevant, "
            'return {"selected_ids": []}.'
        )

        selection_prompt = (
            f"User question:\n{query}\n\n"
            f"Select at most {limit} relevant sources from these candidates. "
            "Return JSON only.\n\n"
            f"{json.dumps(candidates, ensure_ascii=False)}"
        )

        try:
            with self.langfuse.start_as_current_observation(
                as_type="span",
                name="rag.evidence-selection",
                input={
                    "query": query,
                    "candidate_count": len(results),
                    "max_selected": limit,
                },
                metadata={"strategy": "prompt_based_selection"},
            ) as selection_observation:

                def valid_selection_output(raw: str) -> bool:
                    candidate = raw.strip()
                    if not candidate.startswith("{"):
                        json_start = candidate.find("{")
                        json_end = candidate.rfind("}")
                        if json_start < 0 or json_end <= json_start:
                            return False
                        candidate = candidate[json_start : json_end + 1]

                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        return False

                    if not isinstance(parsed, dict) or set(parsed) != {"selected_ids"}:
                        return False

                    ids = parsed["selected_ids"]
                    if not isinstance(ids, list) or len(ids) > limit:
                        return False

                    if any(
                        type(value) is not int or value < 1 or value > len(results)
                        for value in ids
                    ):
                        return False

                    return len(ids) == len(set(ids))

                generate_validated = getattr(
                    self.chat_model,
                    "generate_validated",
                    None,
                )

                if callable(generate_validated):
                    response_text = await generate_validated(
                        prompt=selection_prompt,
                        system_prompt=system_text,
                        temperature=0.0,
                        max_tokens=2000,
                        validator=valid_selection_output,
                    )
                else:
                    response_text = await self.chat_model.generate(
                        prompt=selection_prompt,
                        system_prompt=system_text,
                        temperature=0.0,
                        max_tokens=2000,
                    )
                    if not valid_selection_output(response_text):
                        raise ValueError("Evidence selector returned invalid JSON.")

                raw = response_text.strip()

                # Tolerate a JSON object wrapped in a code fence or prose.
                try:
                    selection = json.loads(raw)
                except json.JSONDecodeError:
                    start = raw.find("{")
                    end = raw.rfind("}")
                    if start < 0 or end <= start:
                        raise ValueError(
                            "Evidence selector did not return a JSON object."
                        )
                    selection = json.loads(raw[start : end + 1])

                selected_ids = selection.get("selected_ids")
                if not isinstance(selected_ids, list):
                    raise TypeError("Evidence selector returned no selected_ids list.")

                valid_ids = []
                for value in selected_ids:
                    if (
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and 1 <= value <= len(results)
                        and value not in valid_ids
                    ):
                        valid_ids.append(value)

                selected = [results[index - 1] for index in valid_ids[:limit]]

                selection_observation.update(
                    output={
                        "selected_count": len(selected),
                        "selected_article_numbers": [
                            item.get("article_number") for item in selected
                        ],
                    },
                )

                return selected

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Evidence selection failed; using top vector results: %s",
                type(exc).__name__,
            )
            return results[:limit]

    @staticmethod
    def _build_sources(
        results: list[dict],
        answer: str | None = None,
    ) -> list[dict]:
        """Return metadata for retrieved articles cited in the answer."""

        if answer is not None:
            digit_map = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
            cited_articles = {
                number.translate(digit_map)
                for number in re.findall(
                    r"(?i)(?:المادة|المواد|articles?)\s*(?:رقم\s*)?[\(\[]?\s*([0-9٠-٩]+)",
                    answer,
                )
            }
            results = [
                result
                for result in results
                if str(result.get("article_number") or "").translate(digit_map)
                in cited_articles
            ]

        sources = []

        for result in results:
            text_ar = (result.get("text_ar") or "").strip()
            text_en = (result.get("text_en") or "").strip()

            # A source without either original-language text cannot
            # provide useful statutory evidence in the response.
            if not text_ar and not text_en:
                continue

            article_number = result.get("article_number")
            citation = result.get("citation")

            if not citation and article_number is not None:
                citation = f"Egyptian Civil Code, Article {article_number}"

            sources.append(
                {
                    "score": float(result.get("score") or 0.0),
                    "article_number": article_number,
                    "book": result.get("book"),
                    "chapter": result.get("chapter"),
                    "section": result.get("section"),
                    "topic": result.get("topic"),
                    "text_ar": text_ar or None,
                    "text_en": text_en or None,
                    "is_repealed": result.get("is_repealed"),
                    "source_page": result.get("source_page"),
                    "citation": citation,
                    "source_file": result.get("source_file"),
                }
            )

        return sources
