from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from services.llm.fallback_chat import FallbackChatModel
from services.llm.llm_manager import ModelManager


def make_manager(monkeypatch, health_results):
    primary = object()
    fallback = object()
    embedding = object()

    settings = SimpleNamespace(
        CHAT_PRIMARY_MODEL="qwen2.5",
        CHAT_FALLBACK_MODEL="minimax",
        EMBEDDING_MODEL="bge-m3",
    )

    manager = ModelManager.__new__(ModelManager)
    manager.factory = SimpleNamespace(settings=settings)

    manager.get_chat_model = Mock(
        side_effect={
            "qwen2.5": primary,
            "minimax": fallback,
        }.get
    )
    manager.get_embedding_model = Mock(return_value=embedding)

    selected = FallbackChatModel(
        primary=primary,
        fallback=fallback,
        primary_name="qwen2.5",
        fallback_name="minimax",
    )
    manager.get_selected_chat_model = Mock(return_value=selected)

    manager._check_model = AsyncMock(side_effect=health_results)

    return manager, selected, primary, fallback, embedding


@pytest.mark.anyio
async def test_healthy_primary_skips_fallback_health_check(monkeypatch):
    manager, selected, _primary, _fallback, _embedding = make_manager(
        monkeypatch,
        [True, True],
    )

    await manager.health_check_models()

    checked_names = [
        call.kwargs["name"] for call in manager._check_model.await_args_list
    ]

    assert checked_names == ["qwen2.5", "bge-m3"]
    assert "minimax" not in checked_names
    manager.get_chat_model.assert_called_once_with("qwen2.5")
    manager.get_embedding_model.assert_called_once_with("bge-m3")
    assert selected._primary_failed_at is None


@pytest.mark.anyio
async def test_unhealthy_primary_checks_fallback_and_starts_with_it():
    manager, selected, _primary, _fallback, _embedding = make_manager(
        None,
        [False, True, True],
    )

    await manager.health_check_models()

    checked_names = [
        call.kwargs["name"] for call in manager._check_model.await_args_list
    ]

    assert checked_names == ["qwen2.5", "minimax", "bge-m3"]
    assert selected._primary_failed_at is not None


@pytest.mark.anyio
async def test_unhealthy_primary_and_fallback_prevent_startup():
    manager, _selected, _primary, _fallback, _embedding = make_manager(
        None,
        [False, False],
    )

    with pytest.raises(
        RuntimeError,
        match="Fallback chat model 'minimax' is unavailable",
    ):
        await manager.health_check_models()

    assert [call.kwargs["name"] for call in manager._check_model.await_args_list] == [
        "qwen2.5",
        "minimax",
    ]


@pytest.mark.anyio
async def test_unhealthy_embedding_prevents_startup_even_if_primary_is_healthy():
    manager, _selected, _primary, _fallback, _embedding = make_manager(
        None,
        [True, False],
    )

    with pytest.raises(
        RuntimeError,
        match="Embedding model 'bge-m3' is unavailable",
    ):
        await manager.health_check_models()

    assert [call.kwargs["name"] for call in manager._check_model.await_args_list] == [
        "qwen2.5",
        "bge-m3",
    ]
