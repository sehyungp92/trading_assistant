from __future__ import annotations

from httpx import ASGITransport, AsyncClient

import pytest

from orchestrator.app import create_app
from orchestrator.config import AppConfig


@pytest.fixture
def protected_app(tmp_path):
    config = AppConfig(orchestrator_api_key="super-secret")
    return create_app(db_dir=str(tmp_path), config=config)


class TestOrchestratorApiKeyAuth:
    @pytest.mark.asyncio
    async def test_health_remains_open(self, protected_app):
        async with AsyncClient(
            transport=ASGITransport(app=protected_app),
            base_url="http://test",
        ) as client:
            response = await client.get("/health")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_metrics_requires_api_key(self, protected_app):
        async with AsyncClient(
            transport=ASGITransport(app=protected_app),
            base_url="http://test",
        ) as client:
            response = await client.get("/metrics")

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid API key"

    @pytest.mark.asyncio
    async def test_metrics_allows_valid_api_key(self, protected_app):
        async with protected_app.router.lifespan_context(protected_app):
            async with AsyncClient(
                transport=ASGITransport(app=protected_app),
                base_url="http://test",
            ) as client:
                response = await client.get(
                    "/metrics",
                    headers={"X-Api-Key": "super-secret"},
                )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_mutating_endpoint_requires_api_key(self, protected_app):
        async with AsyncClient(
            transport=ASGITransport(app=protected_app),
            base_url="http://test",
        ) as client:
            response = await client.put(
                "/notifications/preferences",
                json={"channels": []},
            )

        assert response.status_code == 401
