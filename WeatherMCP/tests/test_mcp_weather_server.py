import pytest
import httpx
from src.mcp_weather_server.server import get_weather

@pytest.mark.asyncio
async def test_get_weather_geocoding_error():
    async def mock_get(url, params=None):
        if "geocoding-api" in url:
            return httpx.Response(500)
        raise ValueError(f"Unexpected URL: {url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(mock_get)) as client:
        result = await get_weather("InvalidCity")
        assert "Error: Could not retrieve coordinates for InvalidCity." in result

@pytest.mark.asyncio
async def test_get_weather_missing_results():
    async def mock_get(url, params=None):
        if "geocoding-api" in url:
            return httpx.Response(200, json={})  # Missing "results" key
        raise ValueError(f"Unexpected URL: {url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(mock_get)) as client:
        result = await get_weather("SomeCity")
        assert "Error: Could not retrieve coordinates for SomeCity." in result

@pytest.mark.asyncio
async def test_get_weather_invalid_city():
    async def mock_get(url, params=None):
        if "geocoding-api" in url:
            return httpx.Response(200, json={"results": []})  # Empty "results"
        raise ValueError(f"Unexpected URL: {url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(mock_get)) as client:
        result = await get_weather("InvalidCity")
        assert "Error: Could not retrieve coordinates for InvalidCity." in result
