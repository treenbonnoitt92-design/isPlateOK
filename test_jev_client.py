import os
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
import httpx
from dotenv import load_dotenv

# Ensure .env is loaded
load_dotenv()

from jev_client import JevClient


def test_missing_api_key_raises_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key"):
        JevClient(api_key=None)


def test_custom_init():
    client = JevClient(api_key="test-key", model="custom-model")
    assert client.api_key == "test-key"
    assert client.model == "custom-model"


def test_predict_mock():
    mock_resp_data = {
        "id": "dec-123",
        "model": "~typesafe/jev-latest",
        "answers": {
            "is_same_vehicle": {
                "type": "noul",
                "noul": 0.85
            }
        }
    }
    client = JevClient(api_key="test-key")
    state = {"entrance_plate": "京NC6545", "exit_plate": "京NC0545"}
    questions = {"is_same_vehicle": {"type": "noul", "instructions": "test"}}
    
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_resp_data

    with patch("httpx.Client.post", return_value=mock_response) as mock_post:
        result = client.predict(state, questions)
        assert mock_post.called
        args, kwargs = mock_post.call_args
        assert args[0] == "https://openrouter.ai/api/alpha/decisions"
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert kwargs["headers"]["Content-Type"] == "application/json"
        assert kwargs["json"]["model"] == "~typesafe/jev-latest"
        assert kwargs["json"]["state"] == state
        assert kwargs["json"]["questions"] == questions
        assert result["answers"]["is_same_vehicle"]["noul"] == 0.85


def test_predict_error_status():
    client = JevClient(api_key="test-key")
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 400
    mock_response.text = "Bad Request error payload"

    with patch("httpx.Client.post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="HTTP 400: Bad Request error payload"):
            client.predict({"test": 1}, {"test": 2})


@pytest.mark.asyncio
async def test_apredict_mock():
    mock_resp_data = {
        "id": "dec-456",
        "model": "~typesafe/jev-latest",
        "answers": {
            "is_same_vehicle": {
                "type": "noul",
                "noul": 0.92
            }
        }
    }
    client = JevClient(api_key="test-key")
    state = {"entrance_plate": "京NC6545", "exit_plate": "京NC0545"}
    questions = {"is_same_vehicle": {"type": "noul", "instructions": "test"}}
    
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_resp_data

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
        result = await client.apredict(state, questions)
        assert mock_post.called
        args, kwargs = mock_post.call_args
        assert args[0] == "https://openrouter.ai/api/alpha/decisions"
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert kwargs["headers"]["Content-Type"] == "application/json"
        assert kwargs["json"]["model"] == "~typesafe/jev-latest"
        assert kwargs["json"]["state"] == state
        assert kwargs["json"]["questions"] == questions
        assert result["answers"]["is_same_vehicle"]["noul"] == 0.92


@pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set in environment")
def test_live_predict():
    client = JevClient()
    state = {
        "entrance_plate": "京NC6545",
        "exit_plate": "京NC0545",
        "description": "6 vs 0 OCR confusion test"
    }
    questions = {
        "is_same_vehicle": {
            "type": "noul",
            "instructions": "Are these plates from the same vehicle?"
        }
    }
    res = client.predict(state, questions)
    assert "answers" in res
    assert "is_same_vehicle" in res["answers"]
    assert "noul" in res["answers"]["is_same_vehicle"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
