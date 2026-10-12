"""Unit tests for provider-agnostic LLM service and redaction safety."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from backend.app.core.config import Settings, reset_settings_for_test
from backend.app.services.llm_service import LLMService, LLMServiceError, redact_error


@pytest.fixture(autouse=True)
def reset_cfg():
    yield
    reset_settings_for_test(None)


@pytest.mark.asyncio
async def test_unconfigured_llm_raises_error():
    reset_settings_for_test(Settings(llm_api_key="", llm_provider="openai-compatible"))
    service = LLMService()
    assert not service.is_configured()

    with pytest.raises(LLMServiceError) as exc_info:
        await service.generate_hypotheses(
            dataset_profile={"columns": ["a"], "dtypes": {"a": "numeric"}},
            target_column="target",
            problem_type="classification",
        )
    assert "not configured" in str(exc_info.value)


@pytest.mark.asyncio
async def test_openai_compatible_adapter():
    reset_settings_for_test(
        Settings(
            llm_api_key="synthetic_key_val",
            llm_provider="openai-compatible",
            llm_base_url="https://api.openai.com/v1",
        )
    )
    service = LLMService()

    fake_response = {
        "choices": [
            {
                "message": {
                    "content": '[{"name": "interaction_a_b", "transform_type": "ratio", "expression": "df[\'a\'] / df[\'b\']", "input_columns": ["a", "b"], "hypothesis": "Ratio effect", "expected_signal": "strong", "risk": "none"}]'
                }
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake_response
    mock_resp.raise_for_status.return_value = None

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        results = await service.generate_hypotheses(
            dataset_profile={"columns": ["a", "b"], "dtypes": {"a": "numeric", "b": "numeric"}},
            target_column="target",
            problem_type="regression",
        )

        assert len(results) == 1
        assert results[0]["name"] == "interaction_a_b"
        assert results[0]["source"] == "llm_advisory"


@pytest.mark.asyncio
async def test_anthropic_adapter():
    reset_settings_for_test(
        Settings(
            llm_api_key="synthetic_key_val",
            llm_provider="anthropic",
            llm_base_url="https://api.anthropic.com",
        )
    )
    service = LLMService()

    fake_response = {
        "content": [
            {
                "text": '[{"name": "anthropic_feat", "transform_type": "log1p", "expression": "np.log1p(df[\'x\'])", "input_columns": ["x"], "hypothesis": "Log scaling", "expected_signal": "moderate", "risk": "none"}]'
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake_response
    mock_resp.raise_for_status.return_value = None

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        results = await service.generate_hypotheses(
            dataset_profile={"columns": ["x"], "dtypes": {"x": "numeric"}},
            target_column="y",
            problem_type="regression",
        )

        assert len(results) == 1
        assert results[0]["name"] == "anthropic_feat"


@pytest.mark.asyncio
async def test_gemini_adapter():
    reset_settings_for_test(
        Settings(
            llm_api_key="synthetic_key_val",
            llm_provider="gemini",
        )
    )
    service = LLMService()

    fake_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": '[{"name": "gemini_feat", "transform_type": "sqrt", "expression": "np.sqrt(df[\'z\'])", "input_columns": ["z"], "hypothesis": "Variance stabilizing", "expected_signal": "weak", "risk": "none"}]'
                        }
                    ]
                }
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake_response
    mock_resp.raise_for_status.return_value = None

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        results = await service.generate_hypotheses(
            dataset_profile={"columns": ["z"], "dtypes": {"z": "numeric"}},
            target_column="target",
            problem_type="classification",
        )

        assert len(results) == 1
        assert results[0]["name"] == "gemini_feat"


def test_error_redaction_synthetic():
    # Use runtime-generated dummy values, avoiding token-shaped literals
    dummy_key = "key_" + ("0" * 32)
    reset_settings_for_test(Settings(llm_api_key=dummy_key))
    raw_error = f"Request failed with secret {dummy_key} and Bearer test_synthetic_bearer_value"
    cleaned = redact_error(raw_error)

    assert dummy_key not in cleaned
    assert "[REDACTED" in cleaned
