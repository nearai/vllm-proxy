import pytest
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set required environment variables before any imports
# Note: MODEL_NAME is no longer required as of multi-backend support
os.environ["VLLM_BASE_URL"] = "http://localhost:8001"
# For multi-backend testing, use VLLM_BASE_URLS:
# os.environ["VLLM_BASE_URLS"] = "http://localhost:8001,http://localhost:8002"
os.environ["CHAT_CACHE_EXPIRATION"] = "1200"
os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PORT"] = "6379"
os.environ["REDIS_DB"] = "0"
os.environ["AUTH_TOKEN"] = "test-token"
os.environ["TOKEN"] = "test_token"
os.environ["SIGNING_METHOD"] = "ecdsa"


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "asyncio: mark test as an asyncio test")


@pytest.fixture(autouse=True)
def reset_backend_state():
    """Reset backend pool and config state between tests."""
    from app.backend.config import reset_backend_config
    from app.backend.pool import reset_backend_pool

    # Reset before test
    reset_backend_config()
    reset_backend_pool()

    yield

    # Reset after test
    reset_backend_config()
    reset_backend_pool()