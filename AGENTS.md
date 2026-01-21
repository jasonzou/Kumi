# AGENTS.md - KUMI Knowledge Base Testing Platform

## Project Overview

KUMI is a visual knowledge base testing platform for evaluating embedding and LLM system effectiveness. Built with FastAPI + Python 3.12+.

## Quick Start

```bash
# Environment setup
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt

# Config files
cp config/.env.example config/.env
cp config/embedding_providers.yaml.example config/embedding_providers.yaml

# Run development server
uv run scripts/start_dev.py
# Access: http://127.0.0.1:8000/web (admin / KUMI-admin-123456)
```

## Development Commands

```bash
# Start server
uv run scripts/start_dev.py          # Preferred
python main.py                        # Alternative

# Start ChromaDB (required dependency)
chroma run --path storage/testdb --port 8081 --host 127.0.0.1

# Start local embedding service (optional)
cd scripts && uv run embedding_serve.py --model_path ./Qwen3-Embedding-0.6B --model_name Qwen3-Embedding-0.6B
```

## Testing

```bash
# Run all tests
uv run pytest

# Run single test file
uv run pytest tests/test_file.py

# Run single test function
uv run pytest tests/test_file.py::test_function_name

# With verbose output
uv run pytest -v tests/test_file.py
```

Note: Test suite is minimal. Most testing is done via service files directly:
```bash
python services/testqa_service.py
```

## Code Style Guidelines

### Imports (order matters)
```python
# 1. Standard library
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# 2. Third-party
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

# 3. Local
from config.settings import settings
from services.knowledge_service import KnowledgeService
```

### Naming Conventions
- **Classes**: PascalCase (`KnowledgeService`, `VectorDBInterface`)
- **Functions/Methods**: snake_case (`create_collection`, `search_knowledge`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_MODEL`, `MAX_FILE_SIZE`)
- **Private**: underscore prefix (`_load_config`, `_validate_input`)

### Type Hints (required for all functions)
```python
def search_knowledge(
    query: str,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Search knowledge base with given query."""
    pass
```

### Logging (use loguru)
```python
from loguru import logger

logger.info(f"Processing: {filename}")
logger.error(f"Failed: {e}")
logger.debug(f"Debug info: {data}")
```

### Error Handling
```python
try:
    result = await some_operation()
    return result
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    raise HTTPException(status_code=400, detail=str(e))
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise HTTPException(status_code=500, detail="Internal server error")
```

### API Endpoints (FastAPI + Pydantic)
```python
from pydantic import BaseModel, Field
from fastapi import APIRouter

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(10, ge=1, le=100, description="Result limit")

router = APIRouter(prefix="/api/v1/knowledge")

@router.post("/search")
async def search_endpoint(request: SearchRequest):
    """Search knowledge base."""
    pass
```

## Architecture

```
KUMI/
├── api/                    # FastAPI route handlers + Pydantic models
├── services/               # Business logic layer
├── vector_db/              # Vector DB clients (ChromaDB, Milvus)
│   ├── base.py            # VectorDBInterface (ABC)
│   └── factory.py         # VectorDBFactory
├── llm/                    # LLM client implementations
├── config/                 # Settings, logging, embedding config
├── web/                    # Frontend (static/, templates/)
└── scripts/                # Utility and startup scripts
```

### Key Patterns
- **Factory Pattern**: `VectorDBFactory.create_client()`, `LLMFactory.create_client()`
- **Interface-based**: All DB/LLM clients implement abstract base classes
- **Async I/O**: Use `async/await` for all I/O operations
- **Settings singleton**: `from config.settings import settings`

## Configuration

All config via environment variables in `config/.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | 0.0.0.0 | Server host |
| `PORT` | 8000 | Server port |
| `DEBUG` | False | Debug mode |
| `VECTOR_DB_TYPE` | chroma | Vector DB (chroma/milvus) |
| `CHROMA_HOST` | localhost | ChromaDB host |
| `CHROMA_PORT` | 8000 | ChromaDB port |
| `OPENAI_API_KEY` | - | LLM API key |
| `ADMIN_USER_NAME` | admin | Web UI username |
| `ADMIN_PASSWORD` | KUMI-admin-123456 | Web UI password |

## Important Notes

- **No linting config**: No ruff/black/flake8 configured. Follow existing code style.
- **Bilingual codebase**: Comments may be in Chinese or English.
- **ChromaDB required**: Must be running on port 8081 for vector operations.
- **Embedding service**: Either configure cloud API in `embedding_providers.yaml` or run local service.

## Common Tasks

### Add new API endpoint
1. Create Pydantic models in `api/models.py`
2. Add route handler in appropriate `api/*.py` file
3. Implement business logic in `services/*.py`
4. Register router in `main.py`

### Add new vector DB support
1. Implement `VectorDBInterface` from `vector_db/base.py`
2. Add to `VectorDBFactory` in `vector_db/factory.py`
3. Add config options in `config/settings.py`

### Add new LLM provider
1. Implement base class from `llm/base.py`
2. Add to LLM factory
3. Configure via environment variables
