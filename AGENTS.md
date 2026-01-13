# AGENTS.md - KUMI Knowledge Base Testing Platform

This file contains essential information for agentic coding agents working in the KUMI repository.

## Project Overview

KUMI is a visual knowledge base testing platform designed for rapid evaluation of knowledge base effectiveness. It provides automated testing capabilities for embedding and LLM systems, with flexible knowledge base construction and management features.

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment (Python 3.12)
uv venv --python 3.12
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Copy configuration files
cp config/.env.example config/.env
cp config/embedding_providers.yaml.example config/embedding_providers.yaml
```

### Running the Application
```bash
# Start development server
uv run scripts/start_dev.py
# Alternative: python main.py

# Access web interface
http://127.0.0.1:8000/web
# Default credentials: admin / KUMI-admin-123456
```

### Testing Commands
```bash
# Run specific test files (no comprehensive test suite found)
python services/testqa_service.py  # Test QA service functionality
python tests/test_RAG/similarity_service.py  # Test similarity calculations
```

### External Services
```bash
# Start ChromaDB (default vector database)
chroma run --path storage/testdb --port 8081 --host 127.0.0.1

# Start embedding service (if using local models)
cd scripts
uv run embedding_serve.py --model_path ./Qwen3-Embedding-0.6B --model_name Qwen3-Embedding-0.6B
```

## Code Style Guidelines

### Python Code Style
- **Language**: Python 3.12+
- **Framework**: FastAPI for web APIs
- **Package Manager**: uv (preferred) or pip
- **Configuration**: Environment variables via .env files

### Import Organization
```python
# Standard library imports first
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Third-party imports
import pandas as pd
import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Local imports
from config.settings import settings
from services.knowledge_service import KnowledgeService
from vector_db.factory import VectorDBFactory
```

### Class and Function Naming
- **Classes**: PascalCase (e.g., `KnowledgeService`, `VectorDBInterface`)
- **Functions/Methods**: snake_case (e.g., `create_collection`, `search_knowledge`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_MODEL`, `MAX_FILE_SIZE`)
- **Private methods**: prefix with underscore (e.g., `_load_config`, `_validate_input`)

### Type Hints
Always use type hints for function signatures and class attributes:
```python
from typing import List, Dict, Any, Optional, Union

def search_knowledge(
    query: str,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Search knowledge base with given query."""
    pass
```

### Error Handling
- Use structured exception handling with specific exception types
- Log errors appropriately using the logging module
- Return meaningful error responses in API endpoints
```python
import logging

logger = logging.getLogger(__name__)

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

### API Design Patterns
- Use FastAPI with Pydantic models for request/response validation
- Follow RESTful conventions for endpoint design
- Include proper HTTP status codes and error handling
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

## Architecture Patterns

### Service Layer Architecture
- **API Layer** (`api/`): FastAPI route handlers and request/response models
- **Service Layer** (`services/`): Business logic and orchestration
- **Data Layer** (`vector_db/`, `llm/`): Database and external service clients
- **Config Layer** (`config/`): Configuration management and settings

### Factory Pattern for Clients
Use factory pattern for creating database and LLM clients:
```python
# In factory.py
class VectorDBFactory:
    @staticmethod
    def create_client() -> VectorDBInterface:
        db_type = settings.vector_db_type
        if db_type == "chroma":
            return ChromaClient()
        elif db_type == "milvus":
            return MilvusClient()
        else:
            raise ValueError(f"Unsupported DB type: {db_type}")
```

### Configuration Management
- Use environment variables for all configuration
- Centralize settings in `config/settings.py`
- Provide sensible defaults for all options
- Use property methods for derived configurations

## File Organization

### Directory Structure
```
KUMI/
├── api/                    # FastAPI route handlers
│   ├── models.py          # Pydantic models
│   ├── knowledge.py       # Knowledge base APIs
│   └── web.py             # Web interface APIs
├── services/              # Business logic layer
│   ├── knowledge_service.py
│   └── embedding_service.py
├── vector_db/             # Vector database clients
│   ├── base.py            # Interface definitions
│   ├── chroma_client.py   # ChromaDB implementation
│   └── factory.py         # Client factory
├── llm/                   # LLM client implementations
│   ├── base.py            # Interface definitions
│   ├── openai_client.py   # OpenAI implementation
│   └── factory.py         # Client factory
├── config/                # Configuration management
│   ├── settings.py        # Main settings class
│   └── embedding_config.py # Embedding provider config
├── web/                   # Frontend assets
│   ├── static/           # CSS, JS, images
│   └── templates/        # HTML templates
└── scripts/              # Utility and startup scripts
```

### Module Conventions
- Each module should have a clear, single responsibility
- Use `__init__.py` to expose public interfaces
- Include docstrings for all public functions and classes
- Follow the dependency injection pattern for configuration

## Testing Guidelines

### Test Organization
- Unit tests should be in the same module or a dedicated `tests/` directory
- Integration tests should test service interactions
- Use descriptive test names that explain the scenario

### Test Data Management
- Store test data in `dataset/` directory
- Use configuration files for test parameters
- Clean up test data after test completion

## Database and External Services

### Vector Database Support
- **Primary**: ChromaDB (default for cross-platform deployment)
- **Secondary**: Milvus (for production-scale deployments)
- Use factory pattern for database client creation

### LLM Integration
- Support for OpenAI-compatible APIs
- Configurable through environment variables
- Use interface-based design for provider flexibility

### Embedding Services
- Support for local and cloud embedding providers
- Configuration via `embedding_providers.yaml`
- Default to local Qwen3-Embedding-0.6B model

## Security Considerations

### Authentication
- Web interface uses basic auth (configurable via .env)
- API keys for external service integration
- Token-based verification for direct connections

### Data Protection
- Sanitize all user inputs
- Validate file uploads and size limits
- Use proper error handling to prevent information leakage

## Performance Guidelines

### Async/Await Usage
- Use async/await for I/O operations
- Implement proper concurrency for external service calls
- Use connection pooling for database operations

### Caching Strategy
- Cache static files for 30 days
- Consider caching frequent embedding computations
- Use appropriate cache headers for API responses

## Logging and Monitoring

### Logging Configuration
- Use Python's logging module
- Configure log levels via environment variables
- Include structured logging for API requests/responses

### Health Checks
- Implement health check endpoints for all services
- Monitor external service connectivity
- Log startup and shutdown events

## Deployment Notes

### Environment Variables
All configuration should be externalized via environment variables. Key variables include:
- `HOST`, `PORT`: Server binding
- `DEBUG`: Development mode flag
- `VECTOR_DB_TYPE`: Database selection
- `OPENAI_API_KEY`: LLM service access
- `ADMIN_USER_NAME`, `ADMIN_PASSWORD`: Web interface credentials

### Service Dependencies
- ChromaDB (port 8081 by default)
- Optional: Local embedding service (port 8504)
- Optional: External LLM APIs

### Production Considerations
- Use proper reverse proxy (nginx) for static file serving
- Implement rate limiting for API endpoints
- Set up monitoring and alerting for service health