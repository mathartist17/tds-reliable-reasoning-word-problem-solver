## Notes

- Build command in render
`uv sync --frozen && uv cache prune --ci`
- Start command in render
`uv run uvicorn main:app --host 0.0.0.0 --port $PORT --reload`
