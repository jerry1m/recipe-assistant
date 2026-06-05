.PHONY: run test lint benchmark clean install

# ── 安装 ──
install:
	pip install -r requirements.txt

# ── 运行 ──
run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-demo:
	python -m examples.agent_trace_demo

# ── 测试 ──
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

coverage:
	pytest tests/ --cov=src --cov-report=term --cov-report=xml

smoke:
	python -c "from src.api.schemas import AskRequest; print('Schema OK')"

# ── Lint ──
lint:
	ruff check src/

format:
	ruff format src/

# ── 评测 ──
benchmark:
	python -m src.eval.benchmark

# ── Docker ──
docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

# ── 索引重建 ──
rebuild-index:
	SKIP_DOWNLOAD=1 MODEL_PATH=/home/luguanghui/PRNet/REMOTE-main/ROMOTE_code/models/all-MiniLM-L6-v2 python3 scripts/ingest_real_data.py

# ── 清理 ──
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf *.db
