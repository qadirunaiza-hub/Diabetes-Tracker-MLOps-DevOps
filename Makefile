.PHONY: setup train serve docker-build docker-up docker-down test lint clean

setup:
	pip install -r requirements.txt

train:
	PYTHONPATH=. bash scripts/train.sh

serve:
	PYTHONPATH=. python src/api/app.py

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down -v

test:
	PYTHONPATH=. pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	flake8 src/ tests/ --max-line-length=100

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
