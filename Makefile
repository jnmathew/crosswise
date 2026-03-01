.PHONY: setup build-db run run-api run-web run-demo help

PYTHON := .venv/bin/python3

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Download all data sources
	bash scripts/setup_data.sh

build-db: ## Build SQLite clue database from raw sources
	$(PYTHON) -c "from src.solver.clue_database import ClueDatabase; ClueDatabase()"

run: ## Start backend and frontend (requires two terminals)
	@echo "Terminal 1:  make run-api"
	@echo "Terminal 2:  make run-web"

run-api: ## Start FastAPI backend
	$(PYTHON) -m src.api.server

run-web: ## Start React dev server
	cd web && npm run dev

run-demo: ## Verify solver can load DB and generate candidates
	$(PYTHON) -c "from src.solver.clue_database import ClueDatabase; db = ClueDatabase(); r1 = db.lookup_by_clue('Prefix with space', 4); r2 = db.lookup_by_pattern('_A_'); print(f'DB loaded — 10M+ entries'); print(f'  clue lookup [Prefix with space, 4]: {r1[:3]}'); print(f'  pattern match [_A_]: {r2[:5]}'); db.close(); print('Demo complete.')"
