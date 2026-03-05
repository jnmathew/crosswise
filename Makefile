.PHONY: install setup clean run run-api run-web run-demo test help

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies (including test)
	uv sync --extra test
	cd web && npm install && cd ..

setup: ## Download data, build clue DB, clean up (~1.5GB peak, ~1GB after)
	bash scripts/setup_data.sh
	uv run python -c "from crosswise.solver.clue_database import ClueDatabase; ClueDatabase()"
	@echo ""
	@echo "Cleaning up intermediate files..."
	rm -rf data/sources
	@echo "Done. Kept: data/clues.db + data/wordlists/"

clean: ## Delete clue DB and wordlists (re-run make setup to restore)
	rm -f data/clues.db
	rm -rf data/wordlists
	@echo "Cleaned. Run 'make setup' to rebuild."

run: ## Start backend + frontend (Ctrl+C stops both)
	@trap "echo; echo Stopping...; kill 0" INT TERM; \
		echo "Starting API + Web (Ctrl+C to stop both)"; \
		uv run python -m crosswise.api.server 2>&1 | sed 's/^/[API] /' & \
		(cd web && npm run dev) 2>&1 | sed 's/^/[WEB] /' & \
		wait

run-api: ## Start FastAPI backend
	uv run python -m crosswise.api.server

run-web: ## Start React dev server
	cd web && npm run dev

test: ## Run all tests
	uv run pytest tests/ -v

run-demo: ## Try the app with a sample puzzle (no API keys needed)
	@mkdir -p web/public/puzzles
	@cp assets/demo/sample_puzzle.json web/public/puzzles/sample.json
	@echo "Starting Crosswise — open http://localhost:5173"
	@$(MAKE) run
