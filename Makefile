.PHONY: setup clean run run-api run-web run-demo help

PYTHON := .venv/bin/python3

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Download data, build clue DB, clean up (~1.7GB peak, ~1.2GB after)
	bash scripts/setup_data.sh
	$(PYTHON) -c "from crosswise.solver.clue_database import ClueDatabase; ClueDatabase()"
	@echo ""
	@echo "Cleaning up intermediate files..."
	rm -rf data/xd data/crosswordqa
	@echo "Done. Kept: data/clues.db + data/wordlists/"

clean: ## Delete clue DB and wordlists (re-run make setup to restore)
	rm -f data/clues.db
	rm -rf data/wordlists
	@echo "Cleaned. Run 'make setup' to rebuild."

run: ## Start backend + frontend (Ctrl+C stops both)
	@trap "echo; echo Stopping...; kill 0" INT TERM; \
		echo "Starting API + Web (Ctrl+C to stop both)"; \
		$(PYTHON) -m crosswise.api.server 2>&1 | sed 's/^/[API] /' & \
		(cd web && npm run dev) 2>&1 | sed 's/^/[WEB] /' & \
		wait

run-api: ## Start FastAPI backend
	$(PYTHON) -m crosswise.api.server

run-web: ## Start React dev server
	cd web && npm run dev

run-demo: ## Verify solver can load DB and generate candidates
	$(PYTHON) -c "from crosswise.solver.clue_database import ClueDatabase; db = ClueDatabase(); r1 = db.lookup_by_clue('Prefix with space', 4); r2 = db.lookup_by_pattern('_A_'); print(f'DB loaded — 10M+ entries'); print(f'  clue lookup [Prefix with space, 4]: {r1[:3]}'); print(f'  pattern match [_A_]: {r2[:5]}'); db.close(); print('Demo complete.')"
