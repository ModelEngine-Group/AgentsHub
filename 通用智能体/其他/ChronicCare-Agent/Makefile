.PHONY: test coverage lint format-check quality real-integration real-coverage

test:
	python3 -m pytest -q

coverage:
	python3 -m pytest \
		--cov=analysis \
		--cov=kg \
		--cov=mcp_adapter \
		--cov=orchestration \
		--cov=runtime_common \
		--cov=tool_server \
		--cov=visualization \
		--cov-report=term-missing \
		-q

lint:
	python3 -m ruff check .

format-check:
	python3 -m ruff format --check tests

real-integration:
	CHRONICCARE_REAL_INTEGRATION=1 python3 -m pytest -m real_integration -ra

real-coverage:
	CHRONICCARE_REAL_INTEGRATION=1 $(MAKE) coverage

quality: lint format-check test

