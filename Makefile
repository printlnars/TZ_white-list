PYTHON  ?= python3
VENV    := .venv
PY      := $(VENV)/bin/python
BACKEND ?= rules

.PHONY: install eval eval-greedy eval-ollama test clean

$(VENV)/.installed: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install -q -r requirements.txt
	touch $@

install: $(VENV)/.installed

eval: install            ## make eval BACKEND=rules|greedy|ollama|openai
	$(PY) eval.py --backend $(BACKEND) --json reports/eval_$(BACKEND).json

eval-greedy: install
	$(PY) eval.py --backend greedy --json reports/eval_greedy.json

eval-ollama: install
	$(PY) eval.py --backend ollama --json reports/eval_ollama.json

test: install
	$(PY) -m pytest -q

clean:
	rm -rf $(VENV) reports .pytest_cache
