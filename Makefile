.PHONY: install test eval charts report run
install: ; pip install -r requirements.txt
test:    ; python -m pytest -q
eval:    ; python -m evals.run_evals
charts:  ; python scripts/make_charts.py
report:  ; python scripts/make_eval_report.py
run:     ; uvicorn app.api:api --reload
