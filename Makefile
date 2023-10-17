clean:
	rm -rf results/shap/*
	rm -rf results/shap_on_test/*
	rm -rf results/*.joblib
	rm -rf results/summary/*

update:
	python scripts/run_hyper_tuning.py
#	python scripts/run_shap.py
	python scripts/inference.py