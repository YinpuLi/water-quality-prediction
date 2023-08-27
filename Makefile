clean:
	rm -rf results/shap/*
	rm -rf results/shap_on_test/*
	rm -rf results/*.joblib

update:
	python scripts/run_hyper_tuning.py
	python scripts/run_shap.py