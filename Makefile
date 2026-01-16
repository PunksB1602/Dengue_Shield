
.PHONY: setup splits train demo clean

setup:
	pip install -r requirements.txt

splits:
	python scripts/make_splits_wingbeats.py

train:
	PYTHONPATH=. python -m src.train.train_main

demo:
	streamlit run app/app.py

clean:
	python scripts/clean.py
