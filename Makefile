SHELL := /bin/bash
ROOT := $(CURDIR)
MICROMAMBA := $(ROOT)/.tools/micromamba
SIM_PREFIX := $(ROOT)/.envs/simulation
PERCEPTION_PYTHON := $(ROOT)/.venvs/perception/bin/python
RUN_SMOKE := $(ROOT)/runs/verify_smoke

.PHONY: verify unit controller camera capture detector evaluator

verify: unit controller camera capture detector evaluator

unit:
	$(MICROMAMBA) run -p $(SIM_PREFIX) python -m unittest discover -s tests -v

controller:
	$(MICROMAMBA) run -p $(SIM_PREFIX) go2-ovd validate-controller --repeats 1

camera:
	$(MICROMAMBA) run -p $(SIM_PREFIX) go2-ovd validate-camera

capture:
	$(MICROMAMBA) run -p $(SIM_PREFIX) go2-ovd capture --config configs/episodes/smoke.yaml --output $(RUN_SMOKE)

detector:
	HF_HOME=$(ROOT)/.cache/huggingface HF_HUB_OFFLINE=1 $(PERCEPTION_PYTHON) -m go2_ovd.cli detect --config configs/detectors/grounding_dino_tiny.yaml --frames $(RUN_SMOKE) --device mps

evaluator:
	$(PERCEPTION_PYTHON) -m unittest tests.test_evaluation -v
	$(PERCEPTION_PYTHON) -m go2_ovd.cli evaluate --run $(RUN_SMOKE) --config configs/evaluation/default.yaml
