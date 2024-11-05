SHELL := /bin/bash
export PYTHONPATH := $(PYTHONPATH):$(shell pwd)

.PHONY: clean

toy_result.pdf: src/test.py toy_model.pt
	time python src/test.py \
		--seed 1 \
		--dataset "toy" \
		--dataset_size 10000 \
		--checkpoint toy_model.pt \
		--M 10 \
		--N 100 \
		--diffusion_steps 1000 \
		--hyper_net_input_dim 1

toy_model.pt: src/train.py data/toy.py src/hyperdm.py
	time python src/train.py \
		--seed 1 \
		--dataset "toy" \
		--dataset_size 10000 \
		--checkpoint toy_model.pt \
		--num_epochs 100 \
		--lr 1e-3 \
		--batch_size 64 \
		--diffusion_steps 1000 \
		--hyper_net_input_dim 1

toy_baseline.pdf: src/toy_baseline.py data/toy.py src/hyperdm.py
	time python src/debug.py

clean:
	rm -rf *.pt *.pdf