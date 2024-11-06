SHELL := /bin/bash
export PYTHONPATH := $(PYTHONPATH):$(shell pwd)

.PHONY: clean

era5_result.pdf: src/test.py era5_model.pt
	time python src/test.py \
		--seed 1 \
		--dataset "era5" \
		--dataset_size 1000 \
		--image_size 256 \
		--checkpoint era5_model.pt \
		--M 10 \
		--N 100 \
		--diffusion_steps 1000 \
		--hyper_net_dims 1 24 24 24 24 24

era5_model.pt: src/train.py
	time python src/train.py \
		--seed 1 \
		--dataset "era5" \
		--dataset_size 1000 \
		--image_size 256 \
		--checkpoint era5_model.pt \
		--num_epochs 50 \
		--lr 1e-4 \
		--batch_size 8 \
		--diffusion_steps 1000 \
		--hyper_net_dims 1 24 24 24 24 24

toy_result.pdf: src/test.py toy_model.pt
	time python src/test.py \
		--seed 1 \
		--dataset "toy" \
		--dataset_size 10000 \
		--checkpoint toy_model.pt \
		--M 10 \
		--N 100 \
		--diffusion_steps 1000 \
		--hyper_net_dims 1 8 8 8 8 8

toy_model.pt: src/train.py
	time python src/train.py \
		--seed 1 \
		--dataset "toy" \
		--dataset_size 10000 \
		--checkpoint toy_model.pt \
		--num_epochs 100 \
		--lr 1e-3 \
		--batch_size 64 \
		--diffusion_steps 1000 \
		--hyper_net_dims 1 8 8 8 8 8

src/train.py: src/hyperdm.py data/era5.py data/toy.py model/mlp.py model/unet.py

src/test.py: src/hyperdm.py data/era5.py data/toy.py model/mlp.py model/unet.py

src/hyperdm.py: model/mlp.py

toy_baseline.pdf: src/toy_baseline.py data/toy.py src/hyperdm.py model/mlp.py
	time python src/debug.py

clean:
	rm -rf *.pt *.pdf