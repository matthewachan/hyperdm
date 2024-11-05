# hyperdm

This is the official codebase for the NeurIPS 2024 paper ["Estimating Epistemic and Aleatoric Uncertainty with a Single Model"](https://nips.cc/virtual/2024/poster/94833).

### Abstract

> "Estimating and disentangling epistemic uncertainty, uncertainty that is reducible with more training data, and aleatoric uncertainty, uncertainty that is inherent to the task at hand, is critically important when applying machine learning to high-stakes applications such as medical imaging and weather forecasting. Conditional diffusion models' breakthrough ability to accurately and efficiently sample from the posterior distribution of a dataset now makes uncertainty estimation conceptually straightforward: One need only train and sample from a large ensemble of diffusion models. Unfortunately, training such an ensemble becomes computationally intractable as the complexity of the model architecture grows. In this work we introduce a new approach to ensembling, hyper-diffusion models (HyperDM), which allows one to accurately estimate both epistemic and aleatoric uncertainty with a single model. Unlike existing single-model uncertainty methods like Monte-Carlo dropout and Bayesian neural networks, HyperDM offers prediction accuracy on par with, and in some cases superior to, multi-model ensembles. Furthermore, our proposed approach scales to modern network architectures such as Attention U-Net and yields more accurate uncertainty estimates compared to existing methods. We validate our method on two distinct real-world tasks: x-ray computed tomography reconstruction and weather temperature forecasting."

# Usage

### Dependencies

Using Python 3.11.9, please install dependencies by running:

```sh
$ pip install -r requirements.txt
```

### Toy Experiment

We include `Makefile` build targets for generating toy experiment figures.

- To visualize HyperDM results, run `make toy_result.pdf`.
- To visualize deep ensemble results, run `make toy_baseline.pdf`.

|   `toy_result.pdf`   |     `toy_baseline.pdf`     |
| :------------------: | :------------------------: |
| ![](img/hyperdm.png) | ![](img/deep_ensemble.png) |

**Note:** As mentioned in our paper, AU is unreliable (and should be disregarded) when EU is high.

# Citation

Please cite our work using this BibTeX snippet:

```
@article{chan2024hyper,
  title={Estimating Epistemic and Aleatoric Uncertainty with a Single Model},
  author={Chan, Matthew A and Molina, Maria J and Metzler, Christopher A},
  journal={arXiv preprint arXiv:2402.03478},
  year={2024}
}
```
