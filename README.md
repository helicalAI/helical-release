<div align="center">
  <img src="docs/assets/logo_name.png" alt="Logo" width="304" height="110">
</div>

![image](docs/assets/logo_name.png)

```{image} docs/assets/logo_name.png
:alt: "Logo"
:width: 400px
:align: center
```

## What is Helical ?

Helical provides a framework for and gathers state-of-the-art pre-trained bio foundation models on genomics and transcriptomics modalities.

Helical simplifies the entire application lifecycle when building with bio foundation models. You will be able to:
- Leverage the latest bio foundation models through our easy-to-use python package
- Run example notebooks on key downstream tasks from examples

We will update this repo on a bi-weekly basis with new models, benchmarks, modalities and functions - so stay tuned.
Let’s build the most exciting AI-for-Bio community together!

## Installation

We recommend installing Helical within a conda environment with the commands below (run them in your terminal) - this step is optional:
```
conda create --name helical-package python=3.11.8
conda activate helical-package
```
To install the Helical package, you can run the command below:
```
pip install --upgrade --force-reinstall git+https://github.com/helicalAI/helical.git
```


## Demo & Use Cases

To run examples, be sure to have installed the Helical package (see Installation) and that it is up-to-date.

You can look directly into the example folder above, look into our [documentation](https://helical.readthedocs.io/) for step-by-step guides or directly clone the repository using:
```
git clone https://github.com/helicalAI/helical.git
```
Within the `example` folder, open the notebook of your choice. We recommend starting with `Geneformer-vs-UCE.ipynb`

### RNA models:
- `Geneformer-vs-UCE.ipynb`: Zero-Shot Reference Mapping with Geneformer & UCE and compare the outcomes.
- Coming soon: new models such as scGPT, SCimilarity, scVI; benchmarking scripts; new use cases; others

### DNA models:
- Coming soon: new models such as Nucleotide Transformer; others

# Stuck somewhere ? Other ideas ?
We are eager to help you and interact with you. Reach out via rick@helical-ai.com. 
You can also open github issues here.

# Why should I use Helical & what to expect in the future?
If you are (or plan to) working with bio foundation models s.a. Geneformer or UCE on RNA and DNA data, Helical will be your best buddy! We provide and improve on:
- Up-to-date model library
- A unified API for all models
- User-facing abstractions tailored to computational biologists, researchers & AI developers
- Innovative use case and application examples and ideas
- Efficient data processing & code-base

We will continuously upload the latest model, publish benchmarks and make our code more efficient.

# Citation

Please use this BibTeX to cite this repository in your publications:

```
@misc{helical,
  author = {Maxime Allard, Benoit Putzeys, Rick Schneider, Mathieu Klop},
  title = {Helical Python Package},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/helicalAI/helical}},
}
