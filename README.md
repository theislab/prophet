[![License: MIT][mit-shield]][mit]
# Prophet

Prophet is a transformer-based regression model that predicts cellular responses by decomposing experiments into cell state, treatment, and functional readout, leveraging extensive screening datasets and scalability to significantly reduce the number of required experiments and identify effective treatments.


## Installation
```
git clone https://github.com/theislab/prophet.git
cd prophet
pip install -e .
```

## Usage

Model checkpoints and input embeddings can be downloaded [here](https://huggingface.co/datasets/aletlvl/Prophet_v1/tree/main) and [here](https://data.mendeley.com/datasets/g7z3pw3bfw). Examples for how to query the results of various experiments can be found at [tutorial.ipynb](https://github.com/theislab/prophet/blob/main/tutorial.ipynb).

If you have used our work in your research, please cite our [preprint](https://www.biorxiv.org/content/10.1101/2024.08.12.607533v2).

[mit]: https://opensource.org/licenses/MIT
[mit-image]: https://img.shields.io/badge/License-MIT-yellow.svg
[mit-shield]: https://img.shields.io/badge/License-MIT-yellow.svg
