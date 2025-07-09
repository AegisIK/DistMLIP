<p align="center">
  <img src="assets/logo.png" width="400" alt="Alt text">
</p>

# DistMLIP: A Distributed Inference Library for Fast, Large Scale Atomistic Simulation

## About

DistMLIP is an easy-to-use, efficient library for running graph-parallel, multi-GPU simulations using popular machine learning interatomic potentials (MLIPs).

DistMLIP currently supports **zero redundancy multi-GPU inference** for MLIPs using graph parallelism. Unlike space partitioning via LAMMPS, there is no redundant calculation being performed.

DistMLIP currently supports the following models:

- CHGNet ([MatGL](https://github.com/materialsvirtuallab/matgl))
- TensorNet ([MatGL](https://github.com/materialsvirtuallab/matgl))
- MACE ([MACE](https://github.com/ACEsuit/mace))
- eSEN ([fairchem](https://github.com/facebookresearch/fairchem)) (coming soon)

> 🚧 **This project is under active development**  
> If you see a bug, please raise an issue or notify us. All messages will, at the latest, be responded to within 12 hours.  

## Getting Started

1. Install PyTorch: https://pytorch.org/get-started/locally/

2. Install DGL here (if using the MatGL models): https://www.dgl.ai/pages/start.html

3. Install DistMLIP from pip
```
TODO
```

or from source:

```bash
git clone git@github.com:AegisIK/DistMLIP.git
cd DistMLIP
pip install -e .[matgl] # If you're using the dgl models
pip install -e .[mace] # If you're using the mace models
python setup.py build_ext --inplace
```

## Running distributed inference
DistMLIP is a wrapper library designed to inherit from other models in order to provided distributed inference support. As a result, all features of the original package (whether it's MatGL or MACE) will still work. View one of our example notebooks [here](./examples) to get started. 



> Although it is supported via DistMLIP, it is recommended to finetune your model using the original model library before loading your model into DistMLIP via `from_existing` and running distributed inference.

Currently only single node inference is supported. Multi-machine inference is future work.


## Roadmap

- [x] Distributing CHGNet
- [x] Distributing TensorNet
- [X] Distributing MACE
- [ ] Multi-machine inference
- [ ] More works coming soon! 

## Citation

If you use DistMLIP in your research, please cite our paper:
<pre>@misc{han2025distmlipdistributedinferenceplatform,
      title={DistMLIP: A Distributed Inference Platform for Machine Learning Interatomic Potentials}, 
      author={Kevin Han and Bowen Deng and Amir Barati Farimani and Gerbrand Ceder},
      year={2025},
      eprint={2506.02023},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2506.02023}, 
}</pre>

## Parallelizing a New Model
If you would like to contribute or want us to parallelize your model, please either raise an issue or email kevinhan@cmu.edu.

## Contact Us
- If you have any questions, feel free to raise an issue on this repo.
- If you have any feature requests, please raise an issue on this repo.
- For collaborations and partnerships, please email kevinhan@cmu.edu.
- All requests/issues/inquiries will receive a response within 6-12 hours.

