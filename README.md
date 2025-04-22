<p align="center">
  <img src="assets/logo.png" width="400" alt="Alt text">
</p>

# DistMLIP: A Distributed Inference Platform for Fast, Large Scale Atomistic Simulation

## About

DistMLIP is an easy-to-use, efficient library for running large-scale multi-GPU simulations using popular machine learning interatomic potentials (MLIPs).

DistMLIP currently supports zero redundancy multi-GPU inference for MLIPs using graph parallelism. Unlike space partitioning via LAMMPS, there is no redundant calculation being performed.

DistMLIP currently supports the following models:

- CHGNet ([MatGL](https://github.com/materialsvirtuallab/matgl))
- TensorNet ([MatGL](https://github.com/materialsvirtuallab/matgl))
- MACE (future work planned)

**Performance Benchmark**
TODO (going to redirect the user to benchmark folder and show a FPIS benchmark + bring up some figures from the paper)

## Getting Started

Install DistMLIP from pip or from source:

```bash
git clone git@github.com:AegisIK/DistMLIP.git
cd DistMLIP
pip install -e .
python setup.py build_ext --inplace
```

Convert your model into its DistMLIP distributed version:
TODO

Enable distributed mode:
TODO

Perform inference:
TODO

> Although it is supported via DistMLIP, it is recommended to finetune your model using the original model library before loading your model into DistMLIP via `from_existing` and running distributed inference.

Currently only single node inference is supported. Multi-machine inference is future work.

View one of our example notebooks [here](./examples) to get started. DistMLIP is a mix-in library designed to inherit other models. As a result, all features of the original package (whether it's MatGL or MACE) should still work.

## Roadmap

- [x] Distributing CHGNet
- [x] Distributing TensorNet
- [ ] Distributing MACE
- [ ] Multi-machine inference
- [ ] LAMMPS integration

## Citation

If you use DistMLIP in your research, please cite our paper:
TODO

## Contact Us

- If you have any feature requests, please raise an issue on this repo.
- If you would like to contribute or want us to parallelize your model, please email kevinhan@cmu.edu.
- For collaborations and partnerships, please email kevinhan@cmu.edu.
