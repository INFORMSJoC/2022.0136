[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# [Control of Dual-Sourcing Inventory Systems using Recurrent Neural Networks](https://doi.org/10.1287/ijoc.2022.0136)

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [General Public License v2.0](LICENSE).

The software in this repository is a snapshot of the software that was used in the research reported 
on in the paper [Control of Dual-Sourcing Inventory Systems using Recurrent Neural Networks](https://doi.org/10.1287/ijoc.2022.0136) by Lucas Böttcher, Thomas Asikis, and Ioannis Fragkos.

## Cite

To cite the contents of this repository, please cite both the paper and this repository, using their respective DOIs.

https://doi.org/10.1287/ijoc.2022.0136

https://doi.org/10.1287/ijoc.2022.0136.cd

Below is the BibTex for citing this snapshot of the respoitory.

```
@article{bottcher2023control,
  author =        {B{\"o}ttcher, Lucas and Asikis, Thomas and Fragkos, Ioannis},
  publisher =     {INFORMS Journal on Computing},
  title =         {Control of Dual-Sourcing Inventory Systems using Recurrent Neural Networks},
  year =          {2023},
  doi =           {10.1287/ijoc.2022.0136.cd},
  note =          {Available for download at \url{https://github.com/INFORMSJoC/2022.0136}},
}  
```

## Description

This project provides implementations of neural networks that control single and dual sourcing problems (with and without empirical demand data). 

We summarize the main model parameters below.

| parameter | type    | description                                                   |
| --------- | ------- | --------------------------------------------------------------|
| `ce`      | int     | expedited order cost (per unit)                               |
| `cr`      | int     | regular order cost (per unit)                                 |
| `fe`      | int     | fixed expedited order cost (per unit)                         |
| `fr`      | int     | fixed regular order cost (per unit)                           |
| `h`       | int     | holding cost (per unit)                                       |
| `b`       | int     | shortage cost (per unit)                                      |
| `le`      | int     | expedited order lead time                                     |
| `lr`      | int     | regular order lead time                                       |
| `T`       | int     | number of simulation periods                                  |

<div align="center">
<img width="800" src="optimization_schematic.png" alt="neural net schematic">
</div>

The above figure shows a schematic of solving discrete-time stochastic control problems with recurrent neural networks.

## Data

The directory `data/cdi_nnc/` contains data on dual sourcing costs associated with both capped dual index (CDI) and neural network policies. It also contains additional data on neural-network training runtimes and different data visualization scripts.

In the directory `data/MSOM_data/`, we store the empirical demand data taken from Manary, Matthew P., and Sean P. Willems. "Data set: 187 weeks of customer forecasts and orders for microprocessors from intel corporation." Manufacturing & Service Operations Management 24.1 (2022): 682-689.

## Code

The directory `sourcing_models/` contains implementations of different dual sourcing heuristics (single index, dual index, capped-dual index, tailored-base surge). The files `sourcing_models/recursion_numba.py` and `sourcing_models/recursion_numba_state_output.py` provide dynamic program implementations with and without state output, respectively. We use the high-performance Python compiler Numba to speed-up the dynamic programming iterations.

Trained neural networks are provided in `sourcing_models/trained_neural_nets`. These files can be used in transfer-learning tasks.

Different neural-network controllers and inventory dynamics implementations (in pytorch) are stored in `neural_control/`.

An optimization example that uses real-world data is available in `MSOM_data/`. 

## Ongoing Development

This code is being developed on an on-going basis at the author's
[Gitlab site](https://gitlab.com/ComputationalScience/inventory-optimization/).

