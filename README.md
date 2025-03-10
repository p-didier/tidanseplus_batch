# Improved Distributed Adaptive Node-Specific Signal Estimation for Topology-Unconstrained Wireless Acoustic Sensor Networks

This repository contains the code files that were used to generate the results reported in the paper ``Improved Distributed Adaptive Node-Specific Signal Estimation for Topology-Unconstrained Wireless Acoustic Sensor Networks'' by Paul Didier, Toon van Waterschoot, Simon Doclo, Jörg Bitzer, and Marc Moonen, submitted to the 33rd European Signal Processing Conference (EUSIPCO 2025) held in Isola delle Femmine – Palermo – Italy, on September 8-12, 2025.

## Abstract

This paper addresses the challenge of distributed adaptive node-specific signal estimation (DANSE) in topology-unconstrained wireless acoustic sensor networks (WASNs) where sensor nodes exchange only fused versions of their local signals. The topology-independent DANSE (TI-DANSE) algorithm has previously been presented to handle non-fully connected and time-varying WASNs, however, its slow convergence limits its applicability. To address this limitation, the TI-DANSE<sup>+</sup> to algorithm is proposed, which allows updating nodes to separately use partial in-network sums from neighbors in their local estimation problem, thus increasing the available degrees of freedom and accelerating convergence with respect to TI-DANSE. Additionally, a tree-pruning strategy is proposed to further increase convergence speed. TI-DANSE<sup>+</sup> converges as fast as the DANSE algorithm in fully connected WASNs while reducing transmit power usage. The advantageous convergence properties of TI-DANSE<sup>+</sup> are demonstrated in numerical simulations.

### Requirements

Required Python packages are listed in `requirements.txt`.

## Running the Experiment

To run the main experiment, execute `main.py` with the configuration specified in `main_cfg.yaml`:

```
python main.py --config main_cfg.yaml
```

After running the experiment, you can post-process the output using `pp.py` with the path to the simulated results specified:

```
python pp.py --path "path/to/your/data.pkl"
```

## Cite us

If you find this code helpful and want to use it in your own research, please cite the following paper:
* Plain text:
```
P. Didier, T. van Waterschoot, S. Doclo, J. Bitzer, and M. Moonen, "Improved Distributed Adaptive Node-Specific Signal Estimation for Topology-Unconstrained Wireless Acoustic Sensor Networks", submitted to the 33rd European Signal Processing Conference (EUSIPCO 2025), 2025, pp. 1--5.
```
* Bibtex
```
@INPROCEEDINGS{didierImproved2025,
  author={Didier, Paul and van Waterschoot, Toon, and Doclo, Simon and Bitzer, Joerg and Moonen, Marc},
  booktitle={2025 European Signal Processing Conference (EUSIPCO)},
  title={Improved Distributed Adaptive Node-Specific Signal Estimation for Topology-Unconstrained Wireless Acoustic Sensor Networks},
  year={2025},
  volume={},
  number={},
  pages={1-5},
}
```

## License

```
Copyright (C) 2025 Paul Didier

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```
