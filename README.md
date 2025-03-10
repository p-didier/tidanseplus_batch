# EUSIPCO 2025 TI-DANSE+

This repository contains the code for the experiments presented in the EUSIPCO 2025 submission ``Improved Distributed Adaptive Node-Specific Signal Estimation for Topology-Unconstrained Wireless Acoustic Sensor Networks'' by Paul Didier, Toon van Waterschoot, Simon Doclo, Jörg Bitzer, and Marc Moonen.

## Getting Started

### Prerequisites

- Python 3.12
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/tidanseplus_batch.git
    ```
2. Navigate to the project directory:
    ```sh
    cd tidanseplus_batch
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Experiment

To run the main experiment, execute `main.py` with the configuration specified in `main_cfg.yaml`:

```sh
python main.py --config main_cfg.yaml
```

## Post-Processing

After running the experiment, you can post-process the output using `pp.py` with the path to the simulated results specified:

```sh
python pp.py --path "path/to/your/data.pkl"
```

## Repository Structure

- `main.py`: Script to run the main experiment.
- `main_cfg.yaml`: Configuration file for the main experiment.
- `pp.py`: Script for post-processing the experiment output.
- `requirements.txt`: List of required Python packages.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the contributors and the research community.

For any questions or issues, please open an issue on the repository.
