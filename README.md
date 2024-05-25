# Beam Analysis Project

## Overview

This project aims to analyze the behavior of a cantilever beam using the Finite Element Method (FEM). The project includes both static and dynamic analyses, and generates visualizations of the beam's behavior under various loading conditions.

## Project Structure

```
Beam_Analysis/
├── input/
│   ├── structure_data/
│   └── material_data/
├── src/
│   ├── __init__.py
│   ├── fem.py
│   ├── beam.py
│   ├── visualization.py
│   ├── utils.py
│   └── main.py
├── tests/
│   ├── test_fem.py
│   ├── test_beam.py
│   ├── test_visualization.py
│   └── test_utils.py
├── output/
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

Ensure you have Python 3.x and Anaconda installed. You can check your Python version using:

```sh
python --version
```

### Installation

1. Clone the repository:

```sh
git clone https://github.com/HNU-WYH/Beam_Analysis
cd Beam_Analysis
```

2. Create a virtual environment and install the required packages:

```sh
conda create --name <env_name> --file requirement.txt
```

3. activate the created environment:

```sh
activate <env_name>
```

## Usage

### Running the Analysis

The main script to run the analysis is `main.py`. You can execute it using:

```sh
python src/main.py
```

This script will:

- Define the beam properties and discretize it.
- Assemble the FEM matrices.
- Apply loads and boundary conditions.
- Solve both static and dynamic problems.
- Generate visualizations of the results.

### Project Modules

- **fem.py**: Contains the `FEM` class for assembling matrices, applying boundary conditions, and solving the equations.
- **beam.py**: Contains the `Beam` class for defining beam properties and discretizing it.
- **visualization.py**: Contains the `Visualization` class for plotting static and dynamic results.
- **utils.py**: Contains utility functions for creating load vectors and boundary conditions.
- **main.py**: The main script that orchestrates the entire process.

### Testing

Tests are located in the `tests/` directory. To run the tests, you can use:

```sh
pytest tests/
```

This will execute all the test cases to ensure the code is working as expected.

## Example

Here is an example of how to define a beam, apply loads, and visualize the results:

```
# Comming soon
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for review.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
