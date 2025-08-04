# 3D Lattice Structure Generation and Visualization

Molly Carton, Jiayi Wu

This repository contains the code for generating and visualizing 3D lattice structures as found in `Design framework for programmable three-dimensional woven metamaterials' (Molly Carton, James Utama Surjadi, Bastien F. G. Aymon, Carlos M. Portela., 2025). The project aims to create complex woven lattice geometries, export them as STL and CSV files, and visualize them interactively.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Package Dependencies](#package-dependencies)
- [Code Structure](#code-structure)
- [Demo](#demo)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project is focused on generating woven lattice structures using mathematical models and algorithms. The code is designed to create intricate 3D shapes that can be exported and used for various applications, including 3D printing and computational simulations. The generated structures are represented as a mesh of interconnected nodes and struts, providing both structural integrity and aesthetic complexity.

## Features

- **Lattice Generation:** Create various types of 3D lattice structures based on specified parameters.
- **Mesh Export:** Export the generated structures to STL files for 3D printing or further processing.
- **CSV Export:** Save the lattice structure data as CSV files for analysis and documentation.
- **Visualization:** Interactive visualization of lattice structures.

## Installation

To set up the project, clone the repository and install the necessary dependencies using `pip`.

### Prerequisites

Ensure that you have Python 3.9 or greater installed on your system. If not, download and install it:

- [Python](https://www.python.org/downloads/)

This software has been tested on MacOS 12 and Windows 11.

### Clone the Repository

```bash
git clone https://github.com/mollyacarton/woven-lattice.git
cd woven-lattice
```

### Install Python Dependencies

You can install the required Python packages using the following command:

```bash
pip install -r requirements.txt
```

If the `requirements.txt` file is not present, you can manually install the dependencies:

```bash
pip install scipy==1.13.1 matplotlib==3.4.3 numpy==1.22.4 numpy-stl==3.1.1 
```

## Quick start

To start with the code, clone this repository and run lattice_main.py.

## Package Dependencies

The following packages are required to run the project:

- **[Scipy](https://www.scipy.org/):** 1.13.1
- **[Matplotlib](https://matplotlib.org/):** 3.4.3
- **[Numpy](https://numpy.org/):** 1.22.4
- **[Numpy-STL](https://pypi.org/project/numpy-stl/):** 3.1.1

These dependencies can be installed using the provided `pip` command. Installation should take no more than 5 minutes with a good connection. 

## Code Structure

Here is an overview of the project's code structure:

- **lattice_main.py**: Main script for generating lattice structures.
- **makeunitcell_func1.py**: Contains functions for creating unit cells and handling geometric calculations.

## Demo

To run the demo, run `woven_lattice.py` with the provided initial parameters. The demo will plot an interactive graphic of a 3x2x1 diamond woven lattice; in the working directory, it will save 1) a single .csv file, `wovenCSV_[date].csv` in a structured format containing all lattice curves; 2) a directory, `wovenCSV_[date]` containing each lattice curve in a separate .csv file; 3) an .stl 3D file, `woven_mesh.stl` of the woven lattice.  

This demo should take no more than 30 seconds to run on a desktop computer. 

## Contributing

Contributions to the project are welcome. If you have suggestions, bug fixes, or improvements, feel free to create a pull request or open an issue. Please ensure that your contributions align with the project's goals and coding standards.
