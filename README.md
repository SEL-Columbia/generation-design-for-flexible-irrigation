# Generation Design for Flexible Irrigation

This repository contains the generation design model for systems with fixed domestic and flexible irrigation. 

It minimizes system costs and determines the capacities of solar panels, battery storage, battery inverters, and diesel generators based on the provided inputs and system constraints. More details can be found in the paper.
## Files and Directories

### Modeling Files

- **main.py**: The main script to run the entire model.
- **capacity_model.py**: Contains the code for phase-1 modeling without irrigation minimum power requirement.
- **operation_model.py**: Contains the code for phase-2 modeling with irrigation minimum power requirement.
- **results_processing.py**: Processes and analyzes the results generated by the model.
- **utils.py**: Utility functions used across the model.
- **params.yaml**: Configuration file for model parameters.

### Input Data Files

- **fixed_load_kw.csv**: Contains data on fixed loads in kilowatts.
- **rain_rate_mm_2014_2015.csv**: Daily rainfall rate data in mm. 
- **irrig_zones_centroidsmodelOutput.txt**: Output data of network from Two-Level Network Design.
- **pts_area.csv**: Contains data on the area of different irrigation zones in square meters. 
- **solar_po_2014_2015.csv**: One-kilowatt solar panel potential output data.

## Before Model Running

1. Fork this repository.
2. Create a Python environment with version 3.7 or higher.
3. To install the necessary dependencies, run:
```sh
pip install -r requirements.txt
```
4. Lastly, the `gurobipy` package is required, which needs a license to use.


## Run the Model:
```sh
python main.py
```
