# What is Unbreakable?
Building the Resilience of the Poor in the Face of Natural Disasters.

## Background
This repository contains a modified version of the code behind Hallegatte et al. (2017).

## Documentation
Read the documentation [here](https://mikhailsirenko.github.io/unbreakable/src.html).

## How to use this repository
You can run the code in this repository by cloning the repository and installing it by running the following command in the terminal:

```
pip install -e .
```

Note that you may need to install some dependencies from the environment.yml. To do that, run the following command in the terminal:

```
conda env create -f environment.yml
```

Additionally, the repository does not contain the data to replicate the results. However, you may request the data.

## Repo structure
This repository has the following structure:
    
``` 
├── data
│   ├── external       <- Data from third party sources.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
├── docs               <- pdoc generated documentation of the model (src module).
├── notebooks          <- Jupyter notebooks for analysis.
├── results            <- Experiment results.
├── src                <- Source code for use in this project.
│   ├── __main__.py    <- Runs the simulation model.
│   ├── model.py       <- The simulation model.
│   ├── data           <- Functions to prepare, read and write data (outcomes of the simulation model)
│   └── modules        <- Functions to run the simulation model.
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment.
├── README.md          <- The top-level README for using this project.
├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported.
```

## Authors
*Mikhail Sirenko*, *Bramka Arga Jafino* and *Brian Walsh*. 

## License
CC BY 3.0 IGO

## Acknowledgements

## References
* Hallegatte, Stephane; Vogt-Schilb, Adrien; Bangalore, Mook; Rozenberg, Julie. 2017. Unbreakable: Building the Resilience of the Poor in the Face of Natural Disasters. Climate Change and Development; © Washington, DC: World Bank. http://hdl.handle.net/10986/25335 License: CC BY 3.0 IGO.