# What is unbreakable?
Unbreakable is a stochastic simulation model for assessing the resilience of households to natural disasters. 

## Background
Disasters of all kinds are becoming more frequent and more severe. While, 

## Model Overview

## Getting Started

### Prerequisites
The model is developed in Python 3.9 and relies on standard packages. The easiest way to install them is to use [Anaconda](https://www.anaconda.com/products/individual) with the `requirements.txt` file:
```bash
conda create --name unbreakable --file requirements.txt
```

### Installation
Clone the repository from GitHub using the following:
```bash
git clone https://github.com/mikhailsirenko/unbreakable
```

### Repository Structure
```
unbreakable

├── config                  <- Configuration files for the model


```

### Usage
You run the model from the command line with the default parameters using:
```bash
python unbreakable/run.py
```

## Documentation
Detailed model documentation, including function descriptions and usage examples, is available [here](https://mikhailsirenko.github.io/unbreakable/src.html).



## How-to guide

### Adding a New Case Study

Imagine you would like to use the model for a new case study. What do you have to do?

First, you must ensure you have two datasets: a household survey and asset damage. Next, you have to fine-tune parameters: constants and uncertainties. You may change policies as well. However, make sure that you follow the defined naming convention. Now, let's dive into the details of each of the steps.

#### Household survey
By household survey, we mean a dataset that contains information about households in the country. Usually, it is a survey that the government or an international organization conducts. The survey must be nationally representative (weighted): each household has a weight of how many households it represents.

The survey must contain the following information:
| No. | Variable description                        | Variable name  | Type of values | Potential ranges |
|-----|---------------------------------------------|----------------|----------------|------------------|
| 1   | Household id                                | hh_id          | Integer        | N/A              |
| 2   | Household weight                            | hh_weight      | Float          | N/A              |
| 3   | District id                                 | district_id    | Integer        | N/A              |
| 4   | District name                               | district_name  | String         | N/A              |
| 5   | Household size                              | hh_size        | Integer        | N/A              |
| 6   | Household income (adult equivalent)         | ae_inc         | Float          | N/A              |
| 7   | Household expenditure (adult equivalent)    | ae_exp         | Float          | N/A              |
| 8   | Household savings (adult equivalent)        | ae_sav         | Float          | N/A              |
| 9   | Whether the household rents or owns a house | rent_own       | String         | "Rent", "Own"    |
| 10  | House price (if the household owns a house) | house_price    | Float          | N/A              |
| 11  | Rent price (if the household rents a house) | rent_price     | Float          | N/A              |
| 12  | Walls material                              | walls_material | String         | N/A              |
| 13  | Roof material                               | roof_material  | String         | N/A              |



#### Asset damage
The asset  

## Parameters
### Constants

The model has a few constants. Note that one could treat them as uncertainties. However, we decided to keep them as constants for the sake of simplicity. The constants are:

1. Average productivity of capital (`average_productivity`)

To update this constant, we use the Penn World Table. You can download it from the [Penn World Table website](https://www.rug.nl/ggdc/productivity/pwt/). To estimate the average productivity of capital, we take the average Output-side real GDP at current PPPs (`cgdpo`) to Capital stock at the current PPPs (`cn`) ratio for the last five years. For example, for Dominica, these values are:

| Year | Output-side real GDP at current PPPs (`cgdpo`) | Capital stock at current PPPs (`cn`) |
|------|------------------------------------------------|--------------------------------------|
| 2015 | 709.8839111                                    | 2500.742676                          |
| 2016 | 771.4655762                                    | 2497.276855                          |
| 2017 | 722.5056763                                    | 2560.524658                          |
| 2018 | 704.8121948                                    | 2742.36377                           |
| 2019 | 746.8662109                                    | 2821.069092                          |

Thus, the average productivity of capital is 0.28.

1. Consumption utility (`consump_util`)
1. Discount rate (`discount_rate`) 
1. Income and expenditure growth (`income_and_expenditure_growth`)
1. Savings rate (`savings_rate`)
1. Poverty line (`poverty_line`)

Besides these, we must adjust the names of the `country` and `districts`. The first one indicates with which country we work, and the second one is a list of districts in the country. Note that those could be called differently: neighborhoods or parishes. For simplicity in the model, we call all of them districts.

### Uncertainties

### Policies

## Contributing

## Authors
*Mikhail Sirenko* and *Bramka Arga Jafino*.

## License
The project license is CC BY 3.0 IGO.

## Contact
For inquiries or collaboration, feel free to reach out to [Mikhail Sirenko](https://twitter.com/mikhailsirenko).

## References
Hallegatte, S., Vogt-Schilb, A., Bangalore, M., & Rozenberg, J. (2017). *Unbreakable: Building the Resilience of the Poor in the Face of Natural Disasters*. Climate Change and Development. Washington, DC: World Bank. [http://hdl.handle.net/10986/25335](http://hdl.handle.net/10986/25335)