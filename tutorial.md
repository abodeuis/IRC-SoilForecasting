# IRC SoilForecasting Basic Tutorial

This is the basic tutorial for using the soil forecasting scripts. 



**Contents**
- [IRC SoilForecasting Basic Tutorial](#irc-soilforecasting-basic-tutorial)
  - [Setting up the Software Environment](#setting-up-the-software-environment)
  - [Running Data Analysis](#running-data-analysis)
  - [Training a Model](#training-a-model)
    - [Results](#results)
  - [Predictions](#predictions)

## Setting up the Software Environment
For this project we will be using python 3.6 or greater. You can see if you have python and what version it is by opening a terminal or command prompt and using ```python --verison```. ***Note** *On Mac OS X you will use the* <code>python3</code> *command instead of* ```python```

![Example of using python --version](tutorial_images/pyversion.png)

**Installing Python :** If you don't have python installed you can downloaded the latest GUI installer from [here](https://www.python.org/downloads/). You'll then need to run the installer and once it is done you can check it worked by using ```python --version``` again.

**Installing Python Packages** : For the next step you will need to have a terminal window that is in the directory that you downloaded this git repository to. Once we are in that directory we can install the packages we will need for the tutorial with ```pip install -r requirements.txt```. This command will download all of the packages listed in requirements.txt file and may take a short amount of time to do so.

## Running Data Analysis ###
The data analysis is performed by **ICN-Analysis.py**. By default all of the python scripts we will be using look for a **config.ini** file in the same directory they are in, but if we want to use a specific config we can use the ```--config``` option. So for this tutorial let's try running the analysis by using ```python ICN-Analysis.py --config sample_configs/dailyConfig.ini```. This will perform the data analysis on the files specified by the ```data_source``` option in the config file we used from ```sample_period_start``` through ```sample_period_end```. The sample config will output its results to a folder called **results** which should contain the following analysis outputs:

* Csv of the statistics for each site.
  
|       | max_wind_gust | avg_wind_speed |  ...  | SM100 | SM150 |
| :---- | :-----------: | :------------: | :---: | :---: | ----: |
| count |     1053      |      1053      |  ...  | 1053  |  1053 |
| mean  |     16.18     |      4.17      |  ...  | 0.361 | 0.355 |
| std   |     6.12      |      2.33      |  ...  | 0.037 | 0.026 |
| min   |     5.69      |      0.89      |  ...  | 0.250 | 0.273 |
| 25%   |     11.69     |      2.50      |  ...  | 0.360 | 0.345 |
| 50%   |     15.10     |      3.59      |  ...  | 0.370 | 0.360 |
| 75%   |     19.39     |      5.40      |  ...  | 0.379 | 0.379 |
| max   |     42.29     |     17.79      |  ...  | 0.409 | 0.389 |

* Corralation graph of the variables for each site.
![CMI Corraltion graph](sample_plots/cmi_corr.png)
* Plot of the orginal data for each variable.
* AutoCorralation Function (ACF) Plot for each variable.
* Partial ACF Plot or each variable.
![Average Air Tempature Plot](sample_plots/avg_air_temp.png)

## Training a Model ###
To train a model on this data we will need to use ```python ICN-Training.py --config sample_configs/dailyConfig.ini```

### Results ###
In the ```save_path``` directory there will be **logs**, **models** and a sample prediction graph to get a quick view of how well the model is performing.

## Predictions ###
Once we have a trained model we might want to run some more tests on it then just the ones that were done at the end of training. To do this we can use the ICN-Predictions script to get predictions for data.

Coming soon, script still has some bugs that break it very easily.
