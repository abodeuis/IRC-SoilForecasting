# IRC SoilForecasting Basic Tutorial

This is the basic tutorial for using the soil forecasting scripts. 



**Contents**
- [IRC SoilForecasting Basic Tutorial](#irc-soilforecasting-basic-tutorial)
  - [Setting up the Software Environment](#setting-up-the-software-environment)
  - [Running Data Analysis](#running-data-analysis)
  - [Training a Model](#training-a-model)
    - [Training Config](#training-config)
    - [Running the Model](#running-the-model)
    - [Results](#results)
  - [Predictions](#predictions)

## Setting up the Software Environment
For this project we will be using python 3.6 or greater. You can see if you have python and what version it is by opening a terminal or command prompt and using ```python --verison```. ***Note** *On Mac OS X you will use the* <code>python3</code> *command instead of* ```python```

![Example of using python --version](tutorial_images/pyversion.png)

**Installing Python :** If you don't have python installed you can downloaded the latest GUI installer from [here](https://www.python.org/downloads/). You'll then need to run the installer and once it is done you can check it worked by using ```python --version``` again.

**Installing Python Packages** : For the next step you will need to have a terminal window that is in the directory that you downloaded this git repository to. Once we are in that directory we can install the packages we will need for the tutorial with ```pip install -r requirements.txt```. This command will download all of the packages listed in requirements.txt file and may take a short amount of time to do so.

## Running Data Analysis ###
The data analysis is performed by **ICN-Analysis.py** by default all of the python scripts we will be using look for a **config.ini** file in the same directory they are in, but if we want to use a specfic config we can use the ```--config``` option. So for this tutorial let's try running the analysis by using ```python ICN-Analysis.py --config sample_data/dailyConfig.ini``` This will perform the data analysis on the files specified by the ```data_source``` option in the config file we used. It will also be performed on the time period specified with ```sample_period_start``` and ```sample_period_end``` in the config file and it includes the following analysis:

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


### Training Config ###
Configure the config file
You should have a file called “example_config.ini” go ahead and open this with the text editor of your choice.
There are a bunch of settings you can tweak but the only required one is “data_source”. You need to set this to the data file(s) that you want to use for the analysis. I would also recommend setting the “save_path” and “sample_period_start”, “sample_period_end” as well.
Once you’ve made your changes make sure to save it to “config.ini”
Once you have a valid config file you can run the program with 

Before you are able to run the software you will need to configure the example config and rename it to ```config.ini```. The only required value in the config that needs to be set is the ```data_source```. This is the option that selects what data will be used for analysis.

*Note if you ever lose or need to regenerate the default config you can delete config.ini and example_config.ini and it will make a new one.

### Running the Model ###
```python ICN-Training.py --config sample_configs/dailyConfig.ini```

### Results ###
In the ```save_path``` directory there will be logs models and a sample prediction graph.


## Predictions ###
Once we have a trained model we might want to run some more tests on it then just the ones that were done at the end of training. To do this we can use the ICN-Predictions script to get predictions for data.

Coming soon, still has some bugs that break it very easily.
