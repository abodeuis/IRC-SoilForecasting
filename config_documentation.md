## Documentation of Config Options
- [Documentation of Config Options](#documentation-of-config-options)
  - [Data](#data)
  - [Training](#training)
  - [Plots](#plots)
  - [Other](#other)

### Data
**data_source** : This option is required. It specifies what data file(s) should be used for analysis. You can specify a single file, a list of files, or a directory for it to use.

**target_cols** : This is the variable in the data that we will be trying to predict.

**validation_percent** : This is the percentage of the data that will be held back from training to be used for validating the model. Right now it just picks the last N percent of the data. Will add better selection methods in the future

**numeric_cols** : These are the variables that will be used as predictors, only variables in this list will be included in the pre-training analysis as well.

### Training
**epochs** : Number of training epochs to do for the neural network based models.

**batch_size** : Number of samples to use at each training step

**early_stopping** : Number of epochs with no improvement in loss required to stop training.

**validation_batchs** : Number of validation batchs to use at each training step

**learning_rate** : The learning rate of the neural network based models.

### Plots
**sample_period_start** : start date of the sample period to use for pre-training analysis. Format is (Month/Day/Year Hour:Min:Sec)

**sample_period_end** : End date of the sample period to use for pre-training alaysis. Format is (Month/Day/Year Hour:Min:Sec)

**acf_days** : Number of days to plot for the ACF/PACF graphs

### Other
**debug_level** : Changes the level of messages included the log log file. Valid options are [DEBUG, INFO, WARNING, ERROR, CRITICAL]

**save_path** : The folder that the results of the run will be saved to. 