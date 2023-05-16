# Forest-Cover-Classification 

### The objectives of this project is built a deep learning model to predict the forest cover type from different cartographic variables.

1. Forest Cover Types: ['Spruce/Fir', 'Lodgepole Pine','Ponderosa Pine', 'Cottonwood/Willow','Aspen', 'Douglas-fir', 'Krummholz']
2. Dataset ('cover_data.csv') that contains 581012 observations. Each observation has 55 columns (54 features and the last one being the class).

Model Test Accuracy: 91.5432 %
Model Test Loss: 0.2159

Model Accuracy & Model Loss with `sparse_categorical_crossentropy` loss function and `Adam` optimizer:
![](/images/model_accuraacy.png)
![](/images/model_loss.png)

From confusion matrix heatmap, we see that `Lodgepole Pine`, `Cottonwood Willow`, `Aspen`, and `Douglas-Fir suffer` from a high percentage of mis-classifications. To investigate the possible causes, one can explore the following:
![](/images/cf.png)

From Classification Report, all value of evaluation metrics become quite high after adding more hidden layers and hyperprameter tuning.

precision  |  recall  | f1-score  | support
---------------------------------------------------------
Spruce/Fir  |     0.93  |    0.90    |  0.91    | 42368
   Lodgepole Pine  |     0.92   |   0.94    |  0.93  |   56661
   Ponderosa Pine  |     0.89   |   0.92    |  0.91  |    7151
Cottonwood/Willow   |    0.85   |   0.77    |  0.81  |     549
            Aspen    |   0.88   |   0.66    |  0.75  |    1899
      Douglas-fir    |  0.85    |   0.80    |  0.82  |    3473
        Krummholz    |   0.91   |   0.95    |  0.93  |    4102

         accuracy    |          |           |  0.92   | 116203
        macro avg    |   0.89   |   0.85    |  0.87   | 116203
     weighted avg    |   0.92   |   0.92    |  0.91   | 116203

### Future Improvement
* Check the proportion of observations for each cover-type. Imbalances in the dataset will affect classification.
* How each wilderness area is distributed.
* Find the similarties among different cover-types (correlation, scatter-plots, etc.) These similarities might be one of the reasons the model might be tripping-up. There are ways to address it - one of it is to carefully remove all of the collinear variables, leaving only one.
* Remove noise, inconsistent data and errors in training data - this should be done carefully with domain experts.
* Try to use some other performance metric other than 'accuracy'. It fails to be a reliable metric when data in imbalanced. That is why we have other metrics such as Precision/Recall, F1-score etc.
* Try resampling the data (undersample or oversample appropriately or stratified). Downsampling could be done with thresholding.
* The most important thing to understand here is that in deep-learning, the gradient(s) of the majority class(es) dominate(s) and will influence the weight-updates. There are also some advanced techniques that will ameliorate this situation.
