# 2248436_Assessment 
Problem Statement 1
An online fashion retailer wants to develop a machine learning model that can classify customer reviews into different sentiment categories. The model will take as input a customer review and output a prediction of the review's sentiment, such as positive,negative, or neutral. A ML model for aforesaid classification has been built.

DATA OVERVIEW
	Nature of the Dataset
This dataset is a collection of customer reviews for women's clothing products sold by an e-commerce retailer. The dataset contains 23,486 reviews, each with 10 attributes:
1.	Clothing ID: An integer identifying the specific clothing item being reviewed.
2.	Age: The reviewer's age in years.
3.	Title: The title of the review, typically a brief summary of the reviewer's opinion.
4.	Review Text: The body of the review, containing the reviewer's detailed feedback.
5.	Rating: The reviewer's overall rating of the product, ranging from 1 (worst) to 5 (best).
6.	Recommended IND: A binary indicator indicating whether the reviewer recommends the product (1) or not (0).
7.	Positive Feedback Count: The number of other customers who found the reviewer's review helpful or positive.
8.	Division Name: The high-level category of the product, such as "Dresses" or "Tops."
9.	Department Name: The specific department within the division, such as "Casual Dresses" or "Formal Tops."
10.	Class Name: The most detailed product category, such as "Maxi Dress" or "T-Shirt."
	Characteristics of the Dataset
•	Mixed data types: The dataset contains both numerical (age, rating, positive feedback count) and categorical (clothing ID, title, review text, recommended IND, division name, department name, class name) variables.
•	Varied review lengths: Review text lengths vary significantly, ranging from short phrases to lengthy paragraphs.
•	Subjective opinions: Review text contains subjective opinions and sentiments, making it challenging to analyze quantitatively.
•	Imbalanced classes: The distribution of ratings is skewed, with more reviews in the lower rating categories (1-2 stars) compared to higher ratings (4-5 stars).

DATA CLEANING
1.	Drop the first column: The first column, likely an index or identifier, is not needed for analysis, so it is dropped.
2.	Check for missing values: Identify the columns with missing values and determine the extent of missingness.
3.	Impute missing values in categorical columns: Replace missing values in categorical columns ('Division Name', 'Department Name', and 'Class Name') with the most frequent value (mode) for each respective column.
4.	Impute missing values in 'Review Text': For missing values in 'Review Text', use a simple imputer to replace them with the most frequent value. Since missing text reviews might indicate a lack of feedback, replace any remaining missing values with a specific phrase like "No review."
5.	Drop missing values in 'Title': Since missing values in 'Title' are likely due to incomplete or irrelevant reviews, drop rows with missing values in this column.
   
EXPLORATORY DATA ANALYSIS
1.	Distribution of Text Length: Create a histogram using Plotly's px.histogram() function to visualize the distribution of the length of the reviews. The histogram shows the frequency of reviews with different text lengths, with separate bars for recommended and non-recommended reviews.
2.	Distribution of Recommendations: Create a pie chart using Plotly's go.Pie() function to visualize the proportion of recommended and non-recommended reviews. The pie chart clearly shows the percentage of each category.
3.	Distribution of Ratings: Create a histogram using Plotly's px.histogram() function to visualize the distribution of product ratings. The histogram shows the frequency of reviews with different ratings (1-star to 5-star).
4.	Distribution of Customer Age: Create a histogram using Plotly's px.histogram() function to visualize the distribution of customer ages. The histogram shows the frequency of customers within different age groups.
5.	Relationship between Ratings and Departments: Create a grouped histogram using Plotly's px.histogram() function to visualize the distribution of ratings within different product departments. The grouped histogram allows for comparison of rating distributions across departments.
6.	Top 200 Frequent Words in the Dataset: Create a treemap using Plotly's px.treemap() function to visualize the top 200 most frequent words in the entire dataset. The treemap structure represents the hierarchical relationship between words, with more frequent words having larger areas.
7.	WordCloud of Recommended Reviews: Generate a word cloud using WordCloud() to visualize the most prominent words in the reviews that are recommended by customers. The word cloud displays words with larger font sizes representing their higher frequency.
8.	Top 200 Frequent Words in Recommended Reviews: Create another treemap using px.treemap() to specifically visualize the top 200 most frequent words in the recommended reviews. This allows for a more focused analysis of the language used in positive reviews.
   
DATA PRE-PROCESSING & FEATURE ENGINEERING
1.	Distribution of Text Length after Cleaning: Create a histogram using Plotly's px.histogram() function to visualize the distribution of the length of the reviews after cleaning. The histogram shows the frequency of reviews with different text lengths, with separate bars for recommended and non-recommended reviews.
2.	Distribution of Recommendations: Create a pie chart using Plotly's go.Pie() function to visualize the proportion of recommended and non-recommended reviews. The pie chart clearly shows the percentage of each category.
3.	Distribution of Ratings: Create a histogram using Plotly's px.histogram() function to visualize the distribution of product ratings. The histogram shows the frequency of reviews with different ratings (1-star to 5-star).
4.	Distribution of Customer Age: Create a histogram using Plotly's px.histogram() function to visualize the distribution of customer ages. The histogram shows the frequency of customers within different age groups.
5.	Relationship between Ratings and Departments: Create a grouped histogram using Plotly's px.histogram() function to visualize the distribution of ratings within different product departments. The grouped histogram allows for comparison of rating distributions across departments.
   
TRAIN – TEST SPLIT 
1.	Train-Test Split: The first step is to split the data into two sets: a training set (80%) and a test set (20%). This is done using the train_test_split() function from the scikit-learn library. The random_state parameter is set to 13 to ensure reproducibility of the results.
2.	Initializing a Tokenizer: A tokenizer object is created using the Tokenizer() function from the TensorFlow Text library. The num_words parameter specifies the maximum number of words to consider in the vocabulary, and the oov_token parameter defines the out-of-vocabulary token used for unknown words.
3.	Fitting the Tokenizer: The tokenizer is trained on the training data using the fit_on_texts() method. This process builds a vocabulary of the most frequent words in the training data.
4.	Tokenizing Training and Validation Data: The training and validation text data are tokenized using the texts_to_sequences() method. This converts the text data into sequences of integers, where each integer represents a word in the vocabulary.
5.	Displaying Examples: Examples of non-tokenized and tokenized text are displayed for two different samples from the training data. This allows for visualizing the transformation of raw text into numerical sequences.
   
MODEL BUILDING 
1.	Padding the Tokenized Sequences: The tokenized training and validation sequences are padded to a maximum length of 50 tokens using the pad_sequences() function from the TensorFlow Keras preprocessing library. The padding='pre' parameter specifies that the padding should be added at the beginning of the sequences. This ensures that all sequences have the same length, which is necessary for the input layer of the neural network.
2.	Creating the Model: A sequential neural network model is created using the Sequential() function from the TensorFlow Keras library. The model consists of the following layers:
o	Embedding Layer: This layer embeds the tokenized sequences into a vector space of dimension 16. This allows the model to learn meaningful representations of the words in the vocabulary.
o	Dropout Layer: This layer drops out 20% of the neurons in the embedding layer to prevent overfitting.
o	Global Average Pooling 1D Layer: This layer averages the outputs of the embedding layer over the entire sequence length. This reduces the dimensionality of the data and helps to capture the overall sentiment of the reviews.
o	Dropout Layer: This layer drops out 50% of the neurons in the global average pooling layer to prevent overfitting.
o	Dense Layer: This layer is a fully connected layer with one output neuron, corresponding to the binary classification task (recommended or not recommended). The activation function for this layer is sigmoid, which produces a probability between 0 and 1 for each review.
3.	Optimizing and Compiling the Model: The Adam optimizer is used with a learning rate of 5.5e-3 to optimize the model's weights during training. The loss function is set to binary_crossentropy, which is appropriate for binary classification tasks. The accuracy metric is used to evaluate the model's performance on the validation dataset.
4.	Model Summary: The model summary is printed to the console, providing an overview of the model architecture, including the number of parameters and the output shape of each layer.
5.	Defining Early Stopping: An EarlyStopping callback is created to prevent overfitting. This callback monitors the val_accuracy metric and stops training if the validation accuracy does not improve for five consecutive epochs. The restore_best_weights=True parameter ensures that the model weights from the best validation epoch are used for evaluation and prediction.
6.	Setting Training Parameters: The number of training epochs is set to 100, and the batch size is set to 32. The batch size determines the number of samples processed before updating the model weights.
7.	Training the Model: The fit() method is used to train the model on the padded training data and the corresponding labels (y_train). The validation data (Padded_val and y_val) is passed as an argument to evaluate the model's performance during training. The callbacks=[early_stopping] parameter specifies the EarlyStopping callback that will be used to monitor the validation accuracy.
8.	Storing Training History: The training history is stored in the hist variable. This history contains information about the model's performance during training, such as the loss and accuracy on both the training and validation datasets.
	LSTM
1.	Loop through predictions for the test data: Iterate over each prediction value in the 'pred_test_lstm' array.
2.	Apply threshold: Check if the prediction value falls within the threshold range (0 to 0.49).
3.	Assign class labels: If the prediction value is within the threshold, assign the class label 0. Otherwise, assign class label 1.
4.	Repeat for training data: Repeat the same process for the 'pred_train_lstm' array.

MODEL EVALUATION
	Plotting Loss Graphs
Plotting Train and Validation Loss: The training loss (hist.history['loss']) and validation loss (hist.history['val_loss']) are plotted as separate lines on the same graph using the plt.plot() function. The label parameter is used to identify each line with a corresponding label.
	Plotting Accuracy Graphs
Plotting Train and Validation Accuracy: The training accuracy (hist.history['accuracy']) and validation accuracy (hist.history['val_accuracy']) are plotted as separate lines on the same graph using the plt.plot() function. The label parameter is used to identify each line with a corresponding label.
	Creating a confusion matrix and heatmap visualization
1.	Calculate confusion matrix: Generate the confusion matrix using the 'confusion_matrix' function, comparing the true labels ('y_true') with the adjusted predictions ('pred_test_lstm').
2.	Create heatmap: Initialize a figure with a specific size ('figsize') and use the 'sns.heatmap' function to visualize the confusion matrix.
3.	Add annotations and labels: Display the actual class labels ('Real Class') on the y-axis and the predicted class labels ('Predicted Class') on the x-axis.
4.	Set title and show plot: Add a title ('Confusion Matrix of the Test Data') and display the heatmap using the 'plt.show()' function.
	Calculating and displaying the training accuracy
1.	Calculate training accuracy: Use the 'accuracy_score' function to compute the accuracy of the LSTM model's predictions on the training data ('y_train') compared to the actual labels ('pred_train_lstm').
2.	Round accuracy value: Round the calculated accuracy value to two decimal places using the 'round' function with the 'precision' argument set to 2.
3.	Print training accuracy: Display the training accuracy.
	Calculating and displaying the test accuracy
1.	Calculate test accuracy: Similarly, compute the accuracy of the LSTM model's predictions on the test data ('y_test') compared to the actual labels ('pred_test_lstm').
2.	Round accuracy value: Round the calculated accuracy value to two decimal places using the 'round' function with the 'precision' argument set to 2.
3.	Print test accuracy: Display the test accuracy.
   
CONCLUSION
The LSTM model developed shows promising results, achieving a training accuracy of 92.44% and a test accuracy of 89.43%. These results indicate that the model can effectively classify flower images and provide accurate information to users.
Integrating this model into a mobile application for a flower bouquet shop would significantly enhance the customer experience and provide several benefits:
1. Accurate Flower Recognition: The model's ability to identify flower species with high accuracy will allow users to easily and accurately determine the type of flower they are photographing.
3. Convenient Flower Ordering: Users will be able to seamlessly order flowers similar to the ones they have photographed directly through the mobile application. This will encourage impulse purchases and increase sales for the flower shop.
4. Enhanced Customer Engagement: The image classification feature will create a more interactive and engaging experience for users, fostering their interest in flowers and encouraging them to explore the shop's offerings.
5. Valuable Customer Insights: The collected image data and user interactions can be analyzed to gain insights into customer preferences, popular flower varieties, and seasonal trends. This information can be used to improve marketing strategies and product offerings.
Overall, the integration of the LSTM model into the mobile application will not only enhance customer experience but also provide the flower shop with valuable data and insights to improve their business operations.



