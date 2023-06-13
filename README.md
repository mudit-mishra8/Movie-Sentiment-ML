 <img src="https://cutewallpaper.org/23/america-psycho-galaxy-wallpaper/438214424.jpg" alt="Image 1" width="1000" height="420">
 <img src="https://www.travelmediagroup.com/wp-content/uploads/2022/04/bigstock-Market-Sentiment-Fear-And-Gre-451706057-2880x1800.jpg" alt="Image 1" width="900" height="280">

Welcome to my Movie Sentiment Analysis project! In this endeavor, I've delved into the exciting world of Natural Language Processing (NLP) to develop a sophisticated deep learning model that classifies movie reviews as either positive or negative. The sentiment analysis of user reviews is a powerful tool, with applications ranging from product recommendations to audience sentiment insights, and is paramount for stakeholders in the entertainment industry.

## Dataset

The dataset employed in this project contains 50,000 movie reviews from the IMDB database and is meticulously balanced, meaning it has an equal number of positive and negative reviews. A balanced dataset is crucial for unbiased model training, ensuring that our model doesn't develop a preference for one class over the other due to unequal representation.

## Workflow Overview

1. **Data Cleaning and Preprocessing**: This initial step involved cleaning the text data by removing unwanted characters, such as HTML tags, and converting the text into a more consistent format. Preprocessing also includes converting all text into lowercase to make the data ready for further analysis.

2. **Vectorization**: To enable the deep learning model to understand and learn from the text data, the words in the reviews were converted into numerical vectors. For this purpose, Word2Vec was employed, which captures the semantic meaning of words by representing them in a high-dimensional space. This approach is highly effective as it retains the context of words within the reviews.

3. **Train-Test Split**: The dataset was split into training and testing sets, which is a standard practice in machine learning. This approach allows us to train the model on one subset of data and validate its performance on an unseen subset, thus giving us a realistic idea of how well the model is likely to perform on real-world data.

4. **Model Building and Training**: A deep learning model was constructed and trained using the processed and vectorized text data. This model is capable of identifying patterns and features in the text that are indicative of the sentiment being positive or negative.

5. **Model Evaluation**: The model's performance was assessed using various metrics such as accuracy, precision, recall, and F1-score, among others.

6. **Sentiment Prediction Function**: Lastly, a user-friendly function was developed, which takes in a raw movie review as an input and returns the sentiment as either positive or negative by utilizing the trained deep learning model. Also i have created  a website for the same, right click to open in a new tab! 👉 <a href="https://imdb-app.herokuapp.com/" target="_blank">MyMovieSentimentAnalyzer</a>


Feel free to explore the code and insights throughout this project. This endeavor demonstrates the potential of deep learning in NLP tasks and provides valuable insights into how the audience perceives movies based on their reviews. It is not just a classification task; it’s an amalgamation of data science techniques and natural language processing to create a model that understands human emotions from the text.

Let's dive into the code and analysis!

## For code please check the python notebook attached below:

clcik here 👉 [Jupyter Notebook](https://github.com/mudit-mishra8/Movie-Sentiment-ML/blob/main/IMDB_sentiment_prediction.ipynb)

## Results
The performance of the model was evaluated using a variety of metrics, including accuracy, precision, recall, and F1-score. Below is a summary of the results:

### Accuracy
The model achieved an accuracy of approximately 91%. In simple terms, this means that out of every 100 movie reviews, the model correctly identified the sentiment (whether it is positive or negative) of about 91 of them.

### Precision and Recall
Precision tells us how many of the reviews that the model predicted as positive were actually positive. In this case, the model’s precision for positive reviews is 0.90, which means that 90% of the reviews that the model labeled as positive were actually positive. 

Recall, on the other hand, tells us how many of the actual positive reviews were correctly identified by the model. The model’s recall for positive reviews is 0.92, indicating that it correctly identified 92% of all the positive reviews in the dataset.

For negative reviews, the precision and recall are 0.92 and 0.89, respectively. 

### F1-Score
F1-score is the harmonic mean of precision and recall and is a measure that combines both. For positive reviews, the model achieved an F1-score of 0.91. Similarly, for negative reviews, the F1-score is 0.91. This indicates that the model is equally good at identifying positive and negative reviews.

### True Positives, False Positives, True Negatives, False Negatives
- True Positives: 5794. This represents the number of positive reviews that were correctly identified as positive.
- False Positives: 662. This represents the number of negative reviews that were incorrectly identified as positive.
- True Negatives: 5566. This represents the number of negative reviews that were correctly identified as negative.
- False Negatives: 478. This represents the number of positive reviews that were incorrectly identified as negative.

## Interpretation
In non-technical language, this means that the model is quite good at figuring out whether a movie review is positive or negative. It correctly identifies the sentiment of a review about 91 times out of 100. 

However, it’s not perfect. Sometimes it gets confused - for example, in some cases, it thinks a review is positive when it’s actually negative and vice versa. But overall, it makes the right call the majority of the time.

## Conclusion
This analysis demonstrates that using machine learning for sentiment analysis on IMDB movie reviews can be highly effective. The model performed well across various metrics. Such a model can be used, for example, by movie producers and directors to gauge public opinion about their movies based on reviews. It can also be a valuable tool for moviegoers looking to find a good movie to watch.


#### Hope you enjoyed! Thanks
