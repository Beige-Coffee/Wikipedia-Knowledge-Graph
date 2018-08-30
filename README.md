# Wikipedia-Knowledge-Graph
Using 30,000 hand-graded Wikipedia articles and NLP to predict the quality of Wikipedia articles and create a knowledge graph that identifies both articles and topics in need of editorial and epistemological attention.


Please note that this is a work-in-progress. 

Thanks,
- Austin (Aug, 2018)

## The Problem In 1 Sentence. My Solution In 1 Sentence.
- **Problem:** Wikipedia contributors are tasked with detecting which articles (out of the 50 million possible articles) are in need of editorial attention and, subsequently, contributing their knowledge to enhance the encyclopedic value of Wikipedia.

- **My Solution:** I built a model which algorithmically predicts an article's quality score and displays the results (aggregated by category and sub-category) on a user-friendly website. 


## Business Understanding / Let's Put This In Context
With roughly 50 million articles in Wikipedia's corpus, once can imagine how daunting of a task it must be to find the optimal balance between quality and quantity. For example, if you take a look at the following two articles, you can begin to see how disparate the range of quality is across Wikipedia. 
- High quality article: https://en.wikipedia.org/wiki/Elizabeth_II
- Low quality article: https://en.wikipedia.org/wiki/Ring-tailed_cardinalfish

Now, at first thought, you may think that this difference in quality is not only understandable but expected. And, if this was your thought, you'd be right. However, the more nuanced point is:
1) These articles were hand-selected to elucidate the point that a difference of quality exists
2) This difference in quality exists across many articles and, therefore, is not easy to readily detect

However, the advent of machine learning and predictive modeling has done a lot to level the scale between quality and quantity. In hopes of contributing towards this endeavor and, *just as importantly*, **building a product that can be intuitively understood by any Wikipedia editor**, I have built a machine learning model to predict the quality score of a Wikipedia article and, subsequently, organize article predictions by category on a user-friendly website. By using these tools, Wikipedia editors can seamlessly find articles that match with their personal expertise and sort by each article's predicted article quality, thus, allowing them to efficiently focus on contributing their knowledge where it's needed most.

## The Data

### Train/Test Data
The training data consisted of roughly 30,000 graded Wikipedia articles - data openly available here: https://analytics.wikimedia.org/datasets/archive/public-datasets/enwiki/article_quality/. In this context, "graded" means that a person assigned a quality score to the Wikipedia article based on its encyclopedic value. For example, as depicted in the table below, "FA" signifies that the article is a "definitive source for encyclopedic information".


<img width="771" alt="screen shot 2018-08-30 at 10 13 59 am" src="https://user-images.githubusercontent.com/34213201/44867782-ad94e900-ac3d-11e8-95c1-c7dd42703773.png">


To train my model, I decided to convert these classes into integers. In doing this, I was effectively pivoting from a classification problem to a regression problem. This was a strategic decision that was made in light of the following factors:
1) While the border between a "**GA**" and a "**FA**" article is somewhat clearly defined *in words*, it's much harder to parse out in practice. For example, a "**FA**" article **is** a professional encyclopedic entry while a "**GA**" article is **almost** "professional". Using just this one example, we can see how the difference between these two classes relies on the definition of "professional", which undoubtedly changes person-to-person.
2) The classes can easily be thought of as categories which differ in a relatively consistent manner. More specifically, the difference between a "**Stub**" and a "**Start**" article is probably similar to the difference between a "**GA**" and a "**FA**" article. To further illustrate this with a counter-example, imagine turning the three categories **"Dog"**, **"Monkey"**, and **"Ant"** into integers. The difference between a "**Dog**" and a "**Monkey**" is not at all similar to the difference between a "**Monkey**" and an "**Ant**".

So, in summary, I mapped the categorical target outputs to numerical outputs and trained my model on those. This also had the nice side effect of being able to predict, for example, a 4.5 - signifying that this article is probably better than a 4 but not quite as good as a 5. Also, an important thing to note is that the distribution of classes (or article labels) was uniform across the data. In other words, there were about 4,500 articles for each class in my training set.


![quality2](https://user-images.githubusercontent.com/34213201/44868821-cfdc3600-ac40-11e8-8b33-a1606df41cf6.png)
Picture of transformed target values used when training the model

Lastly, the data that was transformed into features that acted as input to the model was simply the raw Wikipedia text of an article. In this context, "raw" means the XML text that includes tags for items such as citations, links, images, etc. A sample of the input data, before transformations were applied, is pictured below.

<img width="408" alt="screen shot 2018-08-30 at 10 54 45 am" src="https://user-images.githubusercontent.com/34213201/44869750-3b270780-ac43-11e8-88a4-ddd374e5dd91.png">


### New Data
While plenty of data was available to train my model, I took to the internet to gather new data to make predictions on. To accomplish this, I built a multitude of web-scraping functions that, for example, would take a category as input and, subsequently, return a dataframe including:
- The original category's sub-categories
- The articles in each sub-category
- The predicted quality score of each article

The most important piece of this puzzle was an API call to Wikipedia, which iteratively returned the raw XML text for each article that my web-scraping functions had identified. This data was then aggregated and fed into my website using Flask and Jinja.

## My Models

### Model 1
I took a couple approaches in building a model to accurately predict the quality of a wikipedia class. The first approach, and, ultimately, the most successful, was a random forest model that took about 30 hand-engineered features as input. These hand-engineered features were carefully selected by examining the raw XML data and cross-referencing that with Wikipedia's documentation to determine which tags (the markup text highlighted in green in the image above) to focus on. Additionally, I used **textstat**, a module that specializes in extracting text-related statistics from documents, to construct NLP features that shed more light on the semantics of an article - versus structural features, such as article length. Some of the features used in the model are listed below:
- Article Length
- Number of Web, Book, News, and Journal Citations
- Whether Article Had an "Infobox" or Not (Boolean Value)
- Number of "Difficult" words
  - A "difficult" word is any word that does **not** appear in the 3,000 most common words in a 4th graders vocabulary

### Model 2
In addition to a random forest that took hand-engineered features as input, I sought out to create a model that took hash-vectorized tfidf vectorizors as input. In short, hash-vectorized tfidf vectors are ways to represent a text document in terms of numbers. Doing this makes it easy for a machine learning model to digest text and derive 'meaning' from the words. In this context, I chose to use a hash-vectorized tfidf vectorizors instead of the more ubiquitous tfidf vectorizer because, by hashing the vector, it takes up less memory and is less computationaly expensive. 

After tranforming the training data into hash-vectorized inputs, I experimented with using a random forest and gradient boosted model. In the end, both models had slightly higher mean squared error scores when compared to the aforementioned random forest. 

### Model 2.5
The next model I experimented with was an ensemble model of the orginal rando forrest with hand-engineered features and the new random forest with hash-vectorizers describing each article. More precisely, I fit a linear regression on the predicted values of each article - the predicted values being the **two article quality scores predicted by each of the two random forrests**. My hope in doing this was that the linear regression would be able to find the optimal weights for each coefficient and produce better results than just one model alone. In other words, I was hoping that each random forest would capture something distinctly unique about each article and that, when combined, they would produce a score better than either model alone. However, after comparing the MSE of the ensemble model with *just the original random forest with hand-engineered features*, I determined that the MSE of the ensemble was roughly 0.75 while the MSE of the Hand-Engineered random forest was 0.67. 

### Model 3
In addition to the aforementioned models, I also experimented with a reccurent neural network (RNN). This approach was inspired by reading many NLP research articles and continually be exposed to the effectiveness of RNN on text data. In the end, my RNN did not result in an improvement in predictive power (when compared to either the hash-vectorized random forest or the hand-engineered random forest). Therefore, in the end, I chose to move forward with the hand-engineered random forest - especially when considering the time-factor of training an RNN. 


## Evaluation
As alluded to in a couple of the previous sections, model evaluation took the form of finding the model which minimized the mean squared error (MSE). The reason that I chose MSE was because each model was predicting a continuous score - with whole integers mapping to an article quality class ("**FA**", for example). Therefore, choosing the model that minimized the MSE was akin to choosing the model that, on average, most accurately predicted a quality score that was similar to the actual score - while also severely punishing outlier predictions (Predicting "**FA**" when the actual class is "**Start**", for example). After all was tested and evaluated, Model 1 - the random forrest with hand-engineered features) came out on top as the best model for the job.

## The Final Product
In an effort to deliver article predictions in an intuitive way that any user can understand, I built a website that aggregates article predictions first by sub-category and then by overaching category. My reason for contructing the hierarchy in this manner was to implenent a logical structure that micmiced that of human thought. For example, a user usually has one (or multiple) higher level concepts that interest them. For instance, I am particularly interested in Entrepreneurship, Machine Learning, and Philosophy. Therefore, I would start with these topics and then begin to recursively digress until I reach a sub-category or article that closely aligns with my interest. 

Discloser: The website is days away from going live, however, in the meantime, I have uploaded pictures below that represent how a user would interact with the site.

### Page 1: Homepage: 
#### From here, the user would click a category that interests them. For example, let's pretend we click "Machine Learning"
<img width="1436" alt="screen shot 2018-08-30 at 12 47 07 pm" src="https://user-images.githubusercontent.com/34213201/44875382-dffd1100-ac52-11e8-9169-3ef9c77a69a3.png">


### Page 2: Machine Learning Sub-Categories: 
#### Now, the user will be presented the sub-categories under Machine Learning, with option to sort alphabetically, by predicted sub-category quality, or search for a specific sub-category.
<img width="1437" alt="screen shot 2018-08-30 at 12 50 14 pm" src="https://user-images.githubusercontent.com/34213201/44875526-4d10a680-ac53-11e8-87c0-323844fa0879.png">


### Page 3: Articles In A Certain Sub-Category: 
#### After clicking a sub-category, the user will be presented with all Wikipedia articles in that sub-category and have the option to click directly to an article's Wikipedia page.
<img width="1434" alt="screen shot 2018-08-30 at 12 52 12 pm" src="https://user-images.githubusercontent.com/34213201/44875625-9d880400-ac53-11e8-8ee5-d8336b5325c9.png">


## Project Managment: Trello

To manage my work throughout this process, I created a Trello board. This enabled me to srategically divide my work into small objectives and track my progress in completing each task in a timely manner. Additionally, I assigned specific tasks to certain categories (e.g., Website, Model, Data, Web-Scraping, etc.) so that I can easily visualize the overarching progress of each theme throughout time. Lastly, I've chosen to make this Trello board public because I want others to have the ability to track my progress (in detail) and, more importantly, hold me accountable for my work. Going off that note, you can find my live trello board here: https://trello.com/b/6XRlBmh5/wikipedia-article-quality-classifier



