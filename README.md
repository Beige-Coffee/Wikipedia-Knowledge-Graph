# Wikipedia-Knowledge-Graph
Using 30,000 hand-graded Wikipedia articles and NLP to predict the quality of Wikipedia articles and create a knowledge graph that identifies both articles and topics in need of editorial and epistemological attention.


Please note that this is a work-in-progress. 

Thanks,
- Austin (Aug, 2018)


## In A Nutshell
With roughly 50 million articles in Wikipedia's corupus, once can imagine how daunting of a task it must be to find the optimal balance between quality and quantity. For example, if you take a look at the following two articles, you can begin to see how disparate the range of quality is across Wikipedia articles. 
- High quality article: https://en.wikipedia.org/wiki/Elizabeth_II
- Low quality article: https://en.wikipedia.org/wiki/Ring-tailed_cardinalfish

Now, at first thought, you may think that this difference in quality is not only understandable, but expected. And, if this was your thought, you'd be right. However, the more nuanced point is:
- 1) These articles were hand-selected to illucidate the point that a difference of quality exists
- 2) This difference in quality exists across many articles and, therefore, is not easy to readily detect

However, the advent of machine learning and predictive modeling has done a lot to level the scale between quality and quantity. In hopes of contributing towards this endeavor and, *just as importantly*, **building a project that can be intuitively understood by any wikipedia editor**, I have built a machine learning model to predict the quality score of a Wikipedia article and, subsequently, organize article predictions by category on a user-friendly website.


## Motivation
The purpose of LifeStyleDesign is to challenge the user (in this case, a student) to take part in actively building their life. In doing this, the user is forced to confront new obstacles that they may have otherwise overlooked and, through this process, they achieve a deeper level of insight into the objective at hand. For all of the aforementioned reasons, I built LifeStyleDesign to empower individuals to take hold of the data at hand and leverage it to model *all the possible worlds* they may inhabit. In this case, I have modeled **116** distinct worlds to gain insight regarding an obstacle that many students in today's economy face - namely, **debt**. 

Over the course of the coming weeks, I will be cleaning up the code so that it is more usable and interpretable as well as commenting on further insights that I tease out of this exercise. However, I strategically chose to make my work public because I firmly believe that, in this case, the philosophy behind the project is just as important as the code behind the project. With a bit of an Existentialist flair, the purpose of this endeavor is for a user to strategically and analytically evaluate the choices at hand and make strides to achieve a predetermined goal. This, in essence, is LifeStyleDesi 

## Quick-View Of Parameters
  - Cost-of-living for hand-chosen cities
  - My specific monthly expenditures (cups of coffee per month, gym memberships, rent, gas, groceries, etc.)
  - Predicted salary for entry-level data scientist (I also experimented with data analyst salaries) - specific for each city
  - Income tax for each city

## Detailed Look Into Parameters
To develop my model, I use Selenium to scrape the following information off the internet:
  - Cost-of-living data for 29 predetermined cities
    - This number is somewhat arbitrary. I'll admit that I am not actually considering moving to all 29 cities, however, I was mainly interested to see how each city compared to its counterparts across the country.
    - Cost of living data was scraped from www.numbeo.com/cost-of-living/
  - The user-specific monthly expenditures that characterize my lifestyle
    - I manually sifted through the scraped cost-of-living data so that I could create a **monthly constant** variable which roughly reflected my personal expenditures. This allowed me to calculate my personal cost-of-living for each predetermined city. For example, some of the monthly constant I included where:
      - Number of Cups of Coffee Per Month
      - Taxis/Ubers 
      - Gas
      - Rent (including water, electricity, etc.)
      - And more...
- Data Science salaries for each of these 29 cities
  - An important design decision was made here. Namely, I chose to take the **lower end** of the range for data science salaries. I did this because I am relatively fresh out of college and just beginning my journey into data science. Therefore, I did not want to assume my salary to be closer to the average (which is usually around $100,000) and skew my model.
   - All salary data was scraped from Glassdoor.com

- To further increase the accuracy of my predictions, I chose to include each city's state income tax so that I could accordingly adjust my income for each city. As of now, I manually coded each state's income tax, standard deduction, and personal exemption. For the purposes of my model, I only included the information that reflected my predicted salary in each city. In the coming weeks, I will work to make a function that can do this algorithmically.
    - This information can be found here: https://files.taxfoundation.org/20180315173118/Tax-Foundation-FF576-1.pdf


## Data Visualization
Let's look at some graphs!
<img width="834" alt="screen shot 2018-08-12 at 11 06 54 am" src="https://user-images.githubusercontent.com/34213201/44036537-de037e1e-9ec6-11e8-94af-adc1ba20a60a.png">

I executed some calculations on the aforementioned data points and was able to graph the number of years it would take me to pay off my debt. To do this, I calculated my monthly net income for each city and assumed I would allocate roughly 30% to debt. Subsequently, I determined how long it would take me to pay off my debt, assuming that I chose to live in a 1-bedroom apartment in the center of the city. This graph displays the resulting calculations. A quick note: it's interesting how we all *know* that NYC is extremely expensive, however, it resonates on a deeper level when you actually see NYC *tower* over all other cities. It just goes to show how powerful and ingrained human vision is in our emotions.



## Things I am working on...

As of now, my model is relatively stagnant. By this, I mean that my parameters do not *change over time*. While this certainly does not render the model useless, it does affect its ability to accurately reflect reality. To reiterate the profundity of *change*, I will remind you of a quote by Heraclitus of Ephesus (c. 500 BCE), an ancient Greek philosopher: **“The Only Thing That Is Constant Is Change"**. Using Heraclitus as my guide, I plan on scaling salaries over time, and, unfortunately (for my wallet), finding a way to algorithmically incorporate interest on student loans. 

Another short-term goal of mine is to account for each user's specific monthly debt payment. I believe this is crucial because, simply put, it may show that one city simply cannot be considered as an option for the user to move to. To further illustrate this point, please take a look at the image below.

![image](https://user-images.githubusercontent.com/34213201/44205615-ce245580-a10b-11e8-8f8f-5034d127119b.png)

**Above:** Graph depicts number of years to pay off debt - assuming I live in a 1-Bedroom apartment in the city-center and earn that city's average salary for a data analyst.

In the scenario above, you can see that it would take my approximately 195 years to pay off my debt if I moved to NYC. While this is obviously impossible (lenders would never let this happen), it does reflect something true about reality. Namely, the assumption that I originally built into my model to bootstrap it off the ground was that I would put 30% of my net income towards debt every month - and this simply won't work if that amount turns out to be **less** than my minimum payment each month. Furthermore, I looked at the data behind this graph and found out that my net monthly income (after living expenses, food, fun, etc.) in NYC was less than $100 - making it obvious why it would take so long to pay off my debt. Now, here is the important part! Given this information, living in a 1-Bedroom apartment in NYC would simply not be an option for me if I were to make the average data analyst salary. Now, I feel obligated to mention that I was not expecting a 1-Bedroom apartment in NYC to actually be a reality right out of college. However, I do think that this extreme example precisely reflects the subtle point that I am seeking to convey - namely, that a less extreme version of this situation undoubtedly exists for most people who would consider moving to many different cities. 

## Sneak Peak...

<img width="785" alt="screen shot 2018-08-12 at 3 33 54 pm" src="https://user-images.githubusercontent.com/34213201/44036543-e07607ca-9ec6-11e8-885b-8c3e34362e7e.png">

**Above:** Graph depicts number of data science meetups in each city.

In the coming weeks, I will also be experimenting with Meetup.com's API so that I can gather information on the number of Meetup groups in each city, aggregated by topic. The goal is to pair this data with the previously calculated cost-of-living information to form a more accurate characterization of each city. For example, if I was interested in finding a city that had a profusion of data science meetups that I could attend to learn new skills, meet other professionals in the industry, and, more generally, just be surrounded by like-minded people, then using this data would serve as an insightful proxy to that objective.


