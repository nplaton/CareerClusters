# CareerClusters

#### Motivation
Like many of the products I find useful, CareerClusters was built to address a need of its creator. The Data Science job search is filled with ambiguity, from what "Data Science" really means, to which skills are truly valuable for any given job. This product aims to clarify the different categories of careers that are available while also simplifying the job search itself. 

#### Designing the Product
Each element of a product should aid in accomplishing the true goals of the user. Discovering latent features within a career field may be interesting (to some), but is not inherently valuable. Throughout the evolution of the product three questions emerged as the most important to answer to create something valuable:

1. What kind of job do I want?

2. Where are these types of jobs available?

3. How do I maximize my chances of getting hired?


CareerClusters successfully addresses each of these needs, using three distinct services that can be used in succession or individually. Usability was a main concern, and all input spaces were designed to accept as much information as the user wants, from a few keywords to an entire resume.

#### Getting the Data
An important aspect of simplifying the job search is to give users access to data they could otherwise not find in one place. To properly survey the Data Science job landscape I built a custom scraper that lives in [main.py](https://github.com/LDinLA/CareerClusters/blob/master/main.py). The code was written to be self-contained, so it can be run at regular intervals to fetch the most current postings on the web. Raw text is scraped along with its metadata, to be stored in a DataFrame and accessed by the different services on the site.

#### Clustering
Latent categories are discovered using the [K-means clustering algorithm](http://scikit-learn.org/stable/modules/clustering.html#k-means) implemented using scikit-learn. To make the topics human understandable, N-grams from one to three words in length are returned which most describe each category. A typical cluster could be described by a three phrase array such as ['security', 'cloud security', 'engineering']. Five of these groups are displayed the the user to help understand which types of jobs are currently being displayed under the umbrella of Data Science.

#### Building a Search Engine
Topics are nice, but real examples are much better. A search engine is the connection between what the user asks for and what they actually want. My approach involved basic techniques from the field of information retrieval which transform the user's input and find the most relevant job postings for any search. Specifically, both the text from the postings and the user input are converted into sparse matrices using a [tf-idf vectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), and then compared using [cosine similarity](http://en.wikipedia.org/wiki/Cosine_similarity). This metric is more effective than Euclidean distance in high dimensional space, and accounts for differences in length between documents. Search is the most technically complex part of this product, and has the most room to be optimized in the future.

#### Getting the Job
An intimidating fact about the job search is that many companies analyze incoming resumes algorithmically before they ever reach a human. CareerClusters incorporates a "Resume Optimizer" which quickly compares the user's resume and a job posting, then displays which key words could be added or emphasized with an interactive graph. This feature helps individuals understand how their application could be interpreted by the business to which it is sent. The major drawback is having to find room in your schedule for more frequent interviews once you apply the optimizer to your own application process.

#### Measuring Success
This is an iterative project; the core features form a foundation on which to improve. Unstructured data such as job postings do not come with algorithmically measurable metrics, so quality is to be measured by a clear delivery of value and a useful user experience. In particular, the search engine returns relevant results, but could use an effective filtering mechanism (for location, programming languages). Search speed is also an important factor for a modern website, and can continuously be improved. The clustering and search algorithms get the job done, but have not been tested against all alternatives or using all possible parameters. The site will continue to improve with user feedback, and constant experimentation. 
