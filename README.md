# A Question-Answering System from Reviews!

Hi! This is my diploma thesis that has as a goal to create a question-answering system that automatically answers questions using Yelp's Tips and Yelp's Reviews as a knowledge base.

# Summary

The question answering system is constructed by many parts. The **Preparatory stages** are the first part. The first preparatory stage the **Corpus generation** is responsible for generating the Corpus files from Yelp’s Tips and Yelp’s Reviews by passing their Documents from a length filter. The second preparatory stage the **Centroids and index files generation** is responsible for generating needed intermediate files that speed up the system. The **Retrieval models** are the second part. They are responsible for retrieving Documents from Corpus, in order to later use them for answers’ generation. We have implemented three retrieval models; the **TF-IDF Matching Score**, the **Word Vector Centroid Similarity (WCS)** and the **IDF Re-weighted Word Vector Centroid Similarity (IWCS)**. The third and final part is the **Answers’ generation**. It generates the actual answers that our system returns based on the retrieved Documents. We have two answer generation methods. The first is to break every Yelp’s tip and Yelp’s review into sentences and use them as our Corpus. The second way of approaching this issue is to use the BiDAF model for this task an already well known Q&A model.

After the creation of our system our next task is to evaluate our system. For this task we have collected a bunch of questions and answers from Yelp’s community. With these questions we generate a sufficient number of answers by using all possible combinations of Corpus files, retrieval models and answers’ generation ways. After generating our answers, we have evaluated them with two evaluation strategies and by using a number of evaluation metrics. The evaluation strategies are the **Evaluation based on Yelp’s answers** and the **Evaluation based on manual labeling**, and the evaluation metrics are the **MRR**, **MAP**, **NDCG** and **DCG**.

Finally we show same examples of our system’s answers, make conclusions about our system’s algorithms, conclude on the best one and talk about possible future work. Our results show that by choosing **Yelp’s Reviews with the WCS retrieval model and without the BiDAF model our system generates the best answers.**

![Preprocess1](https://user-images.githubusercontent.com/10975341/144014570-4232695f-58ed-46d2-b949-40be1fdb784f.png)
![Compute1](https://user-images.githubusercontent.com/10975341/144014105-c06c357a-2a85-40cb-ae94-207a89d9cb32.png)
![Pass1](https://user-images.githubusercontent.com/10975341/144014624-f9e53c5a-6263-43eb-bb86-b99179e1bcd0.png)
