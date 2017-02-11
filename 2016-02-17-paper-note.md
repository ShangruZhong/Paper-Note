---
layout: post
title: 《Automatic Tag Recommendation for Metedata Annotation Using Probabilistic Topic Modeling》Paper Note
category: Recommendation 
tags: [tag recommendation, LDA, paper]
image:
  feature: galaxybg.jpg
comments: true
share: true
---

Published on *JCDL* 2013, *The Joint Conference on Digital Libraries*.

- [Paper download](http://dl.acm.org/citation.cfm?id=2467706)

---

## 1. Background

***DataONE*** is a federated data network, harvesting **metadata** about *environmental & ecological science data* from different data providers and making it searchable via search interface ***ONEMercury***([https://cn.dataone.org/onemercury/](https://cn.dataone.org/onemercury/)). However, some harvested metadata records are poorly annotated or lacking meaningful keywords. 

Therefore, this paper develops algorithms for **automatic annotation** of metadata:

- First, transform the problem into a **tag recommendation with a controlled tag library**
- Then, propose an algorithm to recommend tags for metadata from the library

---

## 2. Related Work

---

## 3. Reserach Method

**Problem Definition:**

> Given a tag library **T** = <t1, t2, ..., tm>, and a document corpus **D** = <d1, d2, ..., dn> with **d** = <text, tags>, for a document without tags **q** = <text, None>, recommendation algorithm outputs a ranked list **Tk** with K tags relevant to the **q**.

**Solution:**

- For a document *q* without tags, calculate the **relevance scores** of all tags in library. **If a tag is annotated for several document, the more similarity between *q* & *d*, score of the tag should be higher.**
	- So the score of a tag *t*, should be decided by the similarity between *q* and the document which is marked with *t*.
	- Employ two different methods to represent the documents for similarity measurement, ***TF-IDF*** & ***LDA***
	- Choose **cosine similarity** as measurement for two document vector.
	- Compute the relevance scores of all tags, and define the normalization of score of a tag as the probability to *q*.
- Rank probability of all tags, choose the **top-K** ones as recommendation.

![](/assets/img/2016-02-17/probability.jpg)

![](/assets/img/2016-02-17/comment.jpg)

**Personally**,

- this method looks like the **user-based collaborative filtering** which recommends the similar users' preference on items. **Documents** are **users** while **tags** is similar to **items**. 
- Besides, both two can utilize vector to represent **documents/users**, then for **cosine similarity** calculation. 
- But differently in constructing vector, this paper choose bag-of-words with **TF-IDF** or topic vector with **LDA** for a document, while collaborative filtering generally takes the grades of items, which marked by users.

---

### 3.1 TF-IDF

Formally, given a term *t*, a document *d* of corpus *D*:

![](/assets/img/2016-02-17/tfidf.jpg)

---

### 3.2 LDA

![](/assets/img/2016-02-17/lda.jpg)

The LDA model is used to find *P(z\|d), the hidden topic distribution of document *d*(each topic can be semantically describe by the term *P(t\|z)*). Therefore, a document can be represented with a vector of topic, with probability as value:

![](/assets/img/2016-02-17/lda-vector.jpg)

---

## 4. Evaluation

### 4.1 Metrics

- *Precision, Recall, F1*
- *Mean Reciprocal Rank(MRR)*
- *Binary Preference(Bpref)*

---

### 4.2 Results

![](/assets/img/2016-02-17/result-1.jpg)

![](/assets/img/2016-02-17/result-2.jpg)

![](/assets/img/2016-02-17/result-3.jpg)

![](/assets/img/2016-02-17/result-4.jpg)





