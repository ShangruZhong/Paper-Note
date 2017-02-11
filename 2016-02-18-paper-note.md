---
layout: post
title: 《MATAR-Keywords Enhanced Multi-label Learning for Tag Recommendation》Paper Note
category: Recommendation 
tags: [tag recommendation, multi-label, paper]
image:
  feature: galaxybg.jpg
comments: true
share: true
---

Published on *APWeb* 2015, *Asia Pacific Web Conference*.

- [Paper download](http://link.springer.com/chapter/10.1007/978-3-319-25255-1_22)
- [Dataset: Stack Overflow & Mathematics Stack Exchange](http://blog.stackoverflow.com/category/cc-wiki-dump/)

---

## 1. Background

This paper also concentrates on **Tag Recommendation**.Just as mentioned in the paper, existing tag recommendation methods can be classified into two categories: 

- *collaborative-filtering* method 
	- relies on users' historical behavior
	- suitable to tag a relatively fixed set of items(e.g., music & movies)
- *content-based* method.
	- uses the content information, hence suitable for content-based applications(e.g., blogs & QA sites)
	- relative with *information retreval* or *NLP* task: *Entity Recognition*, *keywords extraction* & *sentiment analysis*. **(Personally)** 

Since the authors focus on the content-based applications, like *Stack Overflow*, they find around 70% tags have appeared in the content, and tags are potential keywords in the content.

Therefore, they propose a tag recommendation method called **MATAR**, combining two CB methods: first model TR problem as *multi-label learning* problem, then incorporate *keyword extraction* into it, which can boost the accuracy of recommendation. 

Further, they also speedup MATAR by employing the *locality-sensitive hashing strategy*.

As for **Multi-label Learning**, here's some review paper written by Zhou Zhi-hua:

- [*A Review on Multi-label Learning Algorithms*](http://cse.seu.edu.cn/people/zhangml/files/TKDE%2713.pdf)
- [*Multi-label Learning(Chinese version)*](http://cse.seu.edu.cn/people/zhangml/files/mla11-mll.pdf)

---

## 2. Related Work

The content-based method uses content itself as input.

In feature aspect, some studies find useful textual features: **tag co-occurrence** & **entropy**:

- Feng, Wang: *Incorporating heterogeneous information for personalized tag recommendation in social tagging systems*. (*SIGKDD 2012*)
- Xia, Lo, Wang, Zhou: *Tag recommendation in software information sites*. (*IEEE MSR 2013*, Mining on Software Repository) 
- Lu, Yu, Chang, Hsu: *A content-based method to enhance tag recommendation*. (*IJCAI,2009*)

In algorithm aspect, classification models & topic models are widely used.

- train classfication model for each tag, and recommend based on multiple classifiers:
	- Saha, Schneider: *A discriminative model approach for suggesting tags automatically for stack overflow questions*. (*IEEE MSR 2013*)
- use topic models find latent topics from content, and recommend based on topics:
	- Krestel, Fankhauser: Latent dirichlet allocation for tag recommendation. (*ACM RecSys 2009*)
- propose a variant LDA model to link tags with latent topics:
	- Si, X., Sun, M.: Tag-lda for scalable real-time tag recommendation. (*JCIS 2009*, Journal of Computational Information Systems)
- Multi-label learning: **e.g. TagCombine for TR, but suffers from the class balance problem**:
	- Xia, Lo, Wang: *Tag recommendation in software information sites*. (*IEEE MSR 2013)
- Keyword extraction: 
	- Murfi, Obermayer: *A two-level learning hierarchy of concept based keyword extraction for tag recommendations*. (*ECML PKDD 2009*)
	- Wang, Hong, Davison: *Rsdc’09: Tag recommendation using keywords and association rules*. (*ECML PKDD 2009*)
