---
layout: post
title: Review for Tag Recommendation Algorithm
category: Recommendation 
tags: [tag recommendation, paper]
image:
  feature: galaxybg.jpg
comments: true
share: true
---

Tag Recommendation问题既有信息检索领域的文本挖掘(内容提取)，又涉及到目前流行的个性化协同推荐问题(定向广告、兴趣发现)。信息检索的理论可以基于内容实现(文本)特征提取，协同方法则用于群体中的推荐问题。

**个性化标签推荐问题**，本质上是个**User-Resource-Tag(tuples)**问题，既不能单纯理解为**文本关键字提取问题**(这类问题不考虑user，缺少Personality)，也不能完全归为**协同过滤推荐问题**(Tag也会源于resource内容本身)。

---

## Reviews of Related Work

早期的TR也是基于协同过滤方法，计算user、resource、tag的相似度。同时也会结合TFIDF来提高性能。

- G. Mishne. "Autotag: a collaborative approach to automated tag assignment for weblog posts." In WWW 2006.(基于相似的weblog的tag作为推荐)

- P. A. Chirita, S. Costache, W. Nejdl, and S. Handschuh. "P-tag: large scale automatic generation of personalized annotation tags for the web." In WWW 2007. (Chirita基于桌面相似文档的tags推荐给web资源)

- Z. Xu et al. "Towards the semantic web: Collaborative tag suggestions." In Proceedings of Collaborative Web Tagging Workshop at WWW 2006. (推荐descriptive标签)

后来提出基于图(graph-based)的算法如Folkrank，通过URT(user-resource-tag)图间接考虑tags的同现关系。基于URT图，也有使用SVD方法对向量/张量进行降维

- R. Jaschke et al. "Tag recommendations in folksonomies". In PKDD 2007.
- P. Symeonidis et al. "Tag recommendations based on tensor dimensionality reduction". In RecSys 2008.

交互性的方法，user给一个新的resource打上tag，算法可以根据历史的tag同现关系进行推荐。

- N. Garg and I. Weber. "Personalized, interactive tag recommendation for flickr". In RecSys 2008.

Shepitsen提出resource recommendation system，使用用户的profile和基于标签空间的层次聚类(hierarhical clustering)方法来推荐resource。

- A. Shepitsen et al. "Personalized recommendation in social tagging systems using hierarchical clustering". In RecSys 2008.

Associative rules关联法则也被用在标签推荐问题上:

- P.Heymann et al. "Socail tag prediction". In ACM SIGIR 2008.

随着主题模型的盛行，Krestel最早使用LDA算法从语义空间上推荐隐含主题的标签。

- Krestel and Fankhauser. "Latent dirichlet allocation for tag recommendation". In RecSys 2009.

对于标签质量的评价方法，Krestel和Chen提出一种标签对资源的描述质量高低的方法。

- R. Krestel and L. Chen. "The art of tagging: Measuring the quality of tags". In ASWC 2008.

对于标签使用的统计研究

- 对于Delicious的evaluation研究发现，有一半被标记的页面，其内容中包含有tag<br/>P. Heymann et al. "Can social bookmarking improve web search?" In WSDM 2008.
- Descriptive的标签(如topic/type)比personal的标签使用得更频繁，尤其在中低频范围内的标签里，这些标签也常被用于检索。<br/>K. Bischoff et al. "Can all tags be used for search?" In CIKM 2008.

---

## 对Tags的理解

个人认为，标签在不同使用场景下，应该分开讨论。

当标签用在搜索引擎优化搜索、文本分类等相对客观的信息检索领域，给文本进行标签推荐的问题就可以类比为**Keyword Extraction**，标签类比为关键字，这要求从文本内容中直接提取标签，又或者基于主题模型产生表达语义的标签。这种情况下，标签不具有表征个人喜好的personality。

当标签用在表达用户爱好行为等的UGC社交网站时，给内容推荐标签就不只是关键字提取的问题，这时就要充分结合社交网络的关系，更多在"user-resource-tag"上进行协同推荐。这种情况下，标签从用户爱好的角度表示内容，比如"刺激、惊悚"。这类推荐的情景源于用户热衷于用自己个性化的词来给内容打标签来分类，也就是所谓的**Folksonomy**大众分类法。

如今所研究的标签推荐问题，更多是基于后者的社会标签推荐问题，既有基于内容的文本挖掘，也有协同学习的大众分类法。因此标签的组成应该包括这两个角度，一方面源于内容的语义，另一方面源于用户的行为爱好。对于非文本的资源(如图片电影音乐)，标签还可以来源于资源的metadata，如一些资源的一些基本信息(作者、时间)。

从多个角度结合产生标签的算法，理应能得到较为准确的标签。

