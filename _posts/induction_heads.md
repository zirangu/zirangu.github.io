---
layout: post
title: "Understanding Induction Heads for Pythia_160"
Last Udpated Date: 2026-08-23 
categories: general
---

This post was motivated by materials covered in ARENA AI Safty Course 3.0, section 1.2 Introduction to Mechanistic Interprebility. 

When I started learning about technical AI Safety research, especially topics in mechanistic interprebility, I came across this concept called induction heads in the ARENA3.0 course material. I find this topic fascinating. It is a mechanic that sounds simple at first glance but really reflects one of the fundamental features of thinking/learning - recognizing patterns. 

I want to use this post to summarize my journey of learning about induction heads, and provide my thoughts on induction heads and what it means. I also try to replicate some of the results in ARENA 1.2regarding induction heads but with a different twist, so at least I creating something new rather than merely copying codes from the working notebook. 

On a high level, transformer-based large language model predicts the next words for a given text inputs by passing the tokenized and encoded inputs, as residual stream, into a series of transformer blocks, each containing several attention heads. We can think of heads analogous to different parts of a brain. Just like you have different regions of a brain attending to different functionalities such as listening, hearing and reasoning, heads in a transformer block serve different functions. It was observed by the research community that certain heads hold the ability to make inductive reasoning on predicting words based on its prior repetitive occurances. For example. Take the sentence "Joe Doe is the manager. Everyone loves Joe Doe." The induction head at the second "Joe" occurance would place a high attention score on the first "Doe", indicating a likely repetition of the phrase "Joe Doe". This observation is telling. This seems to characterize how LLMs learn names, locations, common phrases, and connections between things. 

In this post I implement the following mechanisms to identify attention heads: 1) induction score, 2) zero-ablation scores, and 3) mean-ablation scores. 

![Induction score by head](/assets/images/induction_score_by_head.png)

![Zero-ablation scores](/assets/images/zero-ablation.png)

![Mean-ablation scores](/assets/images/mean-ablation.png)


