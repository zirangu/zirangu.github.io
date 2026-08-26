---
layout: post
title: "Understanding Induction Heads for Pythia_160"
date: 2026-08-23 12:00:00 -0000
categories: general
---

This post was motivated by materials covered in ARENA AI Safty Course 3.0, section 1.2 Introduction to Mechanistic Interprebility. 

When I started learning about technical AI Safety research, especially topics in mechanistic interprebility, I came across this concept called induction heads in the ARENA3.0 course material. I find this topic fascinating. It is a mechanic that sounds simple at first glance but really reflects one of the fundamental features of thinking/learning - recognizing patterns. 

I want to use this post to summarize my journey of learning about induction heads, and provide my thoughts on induction heads and what it means. I also try to replicate some of the results in ARENA 1.2regarding induction heads but with a different twist, so at least I creating something new rather than merely copying codes from the working notebook. 


# What are Induction Heads
On a high level, transformer-based large language model predicts the next words for a given text inputs by passing the tokenized and encoded inputs, as residual stream, into a series of transformer blocks, each containing several attention heads. We can think of heads analogous to different parts of a brain. Just like you have different regions of a brain attending to different functionalities such as listening, hearing and reasoning, heads in a transformer block serve different functions. It was observed by the research community that certain heads hold the ability to make inductive reasoning on predicting words based on its prior repetitive occurances. 

For example. Take the sentence "Joe Doe is the manager. Everyone loves Joe Doe." The induction head at the second "Joe" occurance would place a high attention score on the first "Doe", indicating a likely repetition of the phrase "Joe Doe". This observation is telling. This seems to characterize how LLMs learn names, locations, common phrases, and connections between things. 

In this post I implement the following mechanisms to identify attention heads: 1) induction score, 2) zero-ablation scores, and 3) mean-ablation scores. 

# Finding Induction Heads in a model.

We use Pythia_160 for model. Here is the detail of model spec.

| Spec | Value | 
|----------|----------|
| Layers  | 12  | 
| d_model   | 768   | 
| attention heads | 12 |
| d_heads | 64 |
| context length | 2048 tokens |
| positional embeddings | RoPE  |
| attention mechanism | parallel attention & MLP computation |

It would be difficult to visualize the attention scores assigned to every head across every layer for every token in the sequence(12 * 12 * [seq_length]). Therefore the first motivation is to come up with a score per head that measures the strength of induction exhibited by that head, averaging over all the tokens in a given batch.

# Calculating Induction Scores

What does it mean for an attention head to exhibit induction functionality? Suppose we take a hypothetical sequence "It is Friday", and supposed it is broken down into 3 tokens "It", "is" and "Friday". We construct a repetitive sequence with the following convention: 

| 0 | 1 | 2 | 3 | 4 | 5 | 6|
|---|---|---|---|---|---|---|
| [POS] | It  | is| Friday | It  | is| Friday |

where [POS] is a position identifier that takes up one token.
Suppose this newly constructed sequence is at position 5, the second "is". An induction head would assign a high score to the word after the first "is" which is in this case the first "Friday", so that it is highly probable that the next word in the sequence after the second "is" is "Friday". 

Inspired from the examples above, to build the induction scores, we do the following. We first construct 20 batches of repeated sequence length of size 100 (constructed sequence length = 201). For each attention head, we average the attention score, evaluated at current position i, to past position i - (1 - seq_len), then averaged over all positions in the batch sequence and over all batches. We will then have a induction score calculated for every layer-head combination that measures the likelihood of that head being an induction head, which is displayed below.


![Induction score by head](/assets/images/induction_score_by_head.png)

# Zero-ablation

![Zero-ablation scores](/assets/images/zero-ablation.png)

# Mean-ablation

![Mean-ablation scores](/assets/images/mean-ablation.png)
