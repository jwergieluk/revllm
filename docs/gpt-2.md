---
title: "RevLLM"
lang: 'en-US'
---

## Introduction

RevLLM is a Python library designed to facilitate the analysis of Transformer
Language Models, particularly focusing on generative, decoder-type
transformers. Our library aims to democratize the access to advanced
explainability methods for data scientists and machine learning engineers who
work with language models. Built on top of Andrej Karpathy's esteemed nanoGPT,
RevLLM stands as a robust and user-friendly tool in the field of natural
language processing.


## Our model: nanoGPT 

We construct a model nanoGPT, following the example given by Andrej Karpathy [(Github)](https://github.com/karpathy/nanoGPT).

- nanoGPT comes in 4 sizes: _regular_, _medium_, _large_, and _extra large_. 
- All 4 sizes have the same architecture, but differ in weights and dimensions.
- Like most LLMs, nanoGPT is a variant of the transformer model introduced in [2017](https://arxiv.org/pdf/1706.03762.pdf).
- nanoGPT has the same architecture and weights as [gpt2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), which is trained on a dataset of 8 million web pages. 
  The out-of-the-box model can be imported from [Hugging Face](https://huggingface.co/docs/transformers/model_doc/gpt2).

We focus on nanoGPT for its simplicity and ease of use, as the smaller versions can be run locally on a cpu. The goal of this project is to produce an easily accessable version of analysis with low barriers to entry. Current tools can be later adapted to larger models.

### Architecture 

Differences between nanoGPT versions lie in the numbers of layers, embedding dimensions, and weights of individual components (see the "architecture" tab under the individual models).  However, their architectures are identical. This section describes the models with dimensions excluded. We assume the reader is familiar with basics of a neural network such as linear, activation, and dropout layers, hence they are listed without comment.  

For what follows:

- Let context be a string with length $c$ (context is also referred to as "prompt" in the app's pages).
- Let $n$ be the embedding dimension for choice of model size.
- A tensor is a generalization of a vector.  Our model employs pytorch tensors.
- Any discussion of batching or batching dimension is excluded.
- Several steps described include matrix multiplications with learnable weights. We describe the functionality omitting such details. Exceptions are found under the "Transformer-Specific Details." 
- Dropout layers are included, but are ignored during model inference.
- Layers will follow the format: 
  - <u>Layer description</u> `(layer_name)`, for elements of the architecture
  - <u>[Layer description]</u>, for elements of the work flow, but not the architecture

#### Preprocessing

- <u>[Tokenization]</u>: The tokenizer separates context into a $c$-length list of discrete tokens, one of 50257 choices (the model's vocabulary length).
- <u>[Input_ids]</u>: a $c$-length tensor is populated by each token mapped to an integer from $[0,50256]$.
- <u>[Position_ids]</u>: A tensor $[0,1, \ldots, c-1]$ is generated.

#### Embedding Layers

- <u>Token embedding</u> `(wte)`: Each input_id in input_ids is mapped to a tensor in $\mathbb{R}^n$.
  - Note: the path $(x' + \alpha (x - x'))$ for integrated gradients below is in this embedding space.
  - For some comments about the matrix $M$ in the embedding layer, see the weight tying section.
- <u>Position embedding</u> `(wpe)`: Each position_id in position_ids is mapped to a tensor in $\mathbb{R}^n$.
- <u>[Additive step]</u>: x = token_embedding + position_embedding
- <u>Dropout</u> `(drop)`

#### Transformer Block

The number of repetitions of the transformer block is dependent on nanoGPT size.

  - <u>Layer Norm</u> `(ln_1)`
  - <u>Self-Attention</u> `(attn)`:
    - <u>Attention Initialization Layer</u> `(c_attn)` 
      - <u>[Multi-Headed Self-Attention Sequence]</u> (see the discussion below):
        - $Q_h, K_h, V_h$ are generated by c_attn(x), for $1 \leq h \leq H$, where $H$ is the number of heads.
        - $A_h = \text{softmax}\left(\frac{Q_hK_h^T}{\sqrt{n_h}}\right)$
        - x = Concat $(A_1V_1, \ldots, A_HV_H)W^O$
    - <u>Linear</u> `(c_proj)`: output projection
    - <u>Dropout</u> `(attn_dropout)`
    - <u>Dropout</u> `(resid_dropout)`
  - <u>[Additive step]</u>: x = x + attn(x)
  - <u>Layer Norm</u> `(ln_2)`
  - <u>Feed Forward</u> `(MLP)`
    - <u>Linear</u> `(c_fc)`
    - <u>Activation</u> `(GELU)`
    - <u>Linear</u> `(c_proj)`
    - <u>Dropout</u> `(dropout)`      
  - <u>[Additive Step]</u>: x = x + MLP(x)

#### Layer Norm

- <u>Layer Norm</u> `(ln_f)`

#### Language Model Head

- <u>Language model head</u> `(lm_head)`: Token embeddings in dimension $n$ are converted to vocabulary-length tensors (see discussion below on weight tying).

#### Final Interpretation

Upon the output of the `(lm_head)` layer, we are given the raw $c \times 50257$ tensor ouput of the model, or "logits."  There are some final steps to completed for interpretations:

- Apply softmax row-wise to transform each row into a probability space.
- Translate to natural language. The exact next-token selection is governed by a temperature parameter:
  - Increasing the temperature parameter effectively smooths out the probability distribution, making the selection of various outcomes more uniform. This increase in uniformity raises the level of randomness or entropy in the outcomes.
  - Lowering the temperature sharpens the probability distribution, which biases the selection towards more probable outcomes, thereby reducing randomness and leading to more deterministic outputs. Thus, a temperature very close to zero would predominantly choose the most probable outcome (token).

### Transformer-Specific Details

#### Self-Attention

We include a brief description of self-attention as it functions in our model.  For a full description, see the foundational [2017 paper](https://arxiv.org/pdf/1706.03762.pdf).

##### Single Headed Attention

Suppose for the moment that the model has $H=1$ head (i.e. the embedding space is not subdivided during the attention mechanism).  The `(c_attn)` layer contains matrices $W^Q, W^K, W^V, W^O$, with learnable weights. Upon multiplication with the first three matrices, x is mapped to query, key, and value matrices $Q, K, V \in \mathbb{R}^{c \times n}$. The attention matrix is then $A = \text{softmax}(S) \in \mathbb{R}^{c \times c}$ computed by applying softmax row-wise to the matrix $S$ defined by:

$$
S_{i,j} := 
\begin{cases} 
-\infty & \text{if }  i < j\\ \\
\left(\frac{QK^T}{\sqrt{n}}\right)_{i,j} & \text{otherwise}
\end{cases}
$$

The final output of the self attention layer is: x = $(AV)W^O$.

Upon application of softmax:

- <u>$A$ is Lower-Triangular</u>, since arbitrarily large negative values of $S$ are a masking condition, as they map arbitrarily close to $0$ under softmax.  The entry $a_{i,j}$ numerically represents a relationship or 'attention' the model pays from token $i$ to token $j$.  When predicting a next token, the model can only attend to previously seen elements and not to future ones, hence they are nonzero only when $i \geq j$.
- <u>The rows of $A$ are probability vectors</u>: The multiplication $AV$ therefore yields a weighted sum of the values in $V$ for each token. The current attention focus can be thought of as being distributed across the input context.

Additional notes (for further explanation, see the 2017 paper linked above):

- With this lower triangularity, nanoGPT is a "decoder-only" model, to be distinguished from models which contain an encoder block.
- The scaling factor $\frac{1}{\sqrt{n}}$ is an important performance-improvement feature for softmax use (here we omit details).
- In general, instead of the term $\sqrt{n}$, transformer architectures typically discuss $\sqrt{d_k}$.  It is not a requirement that the $Q, K, V$ matrices have a dimension equal to the embedding dimension.  But for our model, they will.
- In general transformers, $V$ may not have identical dimensions to $Q$ and $K$.
- "Self-attention" refers to the fact that $Q, K,$ and $V$ all come from the input x.  Alternatively, $K$ and $V$ can come from alternate sources, as in "cross-attention."

##### Muli-Headed Attention

In reality, for transformer models the self-attention mechanism subdivides the embedding space into $H > 1$ distinct subspaces called "heads." The `(c_attn)` layer has matrices $W_h^Q, W_h^K, W_h^V,$ and also contains the full-dimensional output matrix $W^O$. Through the first three, x maps to matrices $Q_h,K_h,V_h \in \mathbb{R}^{c \times n_h}$, for $1 \leq h \leq H$, where $n = \sum_h n_h$.


In each size of our nanoGPT, $H$ divides $n$.  For every $h$, $n_h = \frac{n}{H}$, and $A_h = \text{softmax}(S_h) \in \mathbb{R}^{c \times c}$, where: 

$$
\big(S_h\big)_{i,j} := 
\begin{cases} 
-\infty & \text{if }  i < j\\ \\
\left(\frac{Q_hK_h^T}{\sqrt{n_h}}\right)_{i,j} & \text{otherwise}
\end{cases}
$$

The final output of the self attention layer is: x = Concat $(A_1V_1, \ldots, A_HV_H)W^O$.

Benefits of multi-headed attention include:

- <u>Increased capacity</u>: By allowing the model to process different representations or features of the input data in parallel, each head can "focus" on different aspects of the input, enhancing the model's ability to capture complex dependencies and nuances.
- <u>Computational efficiency</u>: The computation of the attention matrix can be parallelized across heads. 

#### Weight Tying of Embedding and Output Projection Matrices

Our model has a learned embedding matrix $M \in \mathbb{R}^{50257 \times n}$.

- <u>Embedding Matrix Lookup</u>: In the embedding `(wte)` layer, the input_ids contain $c$-many indices,  and the $c \times n$ resulting matrix is constructed from the corresponding rows of $M$.
- <u>Output Projection</u>: In the `(lm_head)` layer, the $c \times n$ existing state is projected to the vocabulary space by transposing the same matrix: $M^T \in \mathbb{R}^{n \times 50257}$.

In other words, the input and output matrices are the same, just transposed.  This matrix repetition is a performance improvement feature known as [weight tying](https://arxiv.org/pdf/1608.05859v3.pdf), and is common in transformer architectures.

Additional note: position embedding `(wpe)` is constructed similarly to `(wte)`, but through a different lookup matrix.

# Analysis Methods

## Feature Attributions

Feature attribution is a feature importance method for machine learning models.   Properties include:

- <u>Post-hoc</u>: obtained after a model has been trained.
- <u>local scoring system</u>: It assigns a score for each individual feature in isolation.
- <u>Values are from $\mathbb{R}$</u>:  In contrast, global features  or other feature importance methods typically measure only strength of influence, hence are nonnegative. Attribution behavior is analogous to correlation or linear coefficients:
    - A negative score indicates the presence of the feature decreases the predicted value of the target.
    - A positive score indicates the presence of the feature increases the predicted value of the target.
    - The magnitude of the value indicates the strength of the feature's influence.
- <u>Context Dependent</u>
- <u>Independent of Accuracy</u>


### Limitations

- <u>[Methods often disagree](https://arxiv.org/pdf/2202.01602.pdf)</u>
- <u>Scores may not make sense</u>, and require interpretation.
- <u>Imperfect approximations</u>
- <u>Computationally expensive</u>


### Feature Attribution Methods

Methods to compute feature attributions include: CAM, Grad-CAM, LIME, Integrated Gradients, DeepLIFT, SHAP, SmoothGrad, Anchors, CEM, This looks like that, XRAI, and Contrastive Explanations, among others.  Furthermore, several methods have variants for different contexts. 

Some properties different methods may or may not satisfy include:

- <u>Model agnostic vs non model agnostic</u>: Both types exist.
- <u>Requirement of a baseline</u>: Some methods require a baseline input vector in $x' \in \mathbb{R}^n$ for reference. 
  - This issue implies a sensitivity to baseline, as a poor choice can yield unhelpful scores.
  - In image processing, $x'$ is often the representation of a blank screen.
  - In our scenario, $x'$ will be context-length many embeddings of the token_id $0 \in \mathbb{Z}$ (note: this representation is learned through model training, and is not simply the vector $0 \in \mathbb{R}^n$).
- <u>Target dependence for a classifer model</u>:  if a model $F$ is a classifier, some methods require a specific ***target*** class to be chosen.  Hence $F: \mathbb{R}^n \rightarrow [0,1]$ enables requisite functional behaviors.
  - The predicted class is the natural target choice, though other choices can yield insightful results. 
  - For example, suppose a model is pretrained on the [MNIST dataset](https://cgarbin.github.io/machine-learning-interpretability-feature-attribution/#example-with-mnist).  For each target $0$ through $9$, every pixel can be scored for how strongly it contributes to the prediction of that target.  
  - In our scenario, targets will be the predicted next token (out of vocabulary-length choices).

### Our included method: Integrated Gradients

Integrated gradients (IG) is a baseline and target dependent feature attribution method which satisfies certain [axiomatic properties for feature attributions](https://arxiv.org/pdf/1703.01365.pdf).  As the name implies, it depends on the differentiability of the model, hence is _not_ model agnostic.

Let $x, x' \in \mathbb{R}^n$, for input $x$ and baseline $x'$.  Let $F: \mathbb{R}^n \rightarrow \mathbb{R}$ be an almost-everywhere differentiable model such as a neural network. IG computes the following path integral for the straight line from $x'$ to $x$: 

$$
\text{IG}_i = (x_i - x_i') \int_{\alpha=0}^{1} \frac{\partial F\big(x' + \alpha (x - x')\big)}{\partial x_i} \ d\alpha
$$

- Note: as our input layer is copies of $\mathbb{Z}$, we must construct this path in the images in the embedding space.  This method is actually "layer integrated gradients" with a choice of the embedding layer.

#### Discrete Approximation

In practice, the computation of IG is equal to a Riemann sum (in our example right):  

$$
\text{IG}_i \approx (x_i - x_i') \times \sum_{k=1}^{m} \ \frac{\partial F\big(x' + \frac{k}{m} \times (x - x')\big)}{\partial x_i} \times \frac{1}{m}
$$

Where the number of steps by default is $m=50$.

#### Our Implementations

For text data embedded in $\mathbb{R}^n$, the features $i$ are the dimensions. For each token, we therefore have $n$-many scores. Our final integrated gradient score for that token is the average of these values.

Sequential data, however, add a further complication. In the above description of IG, the computation is performed for a $1 \times n$ input and baseline. We have a $c \times n$ context (prompt) input, and a $1 \times n$ token baseline (the embedding for the token $0$). If we are analyzing a token indexed by $t$, the question arises about how to set our context baseline.  We implement two strategies:

- Integrated Gradients (IG) typically sets the context baseline as $c$-copies of the baseline token.

- [Sequential Integrated Gradients](https://arxiv.org/pdf/2305.15853.pdf) (SIG) is an alternate approach which sets the context baseline as the full context with the baseline token inserted only at index $t$. 

In both cases, the integral is computed only at token $t$. SIG has the disadvantage of significantly increased computational cost. However, both approaches are in use and highlight different aspects of how an isolated token influences the model's prediction. We therefore include both methods in our implementation.

## Logit Lens

Recall that the term "logits" is commonly used to describe the final raw output (here, the final transformed embeddings) of a model, before any softmax transformation and subsequent translation to a prediction (here, natural language). We would also like to extract the middle embeddings in order to view the progression of what the model believes. With this motivation, we implement a modified version of [logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens), which performs this extraction and processes it for next token generation.
 
After each additive step in the architecture described above, we extract the $c \times n$ embedding tensor at that point. We then pass them throught the final interpretation layers (layer norm, `lm_head`) to view the local next token generation:

- `h_00` applies `ln_f`, then `lm_head` to x = token_embedding + position_embedding
- Within each block:
  - `h_block_middle` applies `ln_2`, then `lm_head` to x = x + attn(x)
    - This display is optional and can be toggled on or off. It was inspired by a step of [lm_debugger](https://arxiv.org/pdf/2204.12130v2.pdf).
  - `h_block_out` applies `ln_f`, then `lm_head` to x = x + MLP(x)

### Sub-Token Predictiton

The visual display of logit lens contains a prediction along with every word of the context.  Due to the sequential nature of how models like nanoGPT work, predictions are generated for every sub-context which starts from the first word.  The final model prediction is the last of all such predictions.  As the logit lens display shows its evolution, it does so for each of these sub-token predictions as well.

# References

- Enguehard, Joseph. “Sequential Integrated Gradients: a simple but effective method for explaining language models.” ArXiv abs/2305.15853 (2023)

- Garbin, C. (n.d.). A gentle introduction to the concepts of machine learning interpretability, feature attribution, and SHAP. Retrieved from (https://cgarbin.github.io/machine-learning-interpretability-feature-attribution)

- Geva, Mor, Avi Caciularu, Guy Dar, Paul Roit, Shoval Sadde, Micah Shlain, Bar Tamir and Yoav Goldberg. “LM-Debugger: An Interactive Tool for Inspection and Intervention in Transformer-Based Language Models.” ArXiv abs/2204.12130 (2022)

- Karpathy, A. nanoGPT [Software]. GitHub. (https://github.com/karpathy/nanoGPT)

- Krishna, Satyapriya, Tessa Han, Alex Gu, Javin Pombra, Shahin Jabbari, Steven Wu and Himabindu Lakkaraju. “The Disagreement Problem in Explainable Machine Learning: A Practitioner's Perspective.” ArXiv abs/2202.01602 (2022)

- Press, Ofir and Lior Wolf. “Using the Output Embedding to Improve Language Models.” Conference of the European Chapter of the Association for Computational Linguistics (2016)

- Radford, Alec, Jeff Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever. “Language Models are Unsupervised Multitask Learners.” (2019). OpenAI Blog.

- Sundararajan, Mukund, Ankur Taly and Qiqi Yan. “Axiomatic Attribution for Deep Networks.” International Conference on Machine Learning (2017)

- Vaswani, Ashish, Noam M. Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin. “Attention is All you Need.” Neural Information Processing Systems (2017)

- nostalgebraist. 2020. interpreting gpt: the logit lens. (https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)