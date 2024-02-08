## Introduction

RevLLM is a Python library designed to facilitate the analysis of Transformer
Language Models, particularly focusing on generative, decoder-type
transformers. Our library aims to democratize the access to advanced
explainability methods for data scientists and machine learning engineers who
work with language models. Built on top of Andrej Karpathy's esteemed nanoGPT,
RevLLM stands as a robust and user-friendly tool in the field of natural
language processing.


## Our model: NanoGPT 

We construct a model NanoGPT, following the example given by Andrej Karpathy [(Github)](https://github.com/karpathy/nanoGPT).

- NanoGPT comes in 4 sizes: regular, medium, large, and extra large. 
- All 4 sizes have the same architecture, but differ in weights and dimensions.
- Like all llms, NanoGPT is a variant of the transformer architecture introduced in [2017](https://arxiv.org/pdf/1706.03762.pdf).
- NanoGPT has the same architecture and weights as [gpt2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), which is trained on a dataset of 8 million web pages. The out-of-the-box model can be imported from [Hugging Face](https://huggingface.co/docs/transformers/model_doc/gpt2).

### Architecture 

All 4 sizes of NanoGPT follow the same architecture.  Differences between them lie in the sizes of individual components (see the "architecture" tab under the individual models). This section describes this architecture with dimensions excluded. We assume the reader is familiar with basics of a neural network such as linear, activation, and dropout layers, hence they are listed without comment.  Dropout layers are ignored during inference.

For what follows: 
- Let context be a string with length $c$.
- Let $n$ be the embedding dimension for choice of model size. </br>
- A tensor is a generalization of a vector.  Our model architecture employs pytorch tensors.
- Any discussion of batching or batching dimension is excluded.
- Several steps described include matrix mulitplications with learnable weights. We describe the functionality omitting such details. Exceptions are found under the "Informative Details" section. 

Layers will follow the format: 
- <u>Layer description</u> `(layer_name)`, for elements of the architecture
- <u>[Layer description]</u>, for elements of the work flow, but not the architecture

#### Preprocessing

- <u>Tokenization</u>: The tokenizer separates context into a $c$-length list of discrete tokens, one of 50257 choices (the model's vocabulary length).
- <u>Input_ids</u>: a $c$-length tensor is populated by each token mapped to an integer from $[0,50256]$.
- <u>Position_ids</u>: A tensor $[0,1, \ldots, c-1]$ is generated.

#### Embedding Layers

- <u>Token embedding</u> `(wte)`: Each input_id in input_ids is mapped to a tensor in $\mathbb{R}^n$.
  - Note: the path $(x' + \alpha (x - x'))$ for integrated gradients below is in this embedding space.
- <u>Position embedding</u> `(wpe)`: Each position_id in position_ids is embedded into a tensor in $\mathbb{R}^n$.
- <u>[Additive step]</u>: x = token_embedding + position_embedding
- <u>Dropout</u> ""

#### Transformer Block

The number of repetitions of the transformer block architecture is dependent on NanoGPT size.
  - <u>Layer Norm</u> `(ln_1)`
  - <u>Self-Attention</u> `(attn)`:
    - <u>Attention Initialization Layer</u> `(c_attn)` 
      - <u>[Multi-Headed Self-Attention Sequence]</u> (see below):
        - $Q_h, K_h, V_h =$ c_attn(x), for $1 \leq h \leq H$, where $H$ is the number of heads
        - $A_h = \text{softmax}\left(\frac{Q_hK_h^T}{\sqrt{n_h}}\right)$
        - x = Concat $(A_1V_1, \ldots, A_HV_H)$
    - <u>Linear</u> `(c_proj)`: output projection
    - <u>Dropout</u> `(attn_dropout)`
    - <u>Dropout</u> `(resid_dropout)`
  - <u>[Additive step]</u>: x = x + attn(x)
  - <u>Layer Norm</u> `(ln_2)`
  - <u>Feed Forward</u> `(MLP)`
    - <u>Linear</u> "`(c_fc)`"
    - <u>Activation</u> `(GELU)`
    - <u>Linear</u> `(c_proj)`
    - <u>Dropout</u> `(dropout)`      
  - <u>[Additive Step]</u>: x = x + MLP(x)

#### Layer Norm
- <u>Layer Norm</u> "`(ln_f)`"

#### Language Model Head

- <u>Language model head</u> `(lm_head)`: Token embeddings in dimension $n$ are converted to vocabulary-length tensors (see discussion below on weight tying).

### Informative Details

#### Self-Attention

Here we include a brief description of self attention as introduced in [2017](https://arxiv.org/pdf/1706.03762.pdf), but as it resides in our model.

##### Single Headed Attention

Let $H=1$.  The `(c_attn)` layer contains matrices $W^Q, W^K, W^V$, with learnable weights. Upon multiplication, x is separated into query, key, and value matrices $Q, K, V \in \mathbb{R}^{c \times n}$. The attention matrix is then defined:

$$
A = \text{softmax}\left(\frac{QK^T}{\sqrt{n}}\right) \in \mathbb{R}^{c \times c}
$$

The final step before `(c_proj)` is x = $AV$.

Notes:
- $A_{i,j}$ numerically  represents a relationship or 'attention' the model pays from token $i$ to token $j$. 
- The scaling factor $\frac{1}{\sqrt{n}}$ is a performance-improvement feature.
- Softmax normalizes the attention scores to probabilities. 
- In general instead of the term $\sqrt{n}$, transformer architectures mention $\sqrt{d_k}$ (see the 2017 paper).  It is not a requirement that the $Q, K, V$ matrices have dimensions equal to the embedding dimension.  But for our model, they will.
- In general transformers, $V$ may not share the dimensions as the other two matrices.

##### Muli-Headed Attention

Let $H \geq 1$ be arbitrary.  `(c_attn)` actually subdivides $n$ into $H$-many separate "heads."  Multi-headed attention allows the model to jointly attend to information from different representation subspaces at different positions.  For $1 \leq h \leq H$, we are given:

$$
A_h = \text{softmax}\left(\frac{Q_hK_h^T}{\sqrt{n_h}}\right)
$$

The final step before `(c_proj)` is x = Concat $(A_1V_1, \ldots, A_HV_H)$.

Notes:
- $n = \sum_{h} n_h$.
- For our model, $n_i = n_j$ for all $i,j \in \lbrace 1, \ldots, H \rbrace$ (hence $H|n$).  However, this fact is not a requirement for arbitrary transformer models.

#### Weight Tying of Embedding and Output Projection Matrices

Our model has a learned embedding matrix $M \in \mathbb{R}^{50257 \times n}$.

- <u>Embedding Matrix Lookup</u>: In the embedding `(wte)` layer, the input_ids contain $c$-many indices,  and the $c \times n$ resulting matrix is constructed from the corresponding rows of $M$.
- <u>Output Projection</u>: In the `(lm_head)` layer, the $c \times n$ existing state is projected to the vocabulary space by transposing the same matrix: $M^T \in \mathbb{R}^{n \times 50257}$.

This matrix repetition is a performance improvement feature known as [weight tying](https://arxiv.org/pdf/1608.05859v3.pdf), and is common in transformer architectures.

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


## Feature Attribution Methods

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


# Other analysis methods