# Structure of LlamaHW

## llama.py
This file contains the Llama2 model whose backbone is the [transformer](https://arxiv.org/pdf/1706.03762.pdf). We recommend walking through Section 3 of the paper to understand each component of the transformer. 

### Attention
The multi-head attention layer of the transformer. This layer maps a query and a set of key-value pairs to an output. The output is calculated as the weighted sum of the values, where the weight of each value is computed by a function that takes the query and the corresponding key. To implement this layer, you can:
1. linearly project the queries, keys, and values with their corresponding linear layers
2. split the vectors for multi-head attention
3. follow the equation to compute the attended output of each head
4. concatenate multi-head attention outputs to recover the original shape

$$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$$

Llama2 uses a modified version of this procedure called [Grouped-Query Attention](https://arxiv.org/abs/2305.13245) where, instead of each attention head having its own "query", "key", and "vector" head, some groups of "query" heads share the same "key" and "vector" heads. To simplify your implementation, we've taken care of steps #1, 2, and 4 here; you only need to follow the equation to compute the attended output of each head.

### LlamaLayer
This corresponds to one transformer layer which has 
1. layer normalization of the input (via Root Mean Square layer normalization)
2. self-attention on the layer-normalized input
3. a residual connection (i.e., add the input to the output of the self-attention)
4. layer normalization on the output of the self-attention
5. a feed-forward network on the layer-normalized output of the self-attention
6. a residual connection from the unnormalized self-attention output added to the
    output of the feed-forward network

### Llama
This is the Llama model that takes in input ids and returns next-token predictions and contextualized representation for each word. The structure of ```Llama``` is:
1. an embedding layer that consists of token embeddings ```tok_embeddings```.
2. llama encoder layer which is a stack of ```config.num_hidden_layers``` ```LlamaLayer```
3. a projection layer for each hidden state which predicts token IDs (for next-word prediction)
4. a "generate" function which uses temperature sampling to generate long continuation strings. Note that, unlike most practical implementations of temperature sampling, you should not perform nucleus/top-k sampling in your sampling procedure.

The desired outputs are
1. ```logits```: logits (output scores) over the vocabulary, predicting the next possible token at each point
2. ```hidden_state```: the final hidden state at each token in the given document

### To be implemented
Components that require your implementations are comment with ```#todo```. The detailed instructions can be found in their corresponding code blocks
* ```llama.Attention.forward```
* ```llama.RMSNorm.norm```
* ```llama.Llama.forward```
* ```llama.Llama.generate```
* ```rope.apply_rotary_emb``` (this one may be tricky! you can use `rope_test.py` to test your implementation)
* ```optimizer.AdamW.step```
* ```classifier.LlamaEmbeddingClassifier.forward```

*ATTENTION:* you are free to re-organize the functions inside each class, but please don't change the variable names that correspond to Llama2 parameters. The change to these variable names will fail to load the pre-trained weights.

### Sanity check (Llama forward pass integration test)
We provide a sanity check function at sanity_check.py to test your Llama implementation. It will reload two embeddings we computed with our reference implementation and check whether your implementation outputs match ours.


## classifier.py
This file contains the pipeline to 
* load a pretrained model
* generate an example sentence (to verify that your implemention works)
* call the Llama2 model to encode the sentences for their contextualized representations
* feed in the encoded representations for the sentence classification task
* fine-tune the Llama2 model on the downstream tasks (e.g. sentence classification)


### LlamaSentClassifier (to be implemented)
This class is used to
* encode the sentences using Llama2 to obtain the hidden representation from the final word of the sentence.
* classify the sentence by applying dropout to the pooled-output and project it using a linear layer.

## optimizer.py  (to be implemented)
This is where `AdamW` is defined.
You will need to update the `step()` function based on [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) and [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
There are a few slight variations on AdamW, pleae note the following:
- The reference uses the "efficient" method of computing the bias correction mentioned at the end of section 2 "Algorithm" in Kigma & Ba (2014) in place of the intermediate m hat and v hat method.
- The learning rate is incorporated into the weight decay update (unlike Loshchiloc & Hutter (2017)).
- There is no learning rate schedule.

You can check your optimizer implementation using `optimizer_test.py`.

## rope.py (to be implemented)
Here, you will implement rotary positional embeddings. This may be tricky; you can refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf and Section 3 in https://arxiv.org/abs/2104.09864 for reference. To enable you to test this component modularly, we've provided a unit test at `RoPE_test.py`

## base_llama.py
This is the base class for the Llama model. You won't need to modify this file in this assignment.

## tokenizer.py
This is the tokenizer we will use. You won't need to modify this file in this assignment.

## config.py
This is where the configuration class is defined. You won't need to modify this file in this assignment.

## utils.py
This file contains utility functions for various purpose. You won't need to modify this file in this assignment.
 
## Reference
[Vaswani el at. + 2017] Attention is all you need https://arxiv.org/abs/1706.03762
[Touvron el at. + 2023] Llama 2: Open Foundation and Fine-Tuned Chat Models https://arxiv.org/abs/2307.09288
