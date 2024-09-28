This repository is based on and modifies codes and models of :

**What Can Transformers Learn In-Context? A Case Study of Simple Function Classes** <br>
*Shivam Garg\*, Dimitris Tsipras\*, Percy Liang, Gregory Valiant* <br>
Paper: http://arxiv.org/abs/2208.01066 <br><br>

And Hugging Face Transformers Lib

## Which Part is Modified

### 1. Modifies Model Input

The default input of model in Garg's setting is in the form of
$$
\{x_0,y_0,\dots,x_n,y_n\}
$$
where each x and y are separate tokens
$$
TF(\{x_0,y_0,\dots,x_n,y_n\}) = \{\hat{y}_0,\epsilon,\dots,\hat{y}_n,\epsilon\}
$$
Transformer model will predict the value of y on its corresponding x token

In our setting, we stack each x and y together for GPT2 Model. Therefore the input becomes
$$
\{(x_0,y_0),\dots,(x_n,y_n)\}
$$
where each corresponding x and y are stacked together as one token

When predicting the value of y at a specific position, we need to set it as 0, that is to say
$$
TF(\{(x_0,y_0),\dots,(x_q,0),\dots,(x_n,y_n)\}) = \{\epsilon,\dots,\hat{y}_q,\dots,\epsilon\}
$$
Note that the GPT2 Model we use is autoregressive, hence the succeeding tokens of the query token does not affect query's output

### 2. Modifies Model Input

The default HuggingFace GPT2 Model uses absolute positional embeddings, our model setting for stacking x y together does not specify `position_ids`, hence GPT2 Model will do positional embeddings as follows

```python
self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim) # 

position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
position_ids = position_ids.unsqueeze(0)

position_embeds = self.wpe(position_ids)
hidden_states = inputs_embeds + position_embeds
```

In our setting, we do not use positional embedding because

1. Different tokens `(xi, yi)` does not have sequential difference
2. Absolute Embeddings will undermine model's ability to do length generalization. Since the number of our training in context examples is much less than testing in context examples (Training setting is `40` as default but testing would be `2000`). Positions not seen in training could be considered as random noise

Therefore, we modified code to ignore positional embeddings, the new model is named as `GPT2ModelWithoutPositionEmbedding` in `modified_gpt2.py`

Furthermore, we want to evaluate model without layer norm. In `GPT2Model`, before `hidden_states` is passed as output, layer norm will be applied to it

``` python
self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

hidden_states = self.ln_f(hidden_states)
```

We ignore this layer norm, the new modified model is named as `GPT2ModelWithoutPositionEmbeddingAndLayerNorm`, apart from ignoring layer norm, it is the same as `GPT2ModelWithoutPositionEmbedding`

However, there is still layer norm in the model, which contains in

```python
self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
```

where `GPT2Block` has layer norm. Therefore we introduce `GPT2BlockNoLayerNorm` which removes those layer norm layers and give a new model `GPT2ModelWithoutPositionEmbeddingAndLayerNormAndAttentionNorm`

```python
self.h = nn.ModuleList([GPT2BlockNoLayerNorm(config, layer_idx=i) for i in range(config.num_hidden_layers)])
```

It is the same as `GPT2ModelWithouPositionEmbeddingAndLayerNorm` apart from `self.h` which uses `GPT2BlockNoLayerNorm` instead of `GPT2Block`

All of the codes mentioned above can be found in `modified_gpt2` you can specify which model to use in `base.yaml` :

```yaml
model:
    n_dims: 5
    n_positions: 2001
    type: stackxy
    layer_norm: # specify which model to use here, 3 options: use_norm, no_out, no_attn_out
    # use_norm: Use GPT2ModelWithoutPositionEmbedding
    # no_out: GPT2ModelWithoutPositionEmbeddingAndLayerNorm
    # no_attn_out: GPT2ModelWithoutPositionEmbeddingAndLayerNormAndAttentionNorm
```



## Getting started

You can start by cloning our repository and following the steps below.

1. Install the dependencies for our code using Conda. You may need to adjust the environment YAML file depending on your setup.

   ```
   conda env create -f environment.yml
   conda activate in-context-learning
   ```

2. To train models yourself, you might need to generate data first

   ```bash
   python src/dataset_base.py
   ```

3. After generate corresponding dataset, you need to modified this code in `train.py`

   ```python
       data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "your_dataset_name/index_to_file_dict.yaml")
   
       fp = open(data_dir)
   
       index_to_file_dict = yaml.load(fp)
   
       data_method = LinearReg({"L": your_trainigexample_length, "dx": n_dims, "dy": 1, "number_of_samples": 1, "noise_std": 0}) # This is used to transform data for training, it is not need to be the same as how you generate dataset
   
       training_dataset = DatasetBase(
           index_to_file_dict=index_to_file_dict["train"], 
           data_method=data_method,
           data_method_args_dict={"L": your_trainigexample_length}, # 40 is recommended
           load_data_into_memory=True
       )
   
       validating_dataset = DatasetBase(
           index_to_file_dict=index_to_file_dict["val"], 
           data_method=data_method,
           data_method_args_dict={"L": your_trainigexample_length},
           load_data_into_memory=True
       )
   ```

4. Then you could train model by yourself! After training, it will automatically visualize QK circuit and evaluate the model

   ```
   python src/train.py --config src/conf/linear_regression.yaml
   ```

5. [Optional], if you only want to evaluate model, use this command

   ```
   python src/get_eval.py --dir your_models_directory --runid your_model_runid
   ```

   Usually dir and runid are in this form `models_directory/runid/..(contains state.pt)`

6. [Optional], if you only want to visualize QK circuit, use this command

   ```
   python src/visualize_qk.py --dir your_models_directory --runid your_model_runid
   ```

   

# Maintainers

* Xintian Pan

