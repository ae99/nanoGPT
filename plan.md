If I were to get GPT J pretrained weights running on a variant of this repo, where would I start?


Baseline, compare "models" as pytorch prints them










Interesting point... it *may* be better to skip GPT-J
and go straight to GPT-NeoX because it's code is cleaner
it has a smaller model sizes *and* larger ones












# Embedding
GPT has
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
X has
    (embed_in): Embedding(1024, 32)

# Blocks 
## Blocks LayerNorms
GPT:
(ln_1): LayerNorm()
(ln_2): LayerNorm()
X:
(input_layernorm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
(post_attention_layernorm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
---
slightly different?

## Blocks MLP
GPT:
(c_fc): Linear(in_features=768, out_features=3072, bias=True)
(c_proj): Linear(in_features=3072, out_features=768, bias=True)
(dropout): Dropout(p=0.0, inplace=False)
X:
(dense_h_to_4h): Linear(in_features=32, out_features=37, bias=True)
(dense_4h_to_h): Linear(in_features=37, out_features=32, bias=True)
(act): GELUActivation()

---
Essentially the same, GELU not RELU by the looks of it, and no dropout is only difference

## Blocks Attention
GPT:
    (c_attn): Linear(in_features=768, out_features=2304, bias=True)
    (c_proj): Linear(in_features=768, out_features=768, bias=True)
    (attn_dropout): Dropout(p=0.0, inplace=False)
    (resid_dropout): Dropout(p=0.0, inplace=False)

X:
    (rotary_emb): RotaryEmbedding()
    (query_key_value): Linear(in_features=32, out_features=96, bias=True)
    (dense): Linear(in_features=32, out_features=32, bias=True)
---
Once again, no dropout layers.
dense looks like c_proj (to verify)
query_key_value looks like c_attn (to verify)
+ rotary_emb obvs


# Final Layers
GPT:
(ln_f): LayerNorm()
(lm_head): Linear(in_features=768, out_features=50257, bias=False)

GPTJ
(ln_f): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
(lm_head): Linear(in_features=32, out_features=1000, bias=True)






