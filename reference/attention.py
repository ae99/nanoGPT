from modeling_gpt_neox import GPTNeoXForCausalLM
model = GPTNeoXForCausalLM.from_pretrained("trl-internal-testing/tiny-random-GPTNeoXForCausalLM")
from modeling_gpt_neox import GPTNeoXAttention
attention = GPTNeoXAttention(config=model.config)

new_state_dict = {}
for k, v in model.state_dict().items():
    if k.startswith("gpt_neox.layers.1.attention."):
        new_state_dict[k.replace("gpt_neox.layers.1.attention.", "")] = v

attention.load_state_dict(new_state_dict)

import torch
inputState = torch.randn(5, 7, model.config.hidden_size)

attention(inputState)
print('success')