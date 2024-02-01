import torch
from llama import load_pretrained

seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

sanity_data = torch.load("./sanity_check.data")
# text_batch = ["hello world", "hello neural network for NLP"]
# tokenizer here
sent_ids = torch.tensor([[101, 7592, 2088, 102, 0, 0, 0, 0],
                         [101, 7592, 15756, 2897, 2005, 17953, 2361, 102]])

# load our model
llama = load_pretrained("stories42M.pt")
with torch.no_grad():
    logits, hidden_states = llama(sent_ids)
    assert torch.allclose(logits, sanity_data["logits"], atol=1e-5, rtol=1e-3)
    assert torch.allclose(hidden_states, sanity_data["hidden_states"], atol=1e-5, rtol=1e-3)
    print("Your Llama implementation is correct!")