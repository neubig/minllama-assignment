from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device
    # todo


    # 1. 逆周波数 (inv_freq) の計算
    # 0, 2, 4, ..., head_dim - 2 のインデックス j を使用
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float().to(device) / head_dim))
    
    # 2. 位置インデックス (t) の計算
    t = torch.arange(seqlen, device=device, dtype=inv_freq.dtype)
    
    # 3. 角度 (freqs) の計算: t^T @ inv_freq
    # freqs.shape: (seqlen, head_dim / 2)
    freqs = torch.outer(t, inv_freq) 
    
    # 4. freqs_cis (コサインとサインの結合) の計算
    # freqs_cis.shape: (seqlen, head_dim / 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    # 5. 形状の調整: (1, seqlen, 1, head_dim / 2) にリシェイプしてブロードキャストに備える
    # 呼び出し元のquery/keyの形状は (batch_size, seqlen, n_heads, head_dim)
    # reshape_for_broadcastは (seqlen, head_dim) を期待するため、freps_cisを結合する
    
    # freqs_cisの形状を (seqlen, head_dim) に対応させるために、実部と虚部を結合して (seqlen, head_dim) の形状にする必要があります
    
    # freqs_cisは (seqlen, head_dim/2) の複素数テンソル。
    # ここで (seqlen, head_dim/2) の形状のcosとsinを別途用意するほうが簡単です。
    cos = torch.cos(freqs).to(query.dtype)
    sin = torch.sin(freqs).to(query.dtype)


    x_half_dim = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)[0]

    # 2. reshape_for_broadcastに (seqlen, head_dim/2) のcosと (..., head_dim/2) のquery_realに対応する疑似テンソルを渡す
    cos_emb = reshape_for_broadcast(cos, x_half_dim) # 形状: (1, seqlen, 1, head_dim / 2)
    sin_emb = reshape_for_broadcast(sin, x_half_dim) # 形状: (1, seqlen, 1, head_dim / 2)
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.

    # reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 22 (linked above).

    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.
    # query_real, query_imag は既に (..., head_dim/2) の形状

    # 実部 (query_real) に cos を掛け、虚部 (query_imag) に sin を引く
    query_out_real = query_real * cos_emb - query_imag * sin_emb
    # 実部 (query_real) に sin を掛け、虚部 (query_imag) に cos を足す
    query_out_imag = query_real * sin_emb + query_imag * cos_emb

    # key も同様
    key_out_real = key_real * cos_emb - key_imag * sin_emb
    key_out_imag = key_real * sin_emb + key_imag * cos_emb

    # 3. テンソルの再結合と形状の復元
    # 形状を (..., head_dim/2, 2) に結合し、(..., head_dim) にリシェイプ

    # Qの再結合
    query_out = torch.stack((query_out_real, query_out_imag), dim=-1)
    query_out = query_out.flatten(start_dim=-2)
    query_out = query_out.type_as(query) # 元のデータ型に戻す

    # Kの再結合
    key_out = torch.stack((key_out_real, key_out_imag), dim=-1)
    key_out = key_out.flatten(start_dim=-2)
    key_out = key_out.type_as(key) # 元のデータ型に戻す
    


    #raise NotImplementedError

    #query_out = None
    #key_out = None
    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out