# ============================
# Inject FAST tokens into Qwen VL
# ============================

import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# QWEN_PATH = r"/mnt/project_ai4edu/lxp/vlm_pretrain/saves/qwen3vl-4b/lora/EgoPlan_CoT_SAT_FV_full_egodex_ego4d_buildai_sft_multi_512_8192_1epoch_5e4_b16_DDP_shuffle_nothink_rank8_confirm/scaling_law/checkpoint-13297-merge"
QWEN_PATH = r"/mnt/project_ai4edu/share/models/Qwen3-VL-4B-Instruct/"
OUTPUT_PATH = r"/mnt/project_eai/lsj/starVLA/playground/Pretrained_models"
OUTPUT_NAME = "Qwen3-VL-4B-Instruct-with-Action-Query-Better64"


def _get_initializer_range(model) -> float:
    """
    Try best effort to get initializer std from config.
    """
    cfg = getattr(model.config, "text_config", model.config)
    std = getattr(cfg, "initializer_range", None)
    if std is None:
        std = getattr(model.config, "initializer_range", None)
    return float(std) if std is not None else 0.02


@torch.no_grad()
def _init_added_token_embeddings(
    model,
    old_vocab_size: int,
    num_added: int,
    init_mode: str = "mean+noise",
    noise_std: float | None = None,
    seed: int = 42,
):
    """
    Initialize newly-added token embeddings after resize_token_embeddings.

    Args:
        model: HF model
        old_vocab_size: vocab size BEFORE adding tokens
        num_added: number of tokens newly added
        init_mode:
            - "mean+noise": new = mean(old_emb) + N(0, noise_std)
            - "sample+noise": new = old_emb[random_row] + N(0, noise_std)
            - "hf_default": do nothing (keep HF default init)
        noise_std: if None, use model initializer_range (usually ~0.02)
        seed: for reproducibility
    """
    if num_added <= 0:
        return

    in_emb = model.get_input_embeddings()
    if in_emb is None or not hasattr(in_emb, "weight"):
        raise RuntimeError("Model has no input embeddings or unexpected embedding module.")

    w_in = in_emb.weight  # [new_vocab, hidden]
    device = w_in.device
    dtype = w_in.dtype
    hidden = w_in.shape[1]

    if noise_std is None:
        noise_std = _get_initializer_range(model)

    if init_mode == "hf_default":
        print("[Init] Using HF default init for newly added tokens.")
        return

    # Make randomness reproducible
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    # Stats computed only from old rows (before tokens are added)
    old_rows = w_in[:old_vocab_size]  # [old_vocab, hidden]

    if init_mode == "mean+noise":
        # mean vector in float32 (small tensor)
        mean = old_rows.mean(dim=0, keepdim=True).float()  # [1, hidden]
        noise = torch.randn((num_added, hidden), device=device, dtype=torch.float32, generator=g) * float(noise_std)
        new_rows = (mean + noise).to(dtype)

    elif init_mode == "sample+noise":
        # sample existing embeddings to keep diversity
        idx = torch.randint(low=0, high=old_vocab_size, size=(num_added,), device=device, generator=g)
        base = old_rows.index_select(0, idx).float()  # [num_added, hidden]
        noise = torch.randn((num_added, hidden), device=device, dtype=torch.float32, generator=g) * float(noise_std)
        new_rows = (base + noise).to(dtype)

    else:
        raise ValueError(f"Unknown init_mode: {init_mode}. Use 'mean+noise', 'sample+noise', or 'hf_default'.")

    start = old_vocab_size
    end = old_vocab_size + num_added
    print(f"[Init] Writing new embeddings to rows [{start}, {end}) with mode={init_mode}, noise_std={noise_std}")
    w_in.data[start:end].copy_(new_rows)

    # If output embeddings exist and are NOT tied, also update them
    out_emb = model.get_output_embeddings()
    if out_emb is not None and hasattr(out_emb, "weight") and out_emb.weight is not None:
        w_out = out_emb.weight
        try:
            tied = (w_out.data_ptr() == w_in.data_ptr())
        except Exception:
            tied = False

        if (w_out.shape[0] == w_in.shape[0]) and (not tied):
            w_out.data[start:end].copy_(new_rows.to(w_out.dtype))
            print("[Init] Also initialized output embeddings (not tied).")
        else:
            print("[Init] Output embeddings tied or shape mismatch; skip explicit init.")


def add_fast_token_to_qwen(
    token_len: int = 32,
    init_mode: str = "mean+noise",
    noise_std: float | None = None,
    seed: int = 42,
):
    """
    Inject <|action_i|> tokens into Qwen tokenizer, resize model, initialize embeddings, save.
    """
    action_tokens = [f"<|action_{i}|>" for i in range(token_len)]
    print("Add action tokens:", action_tokens)

    # Load Qwen VL processor/tokenizer
    qwen_processor = AutoProcessor.from_pretrained(QWEN_PATH, trust_remote_code=True)
    qwen_tokenizer = qwen_processor.tokenizer

    # Load model
    qwen_model = AutoModelForImageTextToText.from_pretrained(
        QWEN_PATH,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",  # if this causes trouble for editing, switch to "cpu"
    )

    old_vocab_size = len(qwen_tokenizer)
    print("Original Qwen vocab size:", old_vocab_size)

    # Add action tokens
    num_added = qwen_tokenizer.add_tokens(action_tokens, special_tokens=False)
    print("Added action tokens to Qwen:", num_added)
    print("New Qwen vocab size:", len(qwen_tokenizer))

    # Resize embeddings (required)
    if num_added > 0:
        qwen_model.resize_token_embeddings(len(qwen_tokenizer))

        # Initialize the newly-added rows in embedding matrix
        _init_added_token_embeddings(
            model=qwen_model,
            old_vocab_size=old_vocab_size,
            num_added=num_added,
            init_mode=init_mode,
            noise_std=noise_std,
            seed=seed,
        )

    # Save
    save_dir = os.path.join(OUTPUT_PATH, OUTPUT_NAME)
    os.makedirs(save_dir, exist_ok=True)
    qwen_processor.save_pretrained(save_dir)
    qwen_model.save_pretrained(save_dir)
    print("Qwen + Action Query vocab saved to:", save_dir)


def unit_test_qwen_encode_action_span(qwen_tokenizer, token_len=32):
    """
    Test Qwen tokenizer can encode the action-span string into expected tokens.
    """
    # You can also try " ".join(...) if you want strict boundary.
    text = "".join([f"<|action_{i}|>" for i in range(token_len)])

    ids = qwen_tokenizer.encode(text, add_special_tokens=False)
    toks = qwen_tokenizer.convert_ids_to_tokens(ids)
    print("[Qwen encode test] text:", repr(text))
    print("[Qwen encode test] len(ids):", len(ids))
    print("[Qwen encode test] ids:", ids)
    print("[Qwen encode test] toks:", toks)

    assert len(ids) == token_len, f"Expected {token_len} tokens, got {len(ids)}"
    for i in range(token_len):
        expected_tok = f"<|action_{i}|>"
        assert toks[i] == expected_tok, f"Token {i} mismatch: expected {expected_tok}, got {toks[i]}"


if __name__ == "__main__":
    token_len = 64

    # Recommended defaults:
    # - init_mode="mean+noise": stable
    # - noise_std=None uses initializer_range (~0.02)
    # If you want stronger diversity for query tokens, try init_mode="sample+noise"
    add_fast_token_to_qwen(
        token_len=token_len,
        init_mode="mean+noise",   # or "sample+noise"
        noise_std=None,           # or explicitly set e.g. 0.02 / 0.01
        seed=42,
    )

    # Reload tokenizer for unit test
    save_dir = os.path.join(OUTPUT_PATH, OUTPUT_NAME)
    qwen_processor = AutoProcessor.from_pretrained(save_dir, trust_remote_code=True)
    qwen_tokenizer = qwen_processor.tokenizer

    unit_test_qwen_encode_action_span(qwen_tokenizer, token_len=token_len)
