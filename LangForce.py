# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025].
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from typing import List, Optional, Tuple
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from starVLA.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

# ===== Qwen special tokens (you confirmed) =====
VISION_START_TOKEN_INDEX = 151652  # <|vision_start|>
VISION_END_TOKEN_INDEX   = 151654  # <|vision_end|>
IMAGE_TOKEN_INDEX        = 151655  # <|image_pad|>
VIDEO_TOKEN_INDEX        = 151656  # <|video_pad|>
IM_START_TOKEN_INDEX     = 151644 # <|im_start|>
IM_END_TOKEN_INDEX       = 151645 # <|im_end|>

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("LangForce")
class LangForce(baseframework):
    """
    from paper: LangForce: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries (arXiv:2601.15197)

    Bayesian VLA Training Forward Pass:

    1. Priori Branch (V + action_query + L): learn dataset bias p(a|v)
       - Extract A hidden -> DiT -> prior_loss

    2. Posteriori Branch (V + L + action_query): learn true policy pi(a|v,l)
       - Extract A hidden -> DiT -> main_loss

    3. LLR term (named kl_loss for compatibility):
       - maximize log p(L | V, A_prior) - sg(log p(L | V))
       - Here log p(L|V) is obtained from (V + L + A) where L is before A in causal LM.
       - Crucial: we compute language span by boundaries:
           * last <|vision_end|> position
           * action block start position
           * user <|im_end|> position
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # align dims --> should go into config ideally
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        self.num_latent_action_query = self.config.framework.qwenvl.get("num_latent_action_query", 32)
        self.latent_action_query = "".join([f"<|action_{i}|>" for i in range(self.num_latent_action_query)])
        self.action_token_ids = None  # cached {'first','last'}

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # Bayesian VLA hyperparameters
        self.kl_weight = self.config.framework.get("kl_weight", 0.1)            # LLR loss weight (maximize)
        self.prior_loss_weight = self.config.framework.get("prior_loss_weight", 0.3)

        # cache some special token ids from tokenizer lazily
        self._im_end_id = None

    # ---------------------------------------------------------------------
    # Token id helpers
    # ---------------------------------------------------------------------
    def _ensure_action_token_ids(self, tokenizer):
        if self.action_token_ids is None:
            self.action_token_ids = {
                "first": tokenizer.convert_tokens_to_ids("<|action_0|>"),
                "last": tokenizer.convert_tokens_to_ids(f"<|action_{self.num_latent_action_query-1}|>"),
            }

    def _ensure_im_end_id(self, tokenizer):
        if self._im_end_id is None:
            self._im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def _find_last_pos(self, seq_1d: torch.Tensor, token_id: int) -> int:
        """
        Return last position of token_id in seq_1d, else -1.
        """
        idx = (seq_1d == int(token_id)).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return -1
        return int(idx[-1].item())

    def _find_first_pos_after(self, seq_1d: torch.Tensor, token_id: int, start: int) -> int:
        """
        Return first position >= start where seq_1d[pos]==token_id, else -1.
        """
        if start < 0:
            start = 0
        sub = seq_1d[start:]
        idx = (sub == int(token_id)).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return -1
        return int(start + idx[0].item())

    # ---------------------------------------------------------------------
    # Action block helpers
    # ---------------------------------------------------------------------
    def _get_action_block_start(self, input_ids_1d: torch.Tensor, tokenizer) -> int:
        """
        Return first position of <|action_0|> and validate contiguous K tokens ending with <|action_{K-1}|>.
        Return -1 if not found/invalid.
        """
        self._ensure_action_token_ids(tokenizer)
        first_id = self.action_token_ids["first"]
        last_id = self.action_token_ids["last"]

        pos = (input_ids_1d == int(first_id)).nonzero(as_tuple=True)[0]
        if pos.numel() == 0:
            return -1

        start = int(pos[0].item())
        end = start + self.num_latent_action_query
        if end > input_ids_1d.shape[0]:
            return -1
        if int(input_ids_1d[end - 1].item()) != int(last_id):
            return -1
        return start

    def _extract_action_query_hidden_states(
        self,
        hidden_states: torch.Tensor,   # [B, S, H]
        input_ids: torch.Tensor,       # [B, S]
        tokenizer,
        return_starts: bool = False,
    ):
        """
        Extract latent action query hidden states as [B, K, H].
        Optionally also return action_block_start positions as [B].
        """
        self._ensure_action_token_ids(tokenizer)

        B = hidden_states.shape[0]
        out = []
        starts = []
        for b in range(B):
            start = self._get_action_block_start(input_ids[b], tokenizer)
            assert start != -1, "No valid contiguous action token block found in the sequence."
            end = start + self.num_latent_action_query
            out.append(hidden_states[b, start:end, :])
            starts.append(start)

        out = torch.stack(out, dim=0)  # [B, K, H]
        if return_starts:
            return out, torch.tensor(starts, device=input_ids.device, dtype=torch.long)
        return out

    # ---------------------------------------------------------------------
    # SHIFT-correct mean logp over a span (start inclusive, end exclusive)
    # ---------------------------------------------------------------------
    def _mean_logp_span(
        self,
        logits_1d: torch.Tensor,      # [S, V]
        input_ids_1d: torch.Tensor,   # [S]
        start: int,
        end: int,
        ignore_ids: Optional[set] = None,
    ) -> Optional[torch.Tensor]:
        """
        Compute mean log p(tokens in [start,end)) with next-token alignment:
          logits[pos] predicts input_ids[pos+1]
        So token at position j uses logits[j-1] (requires j>0).

        Returns mean logp (scalar tensor), or None if span empty/invalid.
        """
        if end <= start:
            return None
        S = int(input_ids_1d.shape[0])
        start = max(0, int(start))
        end = min(S, int(end))
        if end <= start:
            return None

        # token positions j in [start,end) and j>0
        j = torch.arange(start, end, device=input_ids_1d.device, dtype=torch.long)
        j = j[j > 0]
        if j.numel() == 0:
            return None

        targets = input_ids_1d[j].long()

        if ignore_ids is not None and len(ignore_ids) > 0:
            # filter out ignore token ids inside the span (rare but safe)
            keep = torch.ones_like(targets, dtype=torch.bool)
            for tid in ignore_ids:
                keep &= (targets != int(tid))
            j = j[keep]
            if j.numel() == 0:
                return None
            targets = input_ids_1d[j].long()

        pred_pos = j - 1
        pred_logits = logits_1d[pred_pos].float()  # [T, V]
        # Use CE for stability: logp = -CE
        nll = F.cross_entropy(pred_logits, targets, reduction="none")  # [T]
        return (-nll).mean()

    # ---------------------------------------------------------------------
    # LLR / "KL" computation using vision_end + action boundaries (NO matching)
    # ---------------------------------------------------------------------
    def _compute_language_llr_from_boundaries(
        self,
        priori_logits: torch.Tensor,            # [B, S, V]
        posteriori_logits: torch.Tensor,        # [B, S, V] (detached)
        priori_input_ids: torch.Tensor,         # [B, S]
        posteriori_input_ids: torch.Tensor,     # [B, S]
        priori_action_starts: torch.Tensor,     # [B]
        posteriori_action_starts: torch.Tensor, # [B]
    ) -> torch.Tensor:
        """
        LLR = mean log p(L | V, A_prior) - sg(mean log p(L | V))

        Language span definition (per sample):
          - posterior (V + L + A):
              lang_start_post = last_pos(<|vision_end|>) + 1
              lang_end_post   = action_start_post
          - prior (V + A + L):
              lang_start_prior = action_start_prior + K
              lang_end_prior   = first_pos_after(<|im_end|>, lang_start_prior)  (fallback to seq end)

        We also ignore a few special ids if they accidentally appear (pad/image/video/vision markers).
        """
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        self._ensure_im_end_id(tokenizer)

        pad_id = tokenizer.pad_token_id
        ignore_ids = set()
        if pad_id is not None:
            ignore_ids.add(int(pad_id))
        # safe ignores (should not appear in language span, but add anyway)
        ignore_ids.add(int(IMAGE_TOKEN_INDEX))
        ignore_ids.add(int(VIDEO_TOKEN_INDEX))
        ignore_ids.add(int(VISION_START_TOKEN_INDEX))
        ignore_ids.add(int(VISION_END_TOKEN_INDEX))
        ignore_ids.add(int(IM_START_TOKEN_INDEX))
        ignore_ids.add(int(IM_END_TOKEN_INDEX))

        B = int(priori_input_ids.shape[0])
        K = self.num_latent_action_query

        llr_vals = []

        for b in range(B):
            ids_prior = priori_input_ids[b]
            ids_post = posteriori_input_ids[b]

            a_start_prior = int(priori_action_starts[b].item())
            a_start_post = int(posteriori_action_starts[b].item())

            # ---- posterior language span: (vision_end+1) : action_start ----
            v_end_post = self._find_last_pos(ids_post, VISION_END_TOKEN_INDEX)
            if v_end_post == -1:
                continue
            lang_start_post = v_end_post + 1
            lang_end_post = a_start_post

            # ---- prior language span: (action_end) : user_im_end ----
            lang_start_prior = a_start_prior + K
            if lang_start_prior >= ids_prior.shape[0]:
                continue

            im_end_pos = self._find_first_pos_after(ids_prior, self._im_end_id, lang_start_prior)
            lang_end_prior = im_end_pos if im_end_pos != -1 else int(ids_prior.shape[0])

            # basic sanity
            if lang_end_post <= lang_start_post:
                continue
            if lang_end_prior <= lang_start_prior:
                continue

            lp_prior = self._mean_logp_span(
                logits_1d=priori_logits[b],
                input_ids_1d=ids_prior,
                start=lang_start_prior,
                end=lang_end_prior,
                ignore_ids=ignore_ids,
            )
            lp_post = self._mean_logp_span(
                logits_1d=posteriori_logits[b],
                input_ids_1d=ids_post,
                start=lang_start_post,
                end=lang_end_post,
                ignore_ids=ignore_ids,
            )
            if lp_prior is None or lp_post is None:
                continue

            llr_vals.append(lp_prior - lp_post)

        if len(llr_vals) == 0:
            return torch.tensor(0.0, device=priori_logits.device, dtype=torch.float32)
        return torch.stack(llr_vals).mean()

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        # print("begin Qwen_GR00T_with_Bayesian forward")
        batch_images = [example["image"] for example in examples]  # [B, [PIL...]]
        instructions_priori = [self.latent_action_query + example["lang"] for example in examples]       # A + L
        instructions_posteriori = [example["lang"] + self.latent_action_query for example in examples]  # L + A

        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        # ===== Step 1: Priori Branch (V + A + L) =====
        qwen_inputs_priori = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions_priori
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs_priori = self.qwen_vl_interface(
                **qwen_inputs_priori,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            priori_last_hidden = qwenvl_outputs_priori.hidden_states[-1]  # [B, S, H]
            priori_action_hidden, priori_action_starts = self._extract_action_query_hidden_states(
                priori_last_hidden,
                qwen_inputs_priori["input_ids"],
                self.qwen_vl_interface.processor.tokenizer,
                return_starts=True
            )  # [B, K, H], [B]
            priori_logits = qwenvl_outputs_priori.logits  # [B, S, V]

        # ===== Step 2: Posteriori Branch (V + L + A) =====
        qwen_inputs_posteriori = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions_posteriori
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs_posteriori = self.qwen_vl_interface(
                **qwen_inputs_posteriori,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            posteriori_last_hidden = qwenvl_outputs_posteriori.hidden_states[-1]  # [B, S, H]
            posteriori_action_hidden, posteriori_action_starts = self._extract_action_query_hidden_states(
                posteriori_last_hidden,
                qwen_inputs_posteriori["input_ids"],
                self.qwen_vl_interface.processor.tokenizer,
                return_starts=True
            )  # [B, K, H], [B]

            # detach baseline logits: we don't want to worsen p(L|V) to inflate LLR
            posteriori_logits = qwenvl_outputs_posteriori.logits.detach()  # [B, S, V]

        # ===== Step 3: LLR loss (your "KL") computed by boundaries =====
        kl_loss = self._compute_language_llr_from_boundaries(
            priori_logits=priori_logits,
            posteriori_logits=posteriori_logits,
            priori_input_ids=qwen_inputs_priori["input_ids"],
            posteriori_input_ids=qwen_inputs_posteriori["input_ids"],
            priori_action_starts=priori_action_starts,
            posteriori_action_starts=posteriori_action_starts,
        )

        # ===== Step 4: Action head losses =====
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=priori_action_hidden.device, dtype=priori_action_hidden.dtype
            )
            actions_target = actions[:, -(self.future_action_window_size + 1):, :]  # [B, chunk_len, action_dim]

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )

            state_tensor = None
            if state is not None:
                state_tensor = torch.tensor(np.array(state), device=priori_action_hidden.device, dtype=priori_action_hidden.dtype)

            # Prior loss: condition on priori_action_hidden
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            priori_cond = priori_action_hidden.repeat(repeated_diffusion_steps, 1, 1).float()
            posteriori_cond = posteriori_action_hidden.repeat(repeated_diffusion_steps, 1, 1).float()

            state_repeated = state_tensor.repeat(repeated_diffusion_steps, 1, 1) if state_tensor is not None else None

            prior_loss = self.action_model(priori_cond, actions_target_repeated, state_repeated)
            main_loss = self.action_model(posteriori_cond, actions_target_repeated, state_repeated)

        # total = main + prior_w * prior - kl_w * llr   (maximize llr)
        total_loss = (1 - self.prior_loss_weight) * main_loss + self.prior_loss_weight * prior_loss - self.kl_weight * kl_loss

        # print("End Qwen_GR00T_with_Bayesian forward")
        return {
            "action_loss": total_loss,
            # "main_loss": main_loss.detach(),
            # "prior_loss": prior_loss.detach(),
            # "kl_loss": kl_loss.detach(),
        }

    # ---------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------
    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
        Inference uses Posteriori branch: (V + L + action_query)
        """
        if type(examples) is not list:
            examples = [examples]

        # robustly preserve PIL for each view
        batch_images = []
        for ex in examples:
            imgs = ex["image"]
            if isinstance(imgs, list):
                batch_images.append([to_pil_preserve(im) for im in imgs])
            else:
                batch_images.append([to_pil_preserve(imgs)])

        instructions_posteriori = [ex["lang"] + self.latent_action_query for ex in examples]
        # instructions_posteriori = ["Just close the drawer!" + self.latent_action_query for ex in examples]
        print(instructions_posteriori)

        state = [ex["state"] for ex in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions_posteriori
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            last_hidden = qwenvl_outputs.hidden_states[-1]
            action_hidden = self._extract_action_query_hidden_states(
                last_hidden,
                qwen_inputs["input_ids"],
                self.qwen_vl_interface.processor.tokenizer,
                return_starts=False
            )  # [B, K, H]

        state_tensor = None
        if state is not None:
            state_tensor = torch.from_numpy(np.array(state)).to(action_hidden.device, dtype=action_hidden.dtype)

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(action_hidden, state_tensor)

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    args.config_yaml = "examples/MultiRobot/train_files/starvla_cotrain_multiRobot.yaml"
    cfg = OmegaConf.load(args.config_yaml)

    model: LangForce = LangForce(cfg)
    print(model)

    # fake sample
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "Put all the toys in the child's room ... inside the toy box.",
    }
    sample2 = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "Put all the toys in the child's room ... inside the toy box.",
    }

    batch  = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    predict_output = model.predict_action(examples=[sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Pred Actions Shape: {normalized_actions.shape}")

    # optional dataloader test
    vla_dataset_cfg = cfg.datasets.vla_data
    from torch.utils.data import DataLoader
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    cfg.datasets.vla_data.include_state = "False"
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,
        collate_fn=collate_fn,
    )

    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        model(batch)

    print("Finished")
