import os
import numpy as np
import torch
from torch.backends.cuda import sdp_kernel

from dataclasses import dataclass

from model import GPT
from dataloader import DataLoaderLite
import utils.hessian_spectrum as hessian_mod

torch.set_float32_matmul_precision('high')


# === GPTConfig: 학습 때와 동일하게 맞춰두기 ===
@dataclass
class GPTConfig:
    context_length: int = 1024
    vocab_size: int = 50304   # 실제 학습에 사용한 값으로 맞춰두는 게 안전
    num_layers: int = 12
    embd_size: int = 768
    num_heads: int = 12


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Hessian spectrum from checkpoint")
    parser.add_argument("--ckpt",type=str,required=True,help="path to checkpoint .pt file (saved with {'model': state_dict, 'config': config, ...})",)
    parser.add_argument("--outdir",type=str,required=True,help="tag or folder name used by Hessian(comment) to save plots",)
    parser.add_argument("--context_length",type=int,default=1024,help="sequence length (block_size) for Hessian computation",)
    parser.add_argument("--batch_size",type=int,default=4,help="batch size used in Hessian computation",)
    parser.add_argument("--grad_accum_steps",type=int,default=1,help="gradient accumulation steps used in Hessian class")
    return parser.parse_args()


def load_model_from_ckpt(ckpt_path: str, device: str):
    print(f"[INFO] Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        config = checkpoint.get("config", None)

        if config is None:
            print("[WARN] 'config' not found in checkpoint, rebuilding GPTConfig manually...")
            config = GPTConfig()
        else:
            print("[INFO] Using GPT config from checkpoint.")
        model = GPT(config=config)
        model.load_state_dict(state_dict)
    else:
        print("[INFO] Checkpoint looks like a raw state_dict, building GPTConfig manually...")
        state_dict = checkpoint
        config = GPTConfig()
        model = GPT(config=config)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def build_train_data_from_dataloader(context_length: int):
    train_loader = DataLoaderLite(
        B=16,                # Hessian용이니까 작은 배치면 충분
        T=context_length,
        process_rank=0,
        num_processes=1,
        split="train",
    )

    if hasattr(train_loader, "tokens"):
        train_data = np.asarray(train_loader.tokens, dtype=np.int64)
    elif hasattr(train_loader, "data"):
        train_data = np.asarray(train_loader.data, dtype=np.int64)
    else:
        raise RuntimeError(
            "Error"
        )
    return train_data


def plot_hessian_from_checkpoint(model, device, train_data, args):
    from contextlib import nullcontext

    ctx = nullcontext()
    sample_layer = [
        n for n, p in model.named_parameters() if p.requires_grad and "weight" in n
    ]
    print(f"[INFO] Number of sampled layers: {len(sample_layer)}")

    comment = args.outdir

    with sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        hessian = hessian_mod.Hessian(
            model=model,
            ckpt_iteration=0,
            train_data=train_data,
            batch_size=args.batch_size,
            block_size=args.context_length,
            ctx=ctx,
            use_minibatch=True,
            gradient_accumulation_steps=args.grad_accum_steps,
            device=device,
            sample_layer=sample_layer,
            comment=comment,
        )
        print("[INFO] Computing Hessian spectrum (layer-by-layer)...")
        hessian.get_spectrum(layer_by_layer=True)

        print("[INFO] Loading / plotting Hessian curves...")
        hessian.load_curve(layer_by_layer=True)

    print("[INFO] Hessian spectrum and plots completed.")


def main():
    args = get_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[INFO] Using device: {device}")

    
    model = load_model_from_ckpt(args.ckpt, device)
    train_data = build_train_data_from_dataloader(args.context_length)
    plot_hessian_from_checkpoint(model, device, train_data, args)


if __name__ == "__main__":
    main()
