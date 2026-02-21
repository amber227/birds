#!/usr/bin/env python3
"""
Check a VAE checkpoint for NaN or corrupted values.
"""
import argparse
from pathlib import Path
import torch


def check_checkpoint(checkpoint_path: Path):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(str(checkpoint_path), map_location='cpu')

    if 'model_state_dict' not in ckpt:
        print("Error: Checkpoint does not contain 'model_state_dict'")
        return

    state = ckpt['model_state_dict']

    print("\n" + "=" * 70)
    print("CHECKING FOR NaN VALUES")
    print("=" * 70)

    nan_found = False
    for name, param in state.items():
        if torch.isnan(param).any():
            num_nan = torch.isnan(param).sum().item()
            total = param.numel()
            print(f"❌ {name}:")
            print(f"   Contains {num_nan}/{total} NaN values ({100*num_nan/total:.2f}%)")
            nan_found = True

    if not nan_found:
        print("✓ No NaN values found in checkpoint")

    print("\n" + "=" * 70)
    print("CHECKING BATCHNORM RUNNING STATISTICS")
    print("=" * 70)

    bn_issues = False
    for name, param in state.items():
        if 'running_var' in name:
            min_val = param.min().item()
            max_val = param.max().item()
            mean_val = param.mean().item()

            has_issue = False
            issues = []

            if min_val <= 0:
                has_issue = True
                issues.append(f"non-positive values (min={min_val:.6f})")

            if torch.isnan(param).any():
                has_issue = True
                issues.append("contains NaN")

            if torch.isinf(param).any():
                has_issue = True
                issues.append("contains Inf")

            if has_issue:
                bn_issues = True
                print(f"⚠️  {name}:")
                print(f"   min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}")
                print(f"   Issues: {', '.join(issues)}")
            else:
                print(f"✓ {name}:")
                print(f"   min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}")

    # Check running_mean too
    print("\n" + "=" * 70)
    print("BATCHNORM RUNNING MEANS")
    print("=" * 70)

    for name, param in state.items():
        if 'running_mean' in name:
            min_val = param.min().item()
            max_val = param.max().item()
            mean_val = param.mean().item()

            has_issue = False
            issues = []

            if torch.isnan(param).any():
                has_issue = True
                issues.append("contains NaN")

            if torch.isinf(param).any():
                has_issue = True
                issues.append("contains Inf")

            if has_issue:
                print(f"⚠️  {name}:")
                print(f"   min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}")
                print(f"   Issues: {', '.join(issues)}")
            else:
                print(f"✓ {name}:")
                print(f"   min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}")

    print("\n" + "=" * 70)
    print("CHECKING OTHER PARAMETERS")
    print("=" * 70)

    other_issues = False
    for name, param in state.items():
        if 'running_' not in name:  # Skip BatchNorm buffers we already checked
            if torch.isnan(param).any():
                other_issues = True
                num_nan = torch.isnan(param).sum().item()
                total = param.numel()
                print(f"❌ {name}:")
                print(f"   Contains {num_nan}/{total} NaN values")
            elif torch.isinf(param).any():
                other_issues = True
                num_inf = torch.isinf(param).sum().item()
                total = param.numel()
                print(f"⚠️  {name}:")
                print(f"   Contains {num_inf}/{total} Inf values")

    if not other_issues:
        print("✓ No NaN or Inf values in weights/biases")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Checkpoint metadata:")
    print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"  Latent dim: {ckpt.get('latent_dim', 'N/A')}")
    print(f"  Beta: {ckpt.get('beta', 'N/A')}")

    if nan_found or bn_issues or other_issues:
        print("\n❌ CHECKPOINT HAS ISSUES - This will cause NaN in model outputs!")
    else:
        print("\n✓ Checkpoint appears healthy")
        print("  If you're still getting NaN outputs, the problem may be elsewhere.")


def main():
    parser = argparse.ArgumentParser(description="Check VAE checkpoint for corruption")
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint file")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return

    check_checkpoint(checkpoint_path)


if __name__ == "__main__":
    main()
