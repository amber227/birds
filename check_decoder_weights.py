#!/usr/bin/env python3
"""
Check if decoder weights are corrupted or if lazy layers exist.
"""
import torch
import sys

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else 'big_vae_latent256_epoch5.pt'

print(f"Loading: {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location='cpu')
state = ckpt['model_state_dict']

print("\n" + "=" * 70)
print("CHECKING DECODER WEIGHTS (excluding BatchNorm running stats)")
print("=" * 70)

decoder_issues = []

for name, param in state.items():
    if 'decoder' in name and 'running' not in name and 'num_batches' not in name:
        has_nan = torch.isnan(param).any().item()
        has_inf = torch.isinf(param).any().item()
        min_val = param.min().item()
        max_val = param.max().item()
        mean_val = param.mean().item()
        std_val = param.std().item()

        if has_nan:
            print(f"❌ {name}: Contains NaN")
            decoder_issues.append((name, "NaN"))
        elif has_inf:
            print(f"❌ {name}: Contains Inf")
            decoder_issues.append((name, "Inf"))
        elif abs(min_val) > 1000 or abs(max_val) > 1000:
            print(f"⚠️  {name}: VERY LARGE - min={min_val:.2f}, max={max_val:.2f}")
            decoder_issues.append((name, "too large"))
        elif abs(mean_val) < 1e-10 and abs(std_val) < 1e-10:
            print(f"⚠️  {name}: ALL ZEROS - mean={mean_val:.2e}, std={std_val:.2e}")
            decoder_issues.append((name, "all zeros"))

if not decoder_issues:
    print("✓ All decoder weights look reasonable")

print("\n" + "=" * 70)
print("CHECKING FOR LAZY LAYERS (fc_mu, fc_logvar, decoder_input)")
print("=" * 70)

lazy_layers = []
for key in state.keys():
    if 'fc_mu' in key or 'fc_logvar' in key or 'decoder_input' in key:
        lazy_layers.append(key)
        print(f"  Found: {key} - shape: {state[key].shape}")

if not lazy_layers:
    print("⚠️  WARNING: No lazy layers found in checkpoint!")
    print("   This might be the problem - the decoder can't map from latent space.")

print("\n" + "=" * 70)
print("CHECKING ENCODER VS DECODER LAYER COUNTS")
print("=" * 70)

encoder_layers = [k for k in state.keys() if 'encoder' in k and 'weight' in k]
decoder_layers = [k for k in state.keys() if 'decoder' in k and 'weight' in k]

print(f"Encoder layers with weights: {len(encoder_layers)}")
print(f"Decoder layers with weights: {len(decoder_layers)}")

print("\nEncoder layer names:")
for name in sorted(encoder_layers)[:10]:
    print(f"  {name}")
if len(encoder_layers) > 10:
    print(f"  ... and {len(encoder_layers) - 10} more")

print("\nDecoder layer names:")
for name in sorted(decoder_layers)[:10]:
    print(f"  {name}")
if len(decoder_layers) > 10:
    print(f"  ... and {len(decoder_layers) - 10} more")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if decoder_issues:
    print(f"❌ Found {len(decoder_issues)} issues with decoder weights")
    for name, issue in decoder_issues[:5]:
        print(f"   - {name}: {issue}")
else:
    print("✓ Decoder weights appear healthy")

if not lazy_layers:
    print("❌ Lazy layers (fc_mu, fc_logvar, decoder_input) are MISSING")
    print("   This will prevent the decoder from working at all!")
else:
    print(f"✓ Found {len(lazy_layers)} lazy layer parameters")
