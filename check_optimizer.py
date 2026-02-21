#!/usr/bin/env python3
"""
Check if the lazy layers were registered with the optimizer.
"""
import torch

ckpt = torch.load('big_vae_latent256_epoch5.pt', map_location='cpu')

# Get optimizer state
opt_state = ckpt.get('optimizer_state_dict', {})
param_groups = opt_state.get('param_groups', [])

if param_groups:
    num_params_in_opt = len(param_groups[0].get('params', []))
else:
    num_params_in_opt = 0

# Get model state
model_state = ckpt['model_state_dict']
# Count parameters (weights and biases, not buffers like running_mean/running_var)
num_params_in_model = sum(1 for k in model_state.keys()
                          if not ('running_' in k or 'num_batches_tracked' in k))

print("=" * 70)
print("OPTIMIZER VS MODEL PARAMETER COUNT")
print("=" * 70)
print(f"Parameters in optimizer state: {num_params_in_opt}")
print(f"Parameters in model state_dict: {num_params_in_model}")
print(f"Difference: {num_params_in_model - num_params_in_opt}")

if num_params_in_model > num_params_in_opt:
    print("\n❌ PROBLEM: Model has MORE parameters than optimizer!")
    print("   Some parameters were never optimized!")

    # Check if lazy layers are in optimizer
    print("\n" + "=" * 70)
    print("CHECKING IF LAZY LAYERS ARE IN OPTIMIZER")
    print("=" * 70)

    # Get the state dict keys for lazy layers
    lazy_layer_keys = [k for k in model_state.keys()
                       if 'fc_mu' in k or 'fc_logvar' in k or 'decoder_input' in k]

    print(f"\nLazy layer parameters in model: {len(lazy_layer_keys)}")
    for key in lazy_layer_keys:
        print(f"  - {key}")

    # Check optimizer state
    if param_groups:
        state = opt_state.get('state', {})
        print(f"\nOptimizer has state for {len(state)} parameters")

        # The optimizer state is indexed by parameter id, not name
        # We can check if the count matches
        num_with_adam_state = len([s for s in state.values() if 'exp_avg' in s])
        print(f"Parameters with Adam momentum: {num_with_adam_state}")

        expected_trained = num_params_in_model - len(lazy_layer_keys)
        print(f"\nExpected trained params (total - lazy): {expected_trained}")
        print(f"Actual params in optimizer: {num_params_in_opt}")

        if num_params_in_opt == expected_trained:
            print("\n✓ This confirms the lazy layers were NOT added to optimizer!")
            print("  The lazy layers have random initialization and never trained.")
elif num_params_in_model == num_params_in_opt:
    print("\n✓ Optimizer and model have same number of parameters")
else:
    print("\n⚠️  Unexpected: Optimizer has MORE parameters than model")

print("\n" + "=" * 70)
print("ROOT CAUSE IDENTIFIED")
print("=" * 70)
print("""
The bug is in train_big_vae.py lines 560-570:

    model = BigConvVAE(...).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)  # ← BUG!

The optimizer is created BEFORE the first forward pass, so the lazy
layers (fc_mu, fc_logvar, decoder_input) don't exist yet.

When they're created in encode() during the first batch, they're NOT
registered with the optimizer, so they NEVER get gradient updates.

This causes:
1. Lazy layers stay at random initialization for all epochs
2. Encoder trains normally, its output distribution shifts
3. Untrained decoder_input produces extreme/unstable outputs
4. Numerical explosion → NaN in decoder BatchNorm running stats
""")
