import sys
import time
from pathlib import Path

import mlx.core as mx

# Add parent directory to Python path
cur_path = Path(__file__).parents[1].resolve()
sys.path.append(str(cur_path))

from esm import ESM2

# Example protein sequence (Green Fluorescent Protein)
protein_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

# Load pretrained ESM-2 model and its tokenizer from local checkpoint
tokenizer, model = ESM2.from_pretrained("checkpoints/mlx-esm2_t33_650M_UR50D")

# Number of sequences to process in each forward pass
batch_size = 5

# Number of timing iterations for performance measurement
steps = 50

# Tokenize the protein sequence into integer IDs for the model
# Replicate the same sequence 'batch_size' times to create a batch
tokens = tokenizer.batch_encode([protein_sequence] * batch_size)

# Warm-up phase
for _ in range(10):
    result = model(tokens)
    mx.eval(result["logits"])  # Force computation to complete

# Measure average inference time over 'steps' iterations
tic = time.time()
for _ in range(steps):
    result = model(tokens)
    mx.eval(result["logits"])  # Synchronize and ensure computation finishes
toc = time.time()

# Compute metrics: average time per step (ms) and throughput (sequences/sec)
ms_per_step = 1000 * (toc - tic) / steps
throughput = batch_size * 1000 / ms_per_step

# Display results
print(f"Time (ms) per step: {ms_per_step:.3f}")
print(f"Throughput: {throughput:.2f} sequences/sec")
