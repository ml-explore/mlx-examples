import time

import torch
from transformers import AutoTokenizer, EsmForMaskedLM

# Example protein sequence (Green Fluorescent Protein)
protein_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

# Hugging Face model identifier for ESM-2 (33 layers, 650M params, UR50D training set)
model_name = "facebook/esm2_t33_650M_UR50D"

# Load tokenizer and model; move model to Apple Metal Performance Shaders (MPS) device
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = EsmForMaskedLM.from_pretrained(model_name).to("mps")

# Number of sequences per forward pass
batch_size = 5

# Number of timing iterations
steps = 50

# Tokenize input sequence and replicate for the batch
# Replicate the same sequence 'batch_size' times to create a batch
inputs = tokenizer(
    [protein_sequence] * batch_size,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=1024,
)
input_ids = inputs["input_ids"].to("mps")
attention_mask = inputs["attention_mask"].to("mps")

# Warm-up phase
for _ in range(10):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    torch.mps.synchronize()  # Ensure all queued ops on MPS are complete before next step

# Timed inference loop
tic = time.time()
for _ in range(steps):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    torch.mps.synchronize()  # Wait for computation to finish before timing next iteration
toc = time.time()

# Compute performance metrics
ms_per_step = 1000 * (toc - tic) / steps
throughput = batch_size * 1000 / ms_per_step

# Report results
print(f"Time (ms) per step: {ms_per_step:.3f}")
print(f"Throughput: {throughput:.2f} sequences/sec")
