// WWDC Session: Get started with MLX for Apple silicon

// Swift
import MLX

// Make an array
let a = MLXArray([1, 2, 3])

// Make another array
let b = MLXArray([1, 2, 3])

// Do an operation
let c = a + b

// Access information about the array
let shape = c.shape
let dtype = c.dtype

// Print results
print("a: \(a)")
print("b: \(b)")
print("c = a + b: \(c)")
print("shape: \(shape)")
print("dtype: \(dtype)")

// WWDC Session: Explore large language models on Apple silicon with MLX

/// Example 1: Simple MLXLM Swift example using Mistral-7B-Instruct-v0.3-4bit
try await SimpleMLXLM()

/// Example 2: Using KVCache and custom TokenIterator with Mistral-7B-Instruct-v0.3-4bit
try await SimpleMLXLMWithKVCache()
