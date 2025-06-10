import Foundation
import MLX
import MLXLMCommon
import MLXLLM

func SimpleMLXLMWithKVCache() async throws {
    // Load the model and tokenizer directly from HF
    let modelId = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
    let modelFactory = LLMModelFactory.shared
    let configuration = ModelConfiguration(id: modelId)
    let model = try await modelFactory.loadContainer(configuration: configuration)

    try await model.perform({context in
        // Prepare the prompt for the model
        let prompt = "Write a quicksort in Swift"
        let input = try await context.processor.prepare(input: UserInput(prompt: prompt))

        // Create the key-value cache
        let generateParameters = GenerateParameters()
        let cache = context.model.newCache(parameters: generateParameters)

        // Low level token iterator
        let tokenIter = try TokenIterator(input: input,
                                          model: context.model,
                                          cache: cache,
                                          parameters: generateParameters)
        let tokenStream = generate(input: input, context: context, iterator: tokenIter)
        for await part in tokenStream {
            print(part.chunk ?? "", terminator: "")
        }
        
        print("\n=============================================================================\n")
        
        // Prompt the model again with a follow up questions:
        let newPrompt = "What is it's time complexity?"
        let newInput = try await context.processor.prepare(input: .init(prompt: newPrompt))
        let newTokenIter = try TokenIterator(input: newInput,
                                     model: context.model,
                                     cache: cache,
                                     parameters: generateParameters)

    
        let newTokenStream = generate(input: newInput, context: context, iterator: newTokenIter)
        for await part in newTokenStream {
            print(part.chunk ?? "", terminator: "")
        }
    })
}
