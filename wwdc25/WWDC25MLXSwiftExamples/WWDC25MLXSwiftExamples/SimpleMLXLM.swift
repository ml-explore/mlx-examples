import Foundation
import MLX
import MLXLMCommon
import MLXLLM

func SimpleMLXLM() async throws {
    // Load the model and tokenizer directly from HF
    let modelId = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
            let modelFactory = LLMModelFactory.shared
            let configuration = ModelConfiguration(id: modelId)
            let model = try await modelFactory.loadContainer(configuration: configuration)
            
            try await model.perform({context in
                // Prepare the prompt for the model
                let prompt = "Write a quicksort in Swift"
                let input = try await context.processor.prepare(input: UserInput(prompt: prompt))
                
                // Generate the text
                let params = GenerateParameters(temperature: 0.0)
                let tokenStream = try generate(input: input, parameters: params, context: context)
                for await part in tokenStream {
                    print(part.chunk ?? "", terminator: "")
                }
            })
}
