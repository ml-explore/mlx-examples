# MLX 示例项目

本代码库包含一系列使用 [MLX 框架](https://github.com/ml-explore/mlx) 的独立示例。

[MNIST手写数字识别](mnist) 示例是学习MLX使用的理想起点。以下列举更多实用示例。如需功能更完善的MLX版LLM Python工具包，请查看 [MLX LM](https://github.com/ml-explore/mlx-lm)。

### 文本模型

- [Transformer语言模型](transformer_lm) 训练
- [LLMs目录](llms) 下提供[LLaMA](llms/llama)、[Mistral](llms/mistral)等大规模文本生成的极简实现
- [Mixtral 8x7B](llms/mixtral) 混合专家(MoE)语言模型
- [LoRA/QLoRA微调](lora) 参数高效微调方案
- [T5](t5) 文本到文本的多任务Transformer
- [BERT](bert) 双向语言理解模型

### 图像模型

- 图像生成
  - [FLUX扩散模型](flux)
  - [Stable Diffusion/SDXL](stable_diffusion)
- [CIFAR-10数据集上的ResNet](cifar) 图像分类
- [MNIST上的卷积变分自编码器(CVAE)](cvae)

### 音频模型

- [OpenAI Whisper](whisper) 语音识别
- [Meta EnCodec](encodec) 音频压缩与生成
- [Meta MusicGen](musicgen) 音乐生成

### 多模态模型

- [CLIP](clip) 图文联合嵌入模型
- [LLaVA](llava) 图文多模态输入文本生成
- [Segment Anything(SAM)](segment_anything) 图像分割

### 其他模型

- [图卷积网络(GCN)](gcn) 图结构数据半监督学习
- [Real NVP标准化流](normalizing_flow) 密度估计与采样

### Hugging Face生态

您可以直接使用或下载[Hugging Face社区MLX专区](https://huggingface.co/mlx-community)的转换后模型权重。欢迎加入社区并[贡献新模型](https://github.com/ml-explore/mlx-examples/issues/155)。

## 贡献指南

我们衷心感谢[所有贡献者](ACKNOWLEDGMENTS.md#Individual-Contributors)。如果您参与MLX示例项目并希望被列入致谢名单，请在PR中补充您的姓名。

## 引用MLX示例

MLX软件套件由Awni Hannun、Jagrit Digani、Angelos Katharopoulos和Ronan Collobert四人平等贡献开发。如果您的研究受益于MLX示例项目并需要引用，请使用以下BibTex条目：

```
@software{mlx2023,
  author = {Awni Hannun and Jagrit Digani and Angelos Katharopoulos and Ronan Collobert},
  title = {{MLX}: 苹果芯片上高效灵活的机器学习框架},
  url = {https://github.com/ml-explore},
  version = {0.0},
  year = {2023},
}
```