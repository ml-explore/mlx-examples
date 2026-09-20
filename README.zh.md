<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

# MLX 示例库 (MLX Examples)

本代码库包含使用 [MLX 机器学习框架](https://github.com/ml-explore/mlx) 构建的丰富独立实战示例。

[MNIST](mnist) 示例是快速学习与上手 MLX 的极佳起点。下面列出了一系列更具实用价值的进阶示例。若需在 MLX 上使用功能更完备的大语言模型专用 Python 算法库，请访问 [MLX LM](https://github.com/ml-explore/mlx-lm)。

### 文本模型 (Text Models)

- [Transformer 语言模型](transformer_lm) 训练。
- 大规模文本生成极简示例：[LLaMA](llms/llama)、[Mistral](llms/mistral)，以及 [LLMs](llms) 目录下的更多大语言模型。
- 基于混合专家架构（MoE）的语言模型：[Mixtral 8x7B](llms/mixtral)。
- 参数高效微调：[LoRA 与 QLoRA](lora)。
- 文本到文本多任务 Transformer 模型：[T5](t5)。
- 双向语言理解模型：[BERT](bert)。

### 图像模型 (Image Models)

- 图像生成：
  - [FLUX](flux)
  - [Stable Diffusion 与 SDXL](stable_diffusion)
- 图像分类：[基于 CIFAR-10 的 ResNet](cifar)。
- 卷积变分自编码器：[基于 MNIST 的 CVAE](cvae)。

### 视频模型 (Video Models)

- 文本生成视频 (T2V) 与图像生成视频 (I2V)：[Wan2.1](video/wan2.1)。

### 音频模型 (Audio Models)

- 语音识别：[OpenAI Whisper](whisper)。
- 音频压缩与生成：[Meta EnCodec](encodec)。
- 音乐生成：[Meta MusicGen](musicgen)。

### 多模态模型 (Multimodal Models)

- 文本与图像联合表征嵌入：[CLIP](clip)。
- 结合图像与文本输入生成文本的多模态大模型：[LLaVA](llava)。
- 图像分割：[Segment Anything (SAM)](segment_anything)。

### 其他模型 (Other Models)

- 图结构数据上的半监督学习：[图卷积网络 (GCN)](gcn)。
- 用于密度估计与采样的 Real NVP [规范化流模型 (Normalizing Flow)](normalizing_flow)。

### Hugging Face 生态

您可以直接使用或下载来自 Hugging Face 上 [MLX Community](https://huggingface.co/mlx-community) 社区组织转换并发布的模型权重。我们非常鼓励您加入社区并[贡献新模型](https://github.com/ml-explore/mlx-examples/issues/155)。

## 参与贡献 (Contributing)

我们由衷感谢[所有的贡献者](ACKNOWLEDGMENTS.md#Individual-Contributors)。如果您向 MLX Examples 贡献了代码或内容并希望列入致谢名录，请在您的 Pull Request 中将您的名字添加至列表中。

## 引用 MLX Examples

MLX 软件套件最初由 Awni Hannun、Jagrit Digani、Angelos Katharopoulos 与 Ronan Collobert 共同主导开发（贡献均等）。如果您在研究中发现 MLX Examples 对您有所帮助并希望进行引用，请使用以下 BibTeX 条目：

```bibtex
@software{mlx2023,
  author = {Awni Hannun and Jagrit Digani and Angelos Katharopoulos and Ronan Collobert},
  title = {{MLX}: Efficient and flexible machine learning on Apple silicon},
  url = {https://github.com/ml-explore},
  version = {0.0},
  year = {2023},
}
```

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年09月20日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
