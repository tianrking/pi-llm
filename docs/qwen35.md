# Qwen3.5-2B 文本推理

pillm 支持 `qwen35` 的 dense 2B 布局,不依赖 Python 推理库。下载脚本和默认服务使用经过本机验证的 GGUF:

```
ref/fetch-qwen35.sh
make
./pillm -m models/qwen3.5-2b-q4_0.gguf -p "你好" --temp 0
./pillm -m models/qwen3.5-2b-q4_0.gguf --serve 8080
```

下载固定于 bartowski 转换版本 `8de6479d2743924f9dc499e3654d4e51ea0d4b9d`,文件 1,296,764,000 字节,SHA-256 为 `91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6`。`.part` 下载完成并校验后才成为模型文件。目录中已有模型时只校验,不会重新下载。模型不提交到 Git。

## 架构

主干 24 层,每四层包含三层 Gated DeltaNet 和一层门控全注意力。GGUF 的 `block_count` 还可能包括 MTP 层;主干层数减去 `nextn_predict_layers`,不执行额外预测层。

- 全注意力:联合 Q/gate 投影、逐头 Q/K RMSNorm、前 64 维的 NeoX RoPE、GQA、sigmoid gate、输出投影。文本的三个 MRoPE 坐标相同,因此使用普通一维位置即可;不支持图像和视频位置。
- DeltaNet:QKV 投影后做因果 depthwise convolution 和 SiLU。Q/K 按 `sqrt(sum(x*x) + eps)` 归一化。每头状态按 `S = exp(g) * S`,再用 `delta = beta * (v - S*k)` 更新 `S += delta * k^T`,输出为 `S*q / sqrt(head_dim)`。随后做 RMSNorm 和 SiLU(z) 门控。
- GGUF 转换已把 A_log 写成负指数,把部分归一化权重的 `1 + weight` 合并进张量,运行时不重复这些变换。
- 新架构的单 token 和批量矩阵使用相同的分块累加顺序,避免舍入差异在后续量化中放大;状态回归检查两者一致。FFN 和残差路径复用已有运算核。提示词可批量做矩阵投影,卷积和递归更新始终按 token 顺序执行。

验证过的文件混合使用 Q4_0、Q4_1、Q8_0、Q6_K 和 F32。Q6_K 矩阵使用 256 元素一组的 Q8_K 激活,NEON 整数点积;小型 F32 投影使用浮点激活。旧 qwen2/llama 的 Q6_K 输出层保留浮点激活路径。加载时检查所需张量的类型、维度和支持的布局;其他尺寸/架构不会自动按 2B 解释。

## 对话与状态

默认使用官方非思考生成前缀:

```
<|im_start|>assistant
<think>

</think>

```

分词器识别包括 `<think>` 在内的 added tokens。Unicode 15.1 的字母、数字、空白和组合标记类别表保存在 `src/unicode.h`,由 `tools/generate_unicode.py` 生成。重新生成时应使用相同 Unicode 数据版本;普通编译不需要 Python 或外部分词库。

DeltaNet 状态不能像 KV 缓存一样切回任意前缀。HTTP 现在保存一个可精确恢复的消息前缀检查点：

- 检查点在最后一条输入消息结束、assistant 生成前缀之前。下一轮会省略上一轮的空 `<think>` 前缀,所以不能直接复用上一轮生成结束时的状态。
- 保存递归矩阵、卷积窗口和 token ID,额外约 20.2 MB。K/V 的对应前缀保留在原位置,不另复制完整 K/V 缓存。
- 新请求只有完整匹配所保存的 token 前缀,且消息历史不短于检查点时才恢复。其余后缀重新计算,`cached_tokens` 返回实际跳过的数量。
- 不同问题、修改系统提示词、修改更早的历史或缩短历史,不满足匹配时会清空状态并重算。只保留最近一个检查点,不会无限累积。
- 交互对话上下文截断会使检查点失效,随后重算保留的历史。旧 qwen2/llama 的 K/V 前缀复用保留。
- `--no-prefix-cache` 可关闭 HTTP 前缀复用作对照。检查点内存分配失败时自动退回完整重算。

检查点之前的消息处理不计算未使用的输出 logits,生成前缀仍会重新计算下一 token 的预测。缓存主要缩短重复请求和多轮对话的首字等待,不会减少回答中的生成 token。首字延迟与生成 token/s 分开测量,复现方法见 [performance.md](performance.md)。

本次支持范围是文本聊天和补全,不包含视觉编码器、工具调用解析、MTP 推测解码或其他 qwen35 尺寸。`--raw` 仍允许调用者提供完整文本提示词。

## 验证

```
make check                               # 无需模型的数值测试
make test                                # 当前 2B 模型的完整回归
make test MODEL=/path/to/other-model.gguf # 原有架构回归
MODEL=models/qwen3.5-2b-q4_0.gguf ref/compare.sh "你好" 2 16
```

`tests/numerics.c` 用独立编码的 Q6_K 权重验证浮点/整数点积,并用双精度标量递归验证多步状态。分词 fixtures 来自 llama.cpp `5676dfbe01e7ebe0387913680eee0ed39bf4a363` 的 `llama_tokenize(add_special=false, parse_special=true)`,覆盖中文、emoji、组合音标、长 pre-token 和 added tokens。真实模型状态测试检查批量/逐 token 结果及截断后的重建。HTTP 测试检查 A/B/A 请求隔离、流式/非流式一致性、视觉输入拒绝、多轮检查点与完整重算的对照,以及历史缩短和系统提示词变化。状态测试另检查检查点恢复的逐项一致性。另用小型合成 GGUF 检查原有 qwen2/llama 加载和生成路径,无需重新下载旧模型。回归脚本另覆盖单线程、多线程、种子确定性、上下文截断与参考输出。

初步本机短测:树莓派 4,2 线程生成,中文提示词 40 token、生成 16 token,提示词约 3.07 token/s,生成约 1.22 token/s。此为满载约 84°C、桌面运行中的单次结果,不代表稳定速度或其他硬件表现,也未达到 10 token/s。

## 参考

- 模型与架构: https://huggingface.co/Qwen/Qwen3.5-2B
- 配置: https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json
- GGUF 文件: https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF
- 参考运行时: https://github.com/ggml-org/llama.cpp/tree/5676dfbe01e7ebe0387913680eee0ed39bf4a363
