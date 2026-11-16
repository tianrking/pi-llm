# pillm

从零编写的大模型推理引擎,只针对一块芯片:树莓派 4 的 BCM2711(4× Cortex-A72)。

目标是成为 Pi 4 上最快的独立推理工具:先在精简 Linux 上超过 llama.cpp,再去掉操作系统直接裸机运行。

## 特点

- 纯 C,无第三方依赖,直接读取 GGUF 模型文件。
- 运算核为 A72 手写:没有 SDOT 指令,用 `smlal` 一次处理四行权重,并预取后续行。
- 对具有低精度词表副本的模型,输出层先筛选 256 个候选再精确计算。Qwen3.5-2B 的共享 Q6_K 输出层计算完整词表。
- 提示词按 16 个 token 一批处理,权重每批只读一遍。
- 自带 BPE 分词器(中英文)、交互对话、温度 / top-k / top-p 采样。
- 内置 HTTP 服务:OpenAI 兼容接口和网页聊天界面。

支持 `qwen2`、`llama` 和文本版 Qwen3.5-2B。Qwen3.5 的混合注意力、状态重建和验证范围见 [docs/qwen35.md](docs/qwen35.md)。支持范围、已验证的模型和这块板子能跑的模型规模上限见 [docs/models.md](docs/models.md)。

## 编译和运行

```
ref/fetch-qwen35.sh                    # 固定版本下载并校验约 1.30 GB 的模型
make
./pillm -m models/qwen3.5-2b-q4_0.gguf                 # 交互对话
./pillm -m models/qwen3.5-2b-q4_0.gguf -p "你好"        # 单次提问
./pillm -m models/qwen3.5-2b-q4_0.gguf -p "你好" --temp 0 --prof   # 贪心解码并显示各阶段耗时
```

不带参数运行 `./pillm` 查看全部选项。对话写满上下文(`-c`,默认 4096 个 token)时会自动丢弃最早的一半内容继续,系统提示词保留。

`make check` 无需模型,检查数值运算。`make test` 跑完整回归(需要模型文件和数分钟;已有 llama.cpp 参考版本时会一并对比输出)。可用 `make test MODEL=/path/to/model.gguf` 测其他模型。参考版本的编译见 `ref/setup.sh`。

## 服务模式

```
./pillm -m models/qwen3.5-2b-q4_0.gguf --serve 8080                  # 只有本机能访问
./pillm -m models/qwen3.5-2b-q4_0.gguf --serve 8080 --host 0.0.0.0   # 局域网内都能访问
```

- 浏览器打开 `http://<地址>:8080/` 是内置的聊天页面,流式显示回答和速度。
- `POST /v1/chat/completions` 与 OpenAI 的接口兼容,支持 `messages`、`max_tokens`、`temperature`、`top_p`、`top_k`、`seed`、`stream`。
- `GET /v1/models`、`GET /health`。
- 每次请求带完整对话。Qwen3.5 恢复完整匹配的消息前缀检查点,再计算剩余内容;不匹配时重算。原有 qwen2/llama 复用相同 K/V 前缀。可用 `--no-prefix-cache` 关闭 HTTP 复用作对照。
- 一次处理一个请求。没有密码,`--host 0.0.0.0` 只应在可信的网络里使用。

```
curl localhost:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"你好"}]}'
```

开机自启动见 `deploy/pillm.service`。

## 当前默认模型

Qwen3.5-2B 的 Q4_0 GGUF 为约 1.30 GB,可在 8 GB Pi 4 上运行。初步中文短测生成约 1.22 token/s,尚未达到 10 token/s;温度、上下文和后台负载都会影响速度。仅支持文本,不执行视觉编码器或 MTP 层。完整说明见 [docs/qwen35.md](docs/qwen35.md)。后续性能优化、首字延迟与交错测速的复现方式见 [docs/performance.md](docs/performance.md)。

## 与 llama.cpp 的历史对比

同一块板子、同一个模型(qwen2.5-0.5b-instruct-q4_0)、贪心解码、生成 100 个 token。测试时没有散热器,芯片 84°C 处于降频状态,桌面环境开着。llama.cpp 为 2026-11-13 的 `5676dfb`,用 `-mcpu=cortex-a72` 编译,各取 1–4 线程中最好的结果。

| 模型 | 引擎 | 读提示词 | 生成 |
|---|---|---|---|
| 0.5B | llama.cpp | 15.2 token/s(3 线程) | 5.6 token/s(2 线程) |
| 0.5B | pillm | 18.3 token/s | 7.5 token/s(2 线程) |
| 1.5B | llama.cpp | 3.5 token/s(2 线程) | 2.0 token/s(2 线程) |
| 1.5B | pillm | 5.2 token/s | 2.7 token/s(2 线程) |
| 3B | llama.cpp | 1.3 token/s(2 线程) | 1.0 token/s(2 线程) |
| 3B | pillm | 2.9 token/s | 1.4 token/s(2 线程) |

pillm 生成快 32–38%,读提示词快 20–120%。1.5B 和 3B 模型只测了 2 线程的 llama.cpp。运行时占用内存约等于模型文件大小(1.5B 约 1.0 GB,3B 约 2.0 GB)。

正确性:

- 分词结果与 llama.cpp 完全一致(中英文、标点、全角字符、表情符号混合的测试串)。
- 英文测试问题的贪心输出逐字相同(0.5B 和 3B 模型 60 个 token,1.5B 模型 70 个 token)。
- 中文测试问题在第 14 个 token 处分叉,该处前两名候选的分数只差 0.02(12.653 对 12.630)。两个实现的浮点累加顺序不同,这种近似平局会翻转,属于正常现象。

上述是原有 Qwen2.5 的历史记录。可提供相应旧模型,用 `MODEL=/path/to/model.gguf ref/compare.sh "问题" 线程数 token数` 复现;Qwen3.5 的验证见 [docs/qwen35.md](docs/qwen35.md)。

## 线程数

默认生成时用 2 个线程,读提示词时用 4 个。这是此前 Qwen2.5 测试选出的配置,Qwen3.5 暂沿用此默认,尚未完成独立的线程调优。带宽、温度和线程影响的历史分析见 [docs/limits.md](docs/limits.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `src/pillm.c` | 模型、前向计算、分词器、采样、命令行 |
| `src/qwen35.h` | Qwen3.5-2B 混合注意力、卷积和递归状态 |
| `src/unicode.h` | 生成的 Unicode 分词类别表 |
| `src/kernels.h` | 量化和矩阵运算核(NEON) |
| `src/server.h` | HTTP 服务、JSON 解析、聊天页面 |
| `src/gguf.c` | GGUF 文件解析 |
| `bench/limits.c` | 测内存和 CPU 的吞吐上限 |
| `bench/cycles.c` | 用硬件计数器测运算核的周期数 |
| `ref/` | llama.cpp 参考版本的构建和对比脚本 |
| `tests/run.sh` | CLI、HTTP 和参考输出回归 |
| `tests/numerics.c` | 无需模型的数值测试 |
| `tests/qwen35.py` | Qwen3.5 分词和请求状态隔离测试 |
| `deploy/pillm.service` | systemd 开机自启动配置 |
| `docs/limits.md` | 这块芯片的性能上限分析 |
| `docs/models.md` | 支持的模型和规模上限 |

## 路线

1. 正确性:输出与 llama.cpp 逐 token 一致。
2. 速度:超过 llama.cpp。
3. 更多模型:K 系列量化(Q4_K_M 等)、SentencePiece 分词器。
4. 裸机:自带启动代码、页表、多核、SD 卡读取,不依赖 Linux。
