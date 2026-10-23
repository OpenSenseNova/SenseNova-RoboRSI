# Model API deployment / 模型 API 部署

This directory contains the [OpenPI VLA Docker environment](openpi/Dockerfile)
and reserves an entry point for the main SenseNova-RoboRSI model API.
[`deploy_model.sh`](deploy_model.sh) is currently an
**unimplemented placeholder**: it prints a message and exits with status 1.
It does not install dependencies, download weights or start a server.

此目录包含已经接入一键评测的 [OpenPI VLA 容器环境](openpi/Dockerfile)，并预留
SenseNova-RoboRSI 主模型的正式部署入口。目前
[`deploy_model.sh`](deploy_model.sh) **尚未实现部署**，执行时会输出提示并以状态码 1
退出。待确认推理框架、模型路径及硬件配置后，将其中的占位内容替换为经过验证的启动命令。
已有可用模型 API 时，直接跳过此步骤。

LIBERO VLA deployment is implemented and managed automatically when the selected
Agent enables it; see [the guide](../../docs/libero-vla.md) /
[VLA 自动下载与部署](../../docs/libero-vla.zh-CN.md). This is independent of the
main-model placeholder above. 所选 Agent 启用 VLA 时，无需手动执行部署命令。

Main API and VLA configuration share the selected `configs/models/default.yaml`
(or `--model-config PATH`). Weight/tokenizer downloads belong to
[`scripts/evaluation/downloads.py`](../evaluation/downloads.py); the OpenPI
serving entry point loads prepared local assets.
主模型和 VLA 使用同一个模型 YAML；下载逻辑统一在下载脚本，部署入口负责加载与服务。

Keep serving dependencies in a separate environment from the evaluation Runner.
Run the completed deployment script in one terminal, wait until the API is ready,
and run the evaluation script in another terminal. The model service must remain
available throughout evaluation.

建议模型推理服务与评测 Runner 使用独立环境。部署脚本补齐后，在一个终端启动服务，
确认 API 就绪，再在另一个终端执行评测脚本；评测期间保持模型服务运行。

## Required interface / 接口约定

| Evaluation setting | Deployment contract / 部署约定 |
| --- | --- |
| `base_url` / `OPENAI_BASE_URL` | Model API base URL, including the API prefix when required, e.g. `http://127.0.0.1:8001/v1`. 模型 API 地址，包含服务要求的路径前缀。 |
| `api_key` / `OPENAI_API_KEY` | Key accepted by the server. For a server with no authentication, explicitly use a nonempty placeholder such as `EMPTY`. 仅经环境变量传递，勿提交真实密钥。 |
| `model_name` / `MODEL_NAME` | Exact served model name accepted in API requests, which may differ from the checkpoint path. 服务接受的模型名称，可能与权重路径不同。 |
| `--openai-format chat` | Default: OpenAI-compatible Chat Completions, including image input and tool calls. 默认使用 Chat Completions，必须支持视觉输入与工具调用。 |
| `--openai-format responses` | Select this only when the server supports the Responses API with those capabilities. 服务支持 Responses 时显式选择。 |

The deployment implementation should document its dependencies, checkpoint,
served name, listening address, GPU requirements and readiness check. Use a
foreground `exec` command so the process manager or operator controls its lifetime.
Do not bake real API keys into the script or source-controlled configuration.

正式部署脚本应说明依赖版本、权重来源、服务模型名、监听地址、GPU 要求及就绪检查方式；
建议以 `exec` 前台启动服务，方便操作者或进程管理器控制生命周期。

The Benchmark simulator service and RoboCasa365's RLDX execution service are
separate dependencies. Their addresses are not the model `base_url`.
模型 API、Benchmark 仿真服务和 RoboCasa365 的 RLDX 执行服务是不同服务，地址不能混用。

Continue with [model evaluation](../../docs/model-evaluation.md) /
[模型一键评测](../../docs/model-evaluation.zh-CN.md).

## SenseNova-RoboRSI-27B: VLA/SAM3

[`libero_tools.py`](libero_tools.py) prepares the tool environment, downloads
RLinf Pi0.5/SAM3 weights, and serves both tools.
See the [English commands](../../docs/sensenova-roborsi-27b-preview.md).

该脚本准备工具环境、下载 RLinf Pi0.5/SAM3 权重并部署两个服务。
使用方式见 [27B 部署与评测命令](../../docs/sensenova-roborsi-27b-preview.zh-CN.md)。
