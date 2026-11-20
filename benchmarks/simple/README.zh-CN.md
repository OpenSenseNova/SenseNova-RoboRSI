# SIMPLE 原生评测接入候选版

[English](README.md) · [构建说明](BUILD.md) · [验证记录](VALIDATION.md)

## 迁入 SenseNova-RoboRSI — 2026-11-18

SIMPLE Benchmark 源码、构建脚本、Agent 预设和测试从本地 `NewRPent` 迁入
`SIMPLE` 分支。标准导出命令已支持 `--benchmark simple`。下文的镜像摘要、PR
状态与原生验证结果均保留自来源仓库，不能作为当前共享框架对应镜像的验证结果。
部署此版本前需重新构建，并核对配套的镜像与源码摘要；详见
[验证记录](VALIDATION.md)。

本分支提供 SIMPLE Provider/Adapter、独立主线程仿真 worker、私有 reset catalog
导出和标准外部 EmbodiedAgent API 预设。不会把旧的“模型＋仿真混合包”直接放进
Benchmark 镜像；已发布服务候选版，不声称复现论文分数或完成完整任务验收。

## 当前状态

- 公共源码登记 #86 已合并，提交为 `2d06bcf1417f779f68611ab3671cd0e13ee63056`。
  本分支已同步 main，包含登记和 #88 的 facade 清理重试修复，可直接使用标准导出和
  服务身份检查。清单见 [FRAMEWORK_REQUEST.md](FRAMEWORK_REQUEST.md)。
- 修复候选版已发布：`ghcr.io/opensensenova/sensenova-roborsi-simple:v0.1.1-cu128`。
  固定地址为 `ghcr.io/opensensenova/sensenova-roborsi-simple@sha256:4b6f9eb28fa87aebba7a3d46fcf0498753d81a47719dadf30a78c22d1f208213`。
  已包含 adapter 和 facade 的清理重试修复，在构建机完成上传、按 digest 拉取及
  运行时源码校验；对方机器与任务成功分支的评分尚未验收。
  启动命令、版本及验证范围见 [DELIVERY.md](DELIVERY.md)。历史 v0.1.0 缺少这些修复。
  本轮 GitHub CI 因账户账单/额度问题在执行前被阻塞，PR 保持 Draft。
- 独立的[原生底座构建脚本](native/README.md)安装固定版本的仿真和控制依赖，
  不包含 Agent SDK 或凭据；catalog、场景资产和历史 ICL 参考包单独提供。
- 新 Agent 预设不是历史实验 Agent：没有历史冻结 ICL 包、数值/FK/IK 隔离工具及
  max/pro 自定义请求扩展，不能仅改模型名就声称复现论文分数。
- 不修改公共框架、全局依赖、上游 SIMPLE 或驱动。

## 原生条件

SIMPLE 固定 `6d10628794d9c7de4596b4f2afb2c054a637c2bc`；数据来自
`USC-PSI-Lab/psi-data` 的 `e78fb93cc28912a3031a10b8656d32d7f0a2b867`。
各任务 test ZIP 的大小和 SHA256 见 `robots/simple/sources.json`。

| 任务 | 1×执行步 | 控制器 |
| --- | ---: | --- |
| xmove | 800 | decoupled WBC |
| bendpick | 400 | AMO |
| handover | 800 | decoupled WBC |
| tabletop | 800 | AMO |
| mobile_pick | 1200 | decoupled WBC |
| xmove_bendpick | 800 | decoupled WBC |

每任务 L0/L1/L2 各10例；显式10×为原动作行预算的十倍。动作保持36维连续 Dex3
接口，单行名义0.02秒。HTTP step 统计动作块，`native_env_steps` 统计真实动作行。
共享 HTTP 大小限制下单块最多512行，且不超过剩余物理预算；不裁剪、不补动作。
原生终止可以提前停止块内剩余行。

评分采用已披露的历史 worker 条件：AMO 0.9、非移动 WBC 0.5、移动任务使用原生
metadata。不是已认证的官方榜单 CLI 阈值。由原生 `task.check_success` 给结果；
模型 finish 和 hold-action smoke 不表示任务成功。推理期间暂停物理，WBC原有壁钟
语义不改。真机 G1/Wuji open/close、训练/ICL准备属于另一条交付路径。

## 独立安装与运行

主控/Agent 用 Python3.10–3.12，按 CONTRIBUTING 安装；Isaac4.5参考原生栈用独立
Python3.10环境，按上游 SIMPLE 文档安装、初始化子模块并获取 LFS 权重和机器人网格。
上游 commit 必须匹配，`src/simple` 不允许未登记修改；本地补丁需另行声明并验证。
本分支不提供驱动修复或任意 H200 栈兼容承诺。

Agent 使用归档 `episodes.jsonl` 中的任务文本；部分 reset 的语言模板与其不一致，
例如 XMove L0/E00 的 reset 写着“弯腰”，任务文本写着“向前走”。启动时按 reset
模板和目标名称校验原生指令，并在私有 reset 证据中记录两份文本及
`instruction_match`；不自动改写任务 prompt 或 reset。上游评测 CLI 使用原生生成
的指令，因此当前 prompt 选择尚不能认定与官方评测等价。
原生 checkout 内先执行 `git lfs install --local`，再获取 LFS 文件；否则已经下载的
权重可能仍被 Git 当成源码改动。运行栈还需 `libGLU.so.1`、`libvulkan.so.1` 和 NVIDIA
图形库；非系统安装的库需在宿主环境配置相应路径。

英文 README 给出准确的 `hf download`、catalog 导出及服务命令。导出必须指向新目录，
验证 ZIP 后保留全部30例，沿用 scene UID 转换并逐例写 reset SHA256。reset 几何不进入
公开 Task/Observation。默认 seed base20261002 是显式 RNG条件，不代表所有历史回合的
真实 seed。禁止模型读私有 catalog、物体位姿和评测源码。

核心变量：`ROBORSI_SIMPLE_CATALOG`、`ROBORSI_SIMPLE_ROOT`、
`ROBORSI_SIMPLE_PYTHON`，可选 `ROBORSI_SIMPLE_ASSETS`。阅读并接受 NVIDIA EULA
后才能设置 `OMNI_KIT_ACCEPT_EULA=Y`。每服务分配一张 GPU 和一个活动原生case；
`ROBORSI_SIMPLE_GPU_INDEX` 默认0，同时选择 Torch/Isaac 设备，两者编号必须一致；
不要只改变 `CUDA_VISIBLE_DEVICES` 的编号映射，EGL 时也需对应的 `MUJOCO_EGL_DEVICE_ID`。
Kit 缓存与日志放在每个case内，AMO图里的 CUDA 设备常量只在内存中映射，权重文件不变。
并行原生分配使用不同 `ROBORSI_SIMPLE_DDS_DOMAIN`。Headless 仍需正确图形库和设备。
旧镜像构建时组合了本分支与 PR #86 的登记提案；该登记现在已合入。
当前源码可直接使用公共 exporter 和摘要核验；新旧镜像验证范围见 DELIVERY.md。

模型只在外部 Agent 环境运行，从 `robots/simple/agent.yaml` 选择预设并配置模型和
endpoint；密钥不进入仓库或仿真 worker。标准 Agent 导出使用已有
`roborsi.agents.package`，只携带声明的prompt/config，不复制仿真器或私有reset。
预设显式1000模型轮上限，不设单次token上限或planner超时；它与机器人执行预算不同。

此版反馈每个动作块末尾的图像、32维state和预算；不复制历史文件工具的中间图像历史。
历史 ICL 包与原 Agent 的完整迁移需单独实现和验证。不要把测试日志、视频、权重、
生成catalog/manifest、填写后的环境文件提交进代码 PR。

## 验证

公共源码登记 [PR #86](https://github.com/SenseTime-FVG/RoboRSI/pull/86) 已合入，
本分支已同步，可使用标准 exporter 构建服务镜像。本接入 PR #85 保持 Draft，
等待修复审阅和镜像交付验收。Draft 的主要 GitHub CI 会被跳过，
聚合检查显示绿色不能当作所有测试已经执行。

对已启动的服务，可在外部主控环境执行最多三步、零模型调用的检查：

```bash
python -m benchmarks.simple.smoke_http --endpoint http://127.0.0.1:8000 \
  --task-id tabletop/L0/E00 --output /data/new-simple-http-smoke \
  --source-sha256 SHA256_FROM_EXPORTED_SOURCE_JSON
```

输出目录必须新建，SHA256 来自该次导出的 `source.json`，不是镜像 digest。
检查真实服务身份、reset、RGB/state 下载、最多三步执行及本次 case 的关闭和删除。
不缩小原生任务预算，不把提前停止计成任务失败；只有原生真的终止才调用 finish。
非 root 子进程现在只信任显式配置的源码目录，并把 LFS 临时文件放在自己的回合目录，
无需透传 `GIT_CONFIG_*`；源码版本和改动检测仍保留。
`requirements-http.lock` 固定 HTTP 层的11个依赖和哈希，不等于完整原生栈的哈希锁。

运行 `pytest tests/unit_tests/robots/simple -v`、`pre-commit run --all-files`、
`pytest tests/unit_tests -v`。实际执行范围和未完成验收见 `VALIDATION.md`。
离线测试使用假物理/假模型，保留真实进程、RPC、HTTP、媒体及结果工件；不等于原生任务成功。

原生环境和上述变量配置好后，可在主控测试环境运行无需模型的三步检查：

```bash
export ROBORSI_RUN_SIMPLE_NATIVE=1
export ROBORSI_SIMPLE_EVIDENCE_DIR=/data/new-simple-native-smoke
pytest tests/integration_tests/robots/simple/test_native.py -v
```

输出目录必须是新目录。检查catalog第一例，保留PNG、worker和Kit日志。此检查直接验证
Adapter，不需要共享源码登记；不绕过reset/动作验证，也不代表完整任务成功、模型可用
或全部六个任务均通过。

## 六任务候选交付

候选镜像、启动命令和 Registry 验证范围见 [DELIVERY.md](DELIVERY.md)，公开资产的固定来源与校验见 [ASSETS.md](ASSETS.md)。
`python -m benchmarks.simple.serve` 使用单张指定 GPU、只读镜像和资产、独立可写输出目录启动标准 HTTP 服务。
另提供 `python -m benchmarks.simple.check_terminal`：固定保持初始状态与零移动命令，
执行到原生终止或原始1×步数上限，验证 finish、分数与下载动作日志一致、结果下载及清理。
这是独立于三步 smoke 的无模型接口检查，不统计成 Agent 任务结果；命令和实际验证范围见 DELIVERY/VALIDATION。
发布镜像已实测 BendPick L0/E00：保持原400步预算，耗时343.12秒，正常截断后 finish 返回失败及0分，
与400行动作记录一致；46个产物下载、回合删除及容器退出均通过。成功任务的原生评分仍未验证。

固定 SIMPLE 版本删除了 `XMoveBendPickTeleop`。候选镜像从上游提交 `c0a01cea95780a7eb56141686b7f25d38713e19a` 下载原始任务文件及 MIT 许可证，核对 SHA256 后加载；不修改文件，也不替换成其他任务。原始 `check_success` 的阈值是 reward >= 0.8，worker 的同名阈值属性不改变此硬编码判断。兼容来源写入 reset 证据。

Registry 清单使用新导出的目录；旧目录中额外的 `state_dim` 字段不符合公共 manifest schema，须用新版 `prepare` 重新导出。模型收到的状态仍为 32 维。评分协议字符串仍保存在每个 task 的 native 元数据中。
框架默认 Docker 执行器只有 1 GB `/tmp`，而这里的 Isaac shader 缓存可超过 1 GB；候选的原生验收通过独立 HTTP 服务进行，未宣称默认 Docker 执行器完整验收。当前仍为候选交付，等待修复审阅与镜像验收。
