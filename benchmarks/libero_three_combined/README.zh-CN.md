# LIBERO-PRO、Long++ 与 Ultra-Long 部署

[English](README.md)

三套 benchmark 服务保留原生仿真与评分。专用 Agent 为每个任务绑定已验证的 ICL
package，在整个 episode 中保持同一段对话，并通过锁定的 tokenizer 管理上下文。
Combined 镜像将 Agent 与 LiLo 或 PRO Benchmark 组合，支持每个任务只能指定一个
镜像的调度器。Long++ 与 Ultra-Long 共用 LiLo 组合，Agent 与原生 Python 包分别存放。

使用 Linux x86_64、Docker、NVIDIA GPU/驱动和 NVIDIA Container Toolkit 进行 EGL
渲染。源码构建与宿主 Runner 使用 Python 3.11。为原生镜像、Agent 环境、构建目录与
结果预留存储空间。模型通过兼容 OpenAI Responses 的 endpoint 调用。

将 `RUN_ROOT` 设为绝对输出路径，`MODEL_API_KEY_FILE` 设为已有的私有 key 文件；
通过 `OPENAI_BASE_URL` 导出模型 endpoint。每个单 case 输出目录必须尚不存在。

## 运行公开 combined 镜像

以下不可变的 `v1.0.1-egl` 产物由
[上游 PR 93](https://github.com/SenseTime-FVG/RoboRSI/pull/93)基于 commit
`5b8c7f7aa6e1b67119ef3e887625601387ce917e` 发布，保留该版本内置的预设与模型路由。
镜像保留该 revision 的目录布局。使用兼容其内置预设的 endpoint，或按下文重建以选择
模型。Combined CLI 接收 endpoint 与 key 文件；模型 ID 由安装后的 bundle 决定。

```bash
export LONGPP_COMBINED_IMAGE='ghcr.io/opensensenova/sensenova-roborsi-libero-long-plus-plus@sha256:814139261e914b114cb89a359c6a39d4f409a556f79928e986b6ade3dfe7dc7a'
export ULTRA_COMBINED_IMAGE='ghcr.io/opensensenova/sensenova-roborsi-libero-ultra-long@sha256:814139261e914b114cb89a359c6a39d4f409a556f79928e986b6ade3dfe7dc7a'
export PRO_COMBINED_IMAGE='ghcr.io/opensensenova/sensenova-roborsi-libero-pro@sha256:a83417f1459a4c555c457caa82ef00a5de50191e3d8deb943894fa9285cff172'
docker pull "$LONGPP_COMBINED_IMAGE"
docker pull "$ULTRA_COMBINED_IMAGE"
docker pull "$PRO_COMBINED_IMAGE"
```

Long++ 与 Ultra-Long 使用不同 GHCR package 名称，但 LiLo 镜像内容相同。每个镜像
包含对应 Benchmark、三套 Agent bundle、sealed ICL package 和 tokenizer。
模型凭据在运行时提供。

运行一个 Long++ case：

```bash
mkdir -p "$RUN_ROOT/results"
docker run --rm --gpus all \
  --mount "type=bind,src=$MODEL_API_KEY_FILE,dst=/run/secrets/model_api.key,readonly" \
  --mount "type=bind,src=$RUN_ROOT/results,dst=/results" \
  "$LONGPP_COMBINED_IMAGE" \
  --benchmark libero_long_plus_plus --task t00-i01 \
  --run-id longpp-one --results-root /results/longpp-one \
  --api-key-file /run/secrets/model_api.key --base-url "$OPENAI_BASE_URL"
```

Ultra-Long 使用 `$ULTRA_COMBINED_IMAGE`、`--benchmark libero_ultra_long` 和
`--task t00-i01`；PRO 使用 `$PRO_COMBINED_IMAGE`、`--benchmark libero_pro` 和
`--task libero_spatial_task-t00-i03`。

通过 subset runner 运行固定子集。以下命令使用上述公开镜像内的原始脚本路径：

```bash
docker run --rm --gpus all --entrypoint python \
  --mount "type=bind,src=$MODEL_API_KEY_FILE,dst=/run/secrets/model_api.key,readonly" \
  --mount "type=bind,src=$RUN_ROOT/results,dst=/results" \
  "$LONGPP_COMBINED_IMAGE" \
  -B /opt/roborsi/scripts/libero_three/run_combined_subset.py \
  --benchmark libero_long_plus_plus --run-id longpp-subset \
  --results-root /results/longpp-subset --max-concurrent 16 --gpu-count 1 \
  --api-key-file /run/secrets/model_api.key --base-url "$OPENAI_BASE_URL"
```

Ultra-Long 与 PRO 使用对应镜像和 benchmark；context map 分别规划 12、9、80 个
case。重复 `--task` 可选择较少 case。默认并发为 16，共用一张 GPU，请按宿主容量配置。
Runner 保留每次尝试；只有镜像/源码、Agent、ICL、endpoint、超时指纹均一致，进程
退出码为零、case 身份一致且原生 score 摘要完整时，才复用结果。只要仍有 case 缺少
已验证的原生终态，就以非零状态退出。其服务采用固定子集与 1× 边界。

Combined 模式让 Agent 与 Benchmark 共用容器；下文的独立容器模式为每个 case
提供独立的 Agent 文件系统和网络。

## 评测契约

| Benchmark | 官方全集 | 固定子集 | 官方 1× 预算 | 主评分 |
| --- | --- | --- | --- | --- |
| LIBERO-PRO Task/Swap | 8 个 suite × 10 个任务 × 10 个状态 = 800 | 80 个任务，state 3 | 10,000 物理步 | 最终目标成功 |
| LIBERO-Long++ | 12 个配置 × 10 次 bare-reset trial = 120 | 12 个配置，state 1 | 200 × 有序技能数 | 有序前缀比例；全部技能完成才成功 |
| LIBERO-Ultra-Long | 9 个配置 × 10 次 bare-reset trial = 90 | 9 个配置，state 1 | 200 × 有序技能数 | 有序前缀比例；全部技能完成才成功 |

Long++ 和 Ultra-Long 的 trial 标签不是随机种子。官方 profile 重复执行已发布的
bare random reset；`migration_subset` 明确采用固定 state 1。PRO 子集采用固定
state 3。所提供的 ICL map 覆盖这 12/9/80 个子集 case；运行全集时，Agent 的任务
覆盖范围需与所选 manifest 匹配。

Agent 可提交 7-DoF Panda `OSC_POSE` 动作，或已声明的反馈 primitive，包括
`move_to`、`move_pose`、腕部旋转和夹爪控制。Benchmark 将 primitive 转为物理动作，
每个物理步都计入同一预算。`rotate_wrist` 表示原地旋转，与 `move_pose.target_yaw`
的语义不同。

有序 benchmark 的 `lilo-instance-pose-feedback-v1` 提供注册实例名、实时世界坐标、
碰撞几何高度、精确夹爪接触、持有物体和当前技能进度，读取不会推进评分器。
`libero-robot-state-feedback-v1` 提供 `robot0_eef_pos`、xyzw 四元数和
`robot0_gripper_qpos`，同时保留紧凑的八维 `states`。这些反馈通道属于所命名的评测
条件，比较结果时需要保持一致。

`plus9_on_failure` 默认关闭。启用后，1× 边界失败的 case 可在同一环境和对话中继续，
最多执行 10× 总步数。`result.json` 与汇总始终保留不可变的 1× 分数；
`native/score.json` 另外记录累计恢复、物理步数和决策步数。有序评分遵循不间断的
已发布技能顺序，不用最终目标成功替代。

## Agent 预设与输入

| `configs/agents/libero/` 下的预设 | 安装后的 bundle | ICL package 数 |
| --- | --- | ---: |
| `long_plus_plus.yaml` | `bundle:libero_long_plus_plus_v13` | 12 |
| `ultra_long.yaml` | `bundle:libero_ultra_long_v13` | 9 |
| `pro.yaml` | `bundle:libero_pro` | 40，供 80 个 case 共用 |

`base.yaml` 和 `v13.yaml` 提供公共设置。三个叶子预设均使用 max reasoning 和精确
Token 计数策略：1,050,000-token 窗口、128,000-token 输出预留、32,768-token
额外预留、0.75 摘要触发比例、最近八组完整交换和两组图片。未达到阈值时保留完整历史。
PRO 允许 100 个模型轮次、5,000 秒 planner 预算，关闭模型重试；有序任务预设允许
10,000 个轮次、43,200 秒 planner 预算，并保留三次有界传输重试。

当前预设使用模型 ID `gpt-6-astra`。在预设或 `roborsi.agents.package --set` 中选择
服务支持的 ID；独立 Docker Agent 也支持 `bench eval --agent-set`。导出可移植 bundle
时，tokenizer 路径必须指向镜像内已验证的文件。

[资产锁](assets.lock.json)标识 61 个 sealed ICL package
与 tokenizer；[context map](../../configs/agents/libero/context_maps/)将任务映射到
package。`prepare_agent_assets.py` 在复制前验证完整清单与哈希；构建 Agent 镜像时，
`verify_agent_image.py` 再对安装后的 bundle 执行相同检查。
[重建步骤](#使用公开输入重建-agent)从公开 GHCR 镜像取得这些输入。

## 使用公开输入重建 Agent

在当前 checkout 根目录执行。将 `LIBERO_BUILD_ROOT` 设为源码目录外的路径；其中
`extracted-assets`、bundle 与 build 子目录必须尚不存在，并为本次构建使用新的
`dist/agent/`。将 `AGENT_BASE_TAG` 设为新的本地 Agent 基础镜像 tag；将
`AGENT_IMAGE_TAG`、`LILO_IMAGE_TAG`、`PRO_IMAGE_TAG` 设为你的容器仓库中的新 tag，
发布前完成登录。

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[runtime]' build uv tokenizers==0.22.2
mkdir -p "$LIBERO_BUILD_ROOT"
```

拉取上面的 Long++ combined 镜像后，从未启动的容器中复制资产；此操作不会运行
模拟器或 Agent。按当前资产锁验证并暂存三个 package family 与 tokenizer：

```bash
ASSET_CONTAINER=$(docker create "$LONGPP_COMBINED_IMAGE")
docker cp "$ASSET_CONTAINER:/opt/roborsi-agent-assets" "$LIBERO_BUILD_ROOT/extracted-assets"
docker rm "$ASSET_CONTAINER"
python benchmarks/libero_three_combined/prepare_agent_assets.py \
  --longpp-root "$LIBERO_BUILD_ROOT/extracted-assets/longpp/packages" \
  --ultra-root "$LIBERO_BUILD_ROOT/extracted-assets/ultra/packages" \
  --pro-root "$LIBERO_BUILD_ROOT/extracted-assets/pro/packages" \
  --tokenizer "$LIBERO_BUILD_ROOT/extracted-assets/tokenizer.json" \
  --output dist/agent/libero-three-assets
```

清单包含 12 个 Long++ package、9 个 Ultra-Long package，以及供 80 个 case 使用
的 40 个 PRO package；准备程序拒绝内容变化或额外 package。

为三个 bundle 配置镜像内 tokenizer 路径和哈希。若 endpoint 使用其他模型 ID，
将三处 `gpt-6-astra` 一致替换：

```bash
python -m roborsi.agents.package \
  --config configs/agents/libero/long_plus_plus.yaml \
  --name libero_long_plus_plus_v13 \
  --set 'runtime.llm.model="gpt-6-astra"' \
  --set 'runtime.context.tokenizer_file="/opt/roborsi-agent-assets/tokenizer.json"' \
  --set 'runtime.context.tokenizer_sha256="0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"' \
  --output "$LIBERO_BUILD_ROOT/bundle-longpp"
python -m roborsi.agents.package \
  --config configs/agents/libero/ultra_long.yaml \
  --name libero_ultra_long_v13 \
  --set 'runtime.llm.model="gpt-6-astra"' \
  --set 'runtime.context.tokenizer_file="/opt/roborsi-agent-assets/tokenizer.json"' \
  --set 'runtime.context.tokenizer_sha256="0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"' \
  --output "$LIBERO_BUILD_ROOT/bundle-ultra"
python -m roborsi.agents.package \
  --config configs/agents/libero/pro.yaml --name libero_pro \
  --set 'runtime.llm.model="gpt-6-astra"' \
  --set 'runtime.context.tokenizer_file="/opt/roborsi-agent-assets/tokenizer.json"' \
  --set 'runtime.context.tokenizer_sha256="0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"' \
  --output "$LIBERO_BUILD_ROOT/bundle-pro"
python -m build --wheel --outdir dist/agent "$LIBERO_BUILD_ROOT/bundle-longpp"
python -m build --wheel --outdir dist/agent "$LIBERO_BUILD_ROOT/bundle-ultra"
python -m build --wheel --outdir dist/agent "$LIBERO_BUILD_ROOT/bundle-pro"
```

合并三个 wheel 的 requirements 与 Agent 基础 requirements。预设声明
`tokenizers==0.22.2`；另外加入 SciPy，因为 combined 镜像会检查
`scipy.spatial.transform.Rotation`。版本 1.15.3 与原生 base 对齐，生成的 lock
固定全部版本与哈希。
本地框架 wheel 提供当前 checkout 的依赖元数据，Dockerfile 安装同一 checkout 的源码。

```bash
python -m build --wheel --outdir "$LIBERO_BUILD_ROOT/framework-wheel" .
python -c 'from pathlib import Path; import sys; print("roborsi @ " + next(Path(sys.argv[1]).glob("roborsi-*.whl")).resolve().as_uri())' \
  "$LIBERO_BUILD_ROOT/framework-wheel" > "$LIBERO_BUILD_ROOT/roborsi-override.txt"
printf 'scipy==1.15.3\n' > "$LIBERO_BUILD_ROOT/combined-runtime.in"
uv pip compile \
  "$LIBERO_BUILD_ROOT/bundle-longpp/requirements.in" \
  "$LIBERO_BUILD_ROOT/bundle-ultra/requirements.in" \
  "$LIBERO_BUILD_ROOT/bundle-pro/requirements.in" \
  agents/python/requirements.in "$LIBERO_BUILD_ROOT/combined-runtime.in" \
  --overrides "$LIBERO_BUILD_ROOT/roborsi-override.txt" --no-emit-package roborsi \
  --python-version 3.11 --generate-hashes --output-file dist/agent/requirements.txt
docker build --target configured -f agents/python/Dockerfile \
  --build-arg AGENT_REQUIREMENTS=dist/agent/requirements.txt \
  --build-arg 'AGENT_WHEEL=dist/agent/roborsi_agent_bundle_libero_*.whl' \
  -t "$AGENT_BASE_TAG" .
docker build -f benchmarks/libero_three_combined/Agent.Dockerfile \
  --build-arg AGENT_BASE_IMAGE="$AGENT_BASE_TAG" \
  --build-arg LIBERO_THREE_ASSETS=dist/agent/libero-three-assets \
  -t "$AGENT_IMAGE_TAG" .
docker push "$AGENT_IMAGE_TAG"
export AGENT_IMAGE=$(docker image inspect "$AGENT_IMAGE_TAG" --format '{{index .RepoDigests 0}}')
```

镜像构建会验证 bundle 任务 map、全部 package 哈希、清单与 tokenizer 身份，并执行
依赖检查。保留生成的 lock、bundle manifest、源码 revision 和实际 repository digest。

## 重建并注册 Benchmark

按照 [PRO 指南](../libero_pro/README.md)准备完整评测目录，并将
`PRO_EVALUATION_SOURCE` 设为该目录。要求 SHA-256 为
`04ae2020acf89315108a68743f99f8fe9c4cb9fc47e67712298a25a7179d616a`。
固定原生源码与资产细节见 [LiLo 说明](../libero_lilo/README.md)。
构建并发布当前两个 context：

```bash
python -m benchmarks.package --benchmark libero_lilo --output "$LIBERO_BUILD_ROOT/build-lilo"
python -m benchmarks.package --benchmark libero_pro --output "$LIBERO_BUILD_ROOT/build-pro"
python -m benchmarks.libero_pro.assets \
  --source "$PRO_EVALUATION_SOURCE" --output-dir "$LIBERO_BUILD_ROOT/pro-assets" \
  --build-context "$LIBERO_BUILD_ROOT/build-pro"
docker build -t "$LILO_IMAGE_TAG" "$LIBERO_BUILD_ROOT/build-lilo"
docker build -t "$PRO_IMAGE_TAG" "$LIBERO_BUILD_ROOT/build-pro"
docker push "$LILO_IMAGE_TAG"
docker push "$PRO_IMAGE_TAG"
export LILO_IMAGE=$(docker image inspect "$LILO_IMAGE_TAG" --format '{{index .RepoDigests 0}}')
export PRO_IMAGE=$(docker image inspect "$PRO_IMAGE_TAG" --format '{{index .RepoDigests 0}}')
```

在同一批镜像中生成 manifest，使代码、内置资产与运行时标识一致。以下选择固定的
12/9/80 子集：

```bash
mkdir -p "$RUN_ROOT/registry/libero_long_plus_plus" \
  "$RUN_ROOT/registry/libero_ultra_long" "$RUN_ROOT/registry/libero_pro"
docker run --rm --entrypoint python \
  --mount "type=bind,src=$RUN_ROOT/registry,dst=/output" "$LILO_IMAGE" \
  -m benchmarks.libero_lilo.manifest --image "$LILO_IMAGE" \
  --benchmark libero_long_plus_plus --profile migration_subset \
  --output /output/libero_long_plus_plus/manifest.json
docker run --rm --entrypoint python \
  --mount "type=bind,src=$RUN_ROOT/registry,dst=/output" "$LILO_IMAGE" \
  -m benchmarks.libero_lilo.manifest --image "$LILO_IMAGE" \
  --benchmark libero_ultra_long --profile migration_subset \
  --output /output/libero_ultra_long/manifest.json
docker run --rm --entrypoint python \
  --mount "type=bind,src=$RUN_ROOT/registry,dst=/output" "$PRO_IMAGE" \
  -m benchmarks.libero_pro.manifest --image "$PRO_IMAGE" --profile migration_subset \
  --output /output/libero_pro/manifest.json
```

Manifest 为 EGL 申请一张 GPU。`official` 选择 benchmark 全集；所提供的专用
context map 覆盖固定子集。`--plus9-on-failure` 用于单独记录的继续执行条件。

## 使用独立容器评测

使用重建后的 Agent digest 与已安装 bundle，凭据只传入 Agent。先查看展开后的
Long++ 计划：

```bash
export OPENAI_API_KEY="$(cat "$MODEL_API_KEY_FILE")"
bench eval --registry-root "$RUN_ROOT/registry" \
  --results-root "$RUN_ROOT/results" --run-id longpp-plan \
  --benchmark libero_long_plus_plus --seed 0 \
  --agent-image "$AGENT_IMAGE" --agent-config bundle:libero_long_plus_plus_v13 \
  --agent-credential-env OPENAI_API_KEY --agent-credential-env OPENAI_BASE_URL \
  --agent-set 'runtime.llm.model="gpt-6-astra"' \
  --max-concurrent 1 --timeout-s 43200 --case-timeout-s 14400 --dry-run
```

去掉 `--dry-run`，选择新 run ID 并保持所选模型 ID，即可执行。添加
`--task libero_long_plus_plus:t00-i01` 可只跑一个 case。其他子集使用以下组合：

| Benchmark | Bundle | 示例任务 |
| --- | --- | --- |
| `libero_ultra_long` | `bundle:libero_ultra_long_v13` | `libero_ultra_long:t00-i01` |
| `libero_pro` | `bundle:libero_pro` | `libero_pro:libero_spatial_task-t00-i03` |

使用 `--resume-from OLD_RUN_ID` 时选择新 run ID 并保持锁定输入一致。
结果查看见[评测指南](../../docs/evaluation.zh-CN.md)。分别统计有分数终态、Agent/基础设施失败
以及原生证据。

## 组合重建后的镜像

将 `NEW_LILO_COMBINED_TAG` 和 `NEW_PRO_COMBINED_TAG` 设为新 tag。在仓库根目录，
使用前面取得的实际 Agent 与 Benchmark digest 构建：

```bash
docker build -f benchmarks/libero_three_combined/Dockerfile \
  --build-arg AGENT_IMAGE="$AGENT_IMAGE" --build-arg BENCHMARK_IMAGE="$LILO_IMAGE" \
  --build-arg ROBORSI_REVISION="$(git rev-parse HEAD)" \
  -t "$NEW_LILO_COMBINED_TAG" .
docker build -f benchmarks/libero_three_combined/Dockerfile \
  --build-arg AGENT_IMAGE="$AGENT_IMAGE" --build-arg BENCHMARK_IMAGE="$PRO_IMAGE" \
  --build-arg ROBORSI_REVISION="$(git rev-parse HEAD)" \
  -t "$NEW_PRO_COMBINED_TAG" .
docker push "$NEW_LILO_COMBINED_TAG"
docker push "$NEW_PRO_COMBINED_TAG"
docker image inspect "$NEW_LILO_COMBINED_TAG" "$NEW_PRO_COMBINED_TAG" \
  --format '{{json .RepoDigests}}'
```

Long++ 与 Ultra-Long 共用重建后的 LiLo combined 镜像。记录当前源码 revision、
两个输入 digest 和新的输出 digest。默认 entrypoint 支持前面的单 case 参数。
运行固定子集时，使用重建镜像内的新路径：

```bash
export NEW_LILO_COMBINED_IMAGE=$(docker image inspect "$NEW_LILO_COMBINED_TAG" --format '{{index .RepoDigests 0}}')
export NEW_PRO_COMBINED_IMAGE=$(docker image inspect "$NEW_PRO_COMBINED_TAG" --format '{{index .RepoDigests 0}}')
docker run --rm --gpus all --entrypoint python \
  --mount "type=bind,src=$MODEL_API_KEY_FILE,dst=/run/secrets/model_api.key,readonly" \
  --mount "type=bind,src=$RUN_ROOT/results,dst=/results" \
  "$NEW_LILO_COMBINED_IMAGE" \
  -B /opt/roborsi/benchmarks/libero_three_combined/run_combined_subset.py \
  --benchmark libero_long_plus_plus --run-id rebuilt-longpp-subset \
  --results-root /results/rebuilt-longpp-subset --max-concurrent 16 --gpu-count 1 \
  --api-key-file /run/secrets/model_api.key --base-url "$OPENAI_BASE_URL"
```

Ultra-Long 保持该镜像并选择 `--benchmark libero_ultra_long`；PRO 使用
`$NEW_PRO_COMBINED_IMAGE` 和 `--benchmark libero_pro`。每次选择独立的 run ID 与输出目录。
上述新路径用于当前源码构建的镜像；前面的不可变公开镜像仍使用原来的
`/opt/roborsi/scripts/libero_three/` 路径。

## 结果与溯源

每个完成的 case 记录 `result.json`、原生评分、可用的原生视频、内容寻址观测、
Runner 消息和 Agent trace/config 指纹。运行记录应保留镜像摘要、源码 revision、
manifest、模型 endpoint 标识与预算。`success` 和原生 `task_failure` 属于有分数
的终态；基础设施与 Agent 错误需单独统计。
