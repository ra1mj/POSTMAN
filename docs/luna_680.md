# Gento Luna V3 680 强化学习遥操作

控制链路为浏览器目标 → POSTMAN 残差 PPO/WBC → `gento_sdk_dds_ui_bridge` → Cyclone DDS → FX 和 RDK。
`luna_control` 使用 Thor 本机的 DDS domain `226`。Gateway 通过 FX 控制上身，通过原生 RDK DDS
控制底盘。手部 XYZ 和 RPY 是相对锚点的 `base_link` 坐标。底盘指令是车体系
`[vx, vy, wz]`，不依赖 RDK 提供有效的全局航向。

| 接口 | 顺序与单位 |
| --- | --- |
| FX `waist` | `Leg_J1_Joint` 到 `Leg_J6_Joint`，rad |
| FX `left_arm` / `right_arm` | 左右臂各按 1 到 7 排列，rad |
| FX `head` | `Head_J1_joint`、`Head_J2_Joint`，rad。按交付 `joint_signs.json` 转换第二轴符号。 |
| RDK `base` | 车体系前后 m/s、左右 m/s、转向 rad/s |
| 策略末端 | URDF 的 `flange_L` 和 `flange_R` |

此版本使用 680 无夹爪模型。[天机官方 Luna 模型下载](https://docs.gentorobotics.ai/zh-CN/resources/downloads/#仿真模型urdf-2)

## 安装与训练

保留 URDF 和同一交付包的 `marvin_description/meshes` 目录。默认模型位置和
Gento 控制台一致，可用环境变量或 `--urdf` 指定。随包 PPO 权重对应 URDF SHA-256
`5d7042aae5372321a7fb784f83ba4c7d6c5b229b3915fec2d7bef7f04d38b75b`。
如果使用不同版本的 URDF，用该版本重新训练并通过 `--policy` 指定导出的权重。

```bash
pip install -e '.[luna]'
export POSTMAN_LUNA_URDF=/path/to/marvin_description/urdf/Gento_Luna_v3_680/Gento_Luna_v3_680.urdf
```

训练使用交付 URDF 的关节动力学、FX 位置伺服近似和 RDK 车体系速度响应近似。
随机化初始姿态、伺服增益、阻尼和一拍动作延迟。PPO 预测有界残差，WBC 提供笛卡尔
任务参考。部署使用与训练一致的观测和动作解码，可用 NumPy 推理。

```bash
pip install -e '.[luna-train]'
OPENBLAS_NUM_THREADS=1 postman-luna-train --seed 680 --iterations 520 \
  --num-envs 384 --workers 12 --output outputs/luna_680_policy.npz
OPENBLAS_NUM_THREADS=1 postman-luna-train --evaluate-only --seed 1680 \
  --output outputs/luna_680_policy.npz
```

随包权重在 seed 680 上训练了 6,389,760 步。用独立 seed 1680 的 128 个仿真任务评估，
后 41 帧双手平均位置误差从 WBC 的 30.2 mm 降至 PPO/WBC 的 16.5 mm，旋转误差从
0.0694 rad 降至 0.0532 rad。训练的底盘响应是速度层近似，真机伺服和停止距离需要单独验收。
第二组 128 个任务（seed 2680）的平均位置误差从 33.9 mm 降至 17.7 mm。

## 本地原生 DDS 验证

使用包含 `gento_sdk_dds_ui_bridge` 和 `gento_sdk_dds_sim_node` 的 Gento SDK 构建。
SDK build 可参考 `gento-luna-runtime/gento_sdk/docs/native-dds.md`。回归测试会启动隔离的
loopback DDS 域，通过浏览器接口送入手部、头部和底盘目标，检查仿真关节变化、底盘速度
和输入断开后的停车。

硬件网关回归使用同一 Gento SDK 构建中的 `gento_sdk_dds_hardware_node` 和
`libgento_sdk_fx_test_stub.so`。该用例通过 loopback RDK DDS 和 FX ABI 测试件运行 `live`，
检查真实网关的 Position 模式确认、命令回执、实测字段回传和停车。两个用例都只使用本机流量。

```bash
pip install -e '.[luna,dev]'
POSTMAN_GENTO_TEST_BUILD=/path/to/gento-sdk-build python -m pytest -q tests/test_luna_680.py
```

交互运行时可用 `postman-luna-teleop make-sdk-model --output outputs/luna_sdk_scene.xml`
生成 SDK 所需的 MJCF。用相同的 loopback Cyclone DDS 环境分别启动以下进程。

```bash
GENTO_DDS_DOMAIN=225 "$GENTO_BIN/gento_sdk_dds_sim_node" \
  --model luna --scope luna_sim --mjcf outputs/luna_sdk_scene.xml
postman-luna-teleop sim --bridge "$GENTO_BIN/gento_sdk_dds_ui_bridge"
```

在有图形桌面的机器上另开一个终端，使用相同的 DDS 环境打开 MuJoCo 3D 查看器。
查看器订阅仿真的关节和底盘位姿，遥操面板位于 `http://127.0.0.1:8768`。

```bash
postman-luna-teleop view --bridge "$GENTO_BIN/gento_sdk_dds_ui_bridge"
```

`scripts/luna_mujoco_actions.py` 会通过隔离的本地 DDS 仿真录制双臂高抬外展、单臂外展和
底盘绕弧三组动作。绕弧时根据底盘里程计，将固定的地面坐标目标转换到 `base_link`，供 PPO/WBC
做腰臂补偿。彩色圆点标出手部目标，底部显示仿真实测关节和底盘变化。一次录制中，底盘绕弧
位移约 37 cm、转向约 42°。录制需要 Pillow、`imageio-ffmpeg` 和中文字体。

```bash
MUJOCO_GL=egl python scripts/luna_mujoco_actions.py --sdk-build "$GENTO_BIN"
```

## Thor 真机

当前验证覆盖仿真和隔离的原生硬件网关回归。实体 Luna 680 的动作验收待完成。

在 Thor 上运行 POSTMAN，桥接程序连接本机 DDS `226`。使用当前控制服务已部署的校准 JSON、
Cyclone DDS XML 和环境文件，跟随 `gento-luna-runtime/gento_sdk/docs/luna-control-runtime.md`
检查 FX/RDK 反馈与控制所有权。观察服务运行时可用 `probe` 或 `shadow`，控制服务运行时可用
`live`。`shadow` 计算并显示目标。`live` 发送 FX `prepare`、完整 `joint` 目标和 RDK 速度，
按每次 SDK 回执推进。

在当前控制阶段目录已知时，在 Thor 上执行以下命令。`GENTO_STAGE` 是实际部署目录，
`GENTO_BIN` 指向包含 DDS bridge 的已验证 SDK 构建。

```bash
set -a
. "$(cat "$GENTO_STAGE/control-env.path")"
set +a
export CYCLONEDDS_URI="file://$GENTO_STAGE/dds-luna-control.xml"
GENTO_LIMITS_JSON=$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["limits"])' \
  "$GENTO_STAGE/control-profile.json")

postman-luna-teleop probe --scope luna_control --limits "$GENTO_LIMITS_JSON" \
  --bridge "$GENTO_BIN/gento_sdk_dds_ui_bridge"
postman-luna-teleop shadow --scope luna_control --limits "$GENTO_LIMITS_JSON" \
  --bridge "$GENTO_BIN/gento_sdk_dds_ui_bridge"
postman-luna-teleop live --limits "$GENTO_LIMITS_JSON" \
  --bridge "$GENTO_BIN/gento_sdk_dds_ui_bridge"
```

`probe` 的 JSON 包含实测 FX 角度、限位检查、底盘速度与配置上限、DDS 命令端点，
以及收到时的 RDK 模式。若报告有越限或底盘上限为零，先复核当前控制器状态和已部署的校准文件。

`shadow` 结束后再用同一端口启动 `live`。如只联调选定上身分组，可用
`--groups left_arm head`。底盘仍独立受校准上限和浏览器输入控制。

在工作站转发网页，然后打开 `http://127.0.0.1:8768`。

```bash
ssh -N -L 127.0.0.1:8768:127.0.0.1:8768 gento-thor
```

点击「以当前姿态为零点」捕获实测锚点。输入目标后按住「运行」或空格。松开时发送底盘停止
和实测关节保持。浏览器输入超过 240 ms 会失效，须释放再按住。FX 故障、反馈中断或命令拒绝
会退出运动并尝试通过 SDK 停止已准备的分组。底盘失去速度刷新时由 Gateway 的 watchdog
兜底。凸包检查覆盖短步自碰撞预测，使用无夹爪机身模型。
