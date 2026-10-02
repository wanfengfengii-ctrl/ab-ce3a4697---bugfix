# 多目标光谱仪光纤臂分配裁决服务 (Fibre-Arm Assignment Adjudicator)

天文台多目标光谱仪曝光前的机械臂联合分配 API。给定 6–12 根带整数基座坐标与
最大伸长的可伸缩光纤臂、6–16 个唯一目标（正整数优先级）、正整数安全净距与
最低分配数，服务在满足全部硬约束的前提下按字典序优化：

1. **分配数最大化** — 连接的臂数最多；
2. **优先级总和最大化**；
3. **伸长平方和最小化**；
4. **稳定分配序列最小化** — 按臂输入序，取所连目标的 1-based 序号
   （闲置臂使用大于任何目标序号的哨兵值），保证最优解唯一可复现。

## 硬约束

- 每根臂至多连接一个目标；目标不得复用；
- 臂只能连接 `max_extension` 范围内（含边界）的目标；
- 任意两条「基座→目标」**闭线段**之间的最小欧氏距离不得小于 `clearance`
  （距离取闭区间，端点重合/线段相交距离为 0；距离恰好等于净距合法）。

达不到 `minimum_assignments` 时仍返回 **200**，`status` 为 `below_minimum`，
并给出实际最大可达数量 `maximum_attainable` 与原因（区分净距碰撞与可达性
不足）。任何非法输入（数量越界、重复 id/坐标、非正整数、非整数等）一律
**422**，不会进入求解器。

## 目录

| 路径 | 说明 |
| --- | --- |
| `app/geometry.py` | 闭线段最短距离（六区域钳制投影） |
| `app/matching.py` | Kuhn 基数匹配 + 矩形匈牙利算法 |
| `app/solver.py` | 分支限界求解器（含已放置段感知的紧界） |
| `app/main.py` / `app/models.py` | FastAPI 裁决接口、校验与证据构造 |
| `tests/` | 几何、匹配、求解器（对拍穷举）、API 测试 |
| `scripts/smoke.py` | 含碰撞约束的 API 冒烟（独立几何校验） |
| `scripts/healthcheck.py` | 容器/Compose 就绪探针 |

## 运行

```bash
# 宿主机端口可配置：复制并编辑 .env，或直接导出环境变量
cp .env.example .env

docker compose up -d --build api        # 启动 API
docker compose up --build verify        # 一次性验证：测试 + 构建检查 + 冒烟
```

- `HOST_PORT`：宿主机发布端口（默认 8000），`API_PORT`：容器内监听端口。
- `api` 的健康检查只在启动自检通过、裁决接口可接收请求后才转为 healthy。
- `verify` 通过 `depends_on: condition: service_healthy` 等待健康状态，
  依次执行 pytest、应用导入构建检查、碰撞约束冒烟，并以退出码汇总成败
  （`docker compose up verify` 看到 `verify exited with code 0` 即通过）。

本地（无 Docker）：

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest -q
uvicorn app.main:app --host 0.0.0.0 --port 8000
python scripts/smoke.py   # 另开终端
```

## 接口

`POST /api/v1/assignment/adjudicate`

```json
{
  "arms": [
    {"id": "A1", "x": 0, "y": 0, "max_extension": 10}
  ],
  "targets": [
    {"id": "T1", "x": 0, "y": 5, "priority": 5}
  ],
  "clearance": 2,
  "minimum_assignments": 4
}
```

`200` 响应包含：

- `assignments`：配对（臂 id、目标 id、伸长长度与平方）；
- `unassigned`：未分配的臂与目标，各自附带原因（够不到 / 被净距或唯一性挤出）；
- `arm_lengths`：每根臂（含闲置臂）的长度证据；
- `clearance_evidence`：要求净距、全部已放线段两两最小距离、最近 5 对的
  最近点坐标证据、是否满足、检查对数（不足两对时最小距离为 `null`）；
- `objectives`：分配数、优先级总和、伸长平方和、稳定序列（0 表示闲置）；
- `status` / `reason` / `maximum_attainable` / `minimum_assignments`。

健康探针：`GET /healthz`（存活）与 `GET /healthz/ready`
（就绪，启动自检完成前返回 503）。

## 求解方法

按臂输入序深度优先分支限界。每个节点上的松弛界忽略「未来线段之间」的碰撞，
只考虑可达性与目标唯一性，因而对真实最优是乐观的：

- 基数上界：Kuhn 增广路二分图匹配，且会删除与已放置线段冲突的边，得到更紧的界；
- 优先级上界：每行使能成本为 0 的虚拟列（闲置）、真实边成本为
  `-priority` 的矩形匈牙利算法；
- 伸长平方下界：真实边成本 `d² - B`（`B` 大于任意匹配总平方长，强制匈牙利
  先最大化基数再最小化平方和），虚拟列成本 0；
- 稳定序列：按目标序号升序潜水，并以前缀字典序剪枝。

松弛界按 `(深度, 已用目标掩码)` 缓存；线段距离惰性缓存。12 臂 × 16 目标的
稠密高碰撞实例在普通硬件上数秒内完成；`tests/test_solver.py` 用独立穷举器
对 40 个随机中小实例逐一对拍最优解。
