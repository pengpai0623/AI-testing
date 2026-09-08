# llm-chat-api

基于 **FastAPI + llmsdk + Redis** 的大模型对话后端服务，提供单轮/多轮对话与 SSE 打字机流式输出。

✅ 单轮问答 ｜ ✅ 多轮会话 ｜ ✅ SSE 流式输出 ｜ ✅ Redis 会话持久化 ｜ ✅ WATCH 乐观锁防会话竞态 ｜ ✅ Redis 滑动窗口限流 ｜ ✅ 全局异常统一返回 ｜ ✅ E2E 自动化测试

---

## 一、功能介绍

| 能力 | 说明 |
|---|---|
| 单轮问答 | `/chat/single`，无会话上下文，直接返回完整结果 |
| 多轮非流式（主） | `/chat/async_session`，底层异步 httpx，原生协程不阻塞事件循环 |
| 多轮非流式（兜底） | `/chat/session`，底层同步 requests，通过 `asyncio.to_thread` 包装 |
| SSE 流式（主） | `/chat/session_stream_httpx`，全异步 httpx，打字机分片输出 |
| SSE 流式（兜底） | `/chat/session_stream_requests`，同步 requests + `asyncio.to_thread` |
| 健康检查 | `/chat/health` |

附加能力：

- **Redis 滑动窗口分布式限流**（按客户端 IP），Redis 故障自动降级放行，不阻断业务
- **两层参数校验**：Pydantic 入参字段长度校验 + Service 层业务字符长度校验
- **统一全局异常处理器**，HTTP 状态码恒为 200，业务错误统一通过响应体 `code` 区分
- **会话跨接口互通**：非流式接口写入的会话，流式接口可直接读取历史上下文
- **WATCH 乐观锁**写入会话，避免同一 `session_id` 并发请求互相覆盖
- **Docker / docker-compose** 一键本地部署，多阶段构建、非 root 运行
- **pytest E2E** 接口自动化测试（`asyncio_mode = auto`）

---

## 二、架构简图

```
┌─────────────────────────────────────────────────────────────┐
│            Client（浏览器 / Postman / pytest）              │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ FastAPI 入口  api_server/main.py                            │
│  • RequestLogMiddleware 请求日志/耗时                        │
│  • 全局异常处理器：                                          │
│    - 422 (RequestValidationError)                            │
│    - HTTPException（类注册，覆盖 429 等）                   │
│    - 404 / 405（按 status_code 单独注册）                    │
│    - LLMBaseError 业务异常                                  │
│    - Exception 兜底 500                                      │
│  • 路由挂载：/chat，依赖 rate_limiter_dep                   │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Router 层  api_server/routers/chat_router.py                │
│  • Pydantic 解析请求体                                       │
│  • SSE 接口：在生成器【外部】await build_chat_prepare_messages│
│  ⚠️ 预处理阶段异常 → 全局异常处理器 → 返回 application/json  │
│  • 返回 ApiResponse / EventSourceResponse                   │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Service 层  api_server/service/chat_service.py               │
│  • _validate_input_length 业务字符长度校验                    │
│  • build_chat_prepare_messages：读 Redis→拼消息→校验→截断   │
│  • 封装 LLMBaseClient（同步/异步/流式）                       │
│  • SSE 异步生成器：分片 yield，内部异常转 error 事件          │
│  • safe_set_session_with_watch 乐观锁写 Redis                │
└───────────┬──────────────────────────────┬──────────────────┘
            ↓                            ↓
┌────────────────────┐      ┌──────────────────────────────┐
│ llmsdk SDK 层      │      │ api_server/db/chat_db.py     │
│ • LLMBaseClient    │      │ • get_session                │
│ • MessageItem      │      │ • safe_set_session_with_watch│
│ • ChatSession 截断 │      │ • delete_session             │
│ • 异常/常量/日志   │      └──────────────┬───────────────┘
└────────────────────┘                     ↓
                                    ┌────────────────┐
                                    │ Redis          │
                                    │ key 前缀 llm:session: │
                                    └────────────────┘
```

### 异常流转约定（重要）

1. **Router 预处理阶段（生成器迭代之前）抛出异常**
   Pydantic 校验失败、prompt 超长、Redis 读会话失败、token 超限、限流命中 → 全局异常处理器捕获 → 返回 `Content-Type: application/json`，**不会进入 SSE 流**。
2. **路由匹配阶段抛出异常（404 / 405）**
   路径不存在 / 请求方法不允许 → 按 `status_code` 注册的 handler 捕获 → 返回统一 JSON，`code=40001`，原始 HTTP 状态码放在 `data.http_status`。
   > 404/405 单独按 status_code 注册，是因为它们在路由匹配阶段抛出，走 HTTPException 类注册的 handler 不会被命中。
3. **SSE 生成器迭代过程中抛出异常**
   已进入 `text/event-stream` 通道，生成器内部 `try/except` 捕获并通过 `event: error` 事件返回，保持流式通道正常结束。

---

## 三、项目目录

```
ai_test/
├── api_server/
│   ├── main.py                    # FastAPI 入口、中间件、全局异常处理器、路由挂载
│   ├── routers/
│   │   └── chat_router.py        # 路由层：入参解析、调用 service、组装响应
│   ├── service/
│   │   └── chat_service.py        # 业务逻辑：预处理、LLM 调用、SSE 生成器、会话写入
│   ├── common/
│   │   ├── limiter.py             # Redis 滑动窗口限流依赖 + get_client_ip
│   │   ├── param_validator.py     # prompt / system_prompt 字符长度校验
│   │   └── response.py            # 统一返回体 ApiResponse
│   ├── models/
│   │   └── chat_models.py         # 请求/响应 Pydantic 模型
│   └── db/
│       └── chat_db.py            # Redis 操作封装
├── llmsdk/                        # 底层 SDK（不依赖 api_server）
│   ├── client/
│   │   ├── base_llm.py           # LLMBaseClient：同步/异步/流式调用
│   │   └── llm_struct_client.py  # 结构化输出客户端
│   ├── common/
│   │   └── schemas.py            # MessageItem（role/content）
│   ├── session/
│   │   └── chat_session.py       # ChatSession：按 token 截断历史消息
│   ├── prompt_repo/              # 版本化 Prompt 模板库（v1/v2_CoT）
│   ├── config/
│   │   └── settings.py           # .env 加载与必填配置校验
│   └── utils/
│       ├── constants.py          # 全部错误码、限流/token/长度常量
│       ├── exceptions.py         # LLMBaseError 及全部业务异常
│       ├── session_redis.py      # SessionRedis：get/set/watch 写入
│       └── logger.py             # loguru 日志
├── tests/
│   ├── conftest.py               # base_url / client / session_id fixture
│   └── test_chat_api.py          # E2E 全接口自动化测试
├── main.py                       # 独立的 asyncio 学习示例（非服务入口）
├── Dockerfile                    # 多阶段构建，非 root 运行
├── docker-compose.yml            # chat-api 编排，连接宿主机 Redis
├── requirements.in / requirements.txt / requirements_dev.txt
├── pytest.ini
├── .env.example（实际为 .env，已 gitignore）
└── .gitignore
```

---

## 四、环境要求

- Python **3.11**（Docker 镜像基于 `python:3.11-slim`）
- Redis 5+（需支持 `ZADD/ZCARD/ZREMRANGEBYSCORE/WATCH`）
- 大模型：火山方舟（豆包）兼容 OpenAI Chat Completions 接口

---

## 五、环境变量

项目通过 `.env` 加载（`llmsdk/config/settings.py` 启动时自动 `load_dotenv()` 并校验必填项）。

| 环境变量 | 说明 | 必填 |
|---|---|---|
| `DOUBAO_API_KEY` | 大模型 API Key | ✅ |
| `DOUBAO_ENDPOINT` | Chat Completions 接口地址，如 `https://ark.cn-beijing.volces.com/api/v3/chat/completions` | ✅ |
| `DOUBAO_MODEL` | 模型名，如 `doubao-seed-2-0-lite-260428` | ✅ |
| `REQUEST_TIMEOUT` | LLM 请求超时（秒），默认 `60` | ❌ |
| `REDIS_URL` | Redis 连接串，本地默认 `redis://127.0.0.1:6379/0`；Docker 内指向 `redis://host.docker.internal:6379/0` | ✅ |

`.env` 示例（**不要提交真实密钥**）：

```env
DOUBAO_API_KEY=your-api-key
DOUBAO_ENDPOINT=https://ark.cn-beijing.volces.com/api/v3/chat/completions
DOUBAO_MODEL=your-model-name
REQUEST_TIMEOUT=60
REDIS_URL=redis://127.0.0.1:6379/0
```

> 限流窗口大小（`MAX_REQUEST_PER_WINDOW=20`、`WINDOW_SECONDS=60`）、会话 TTL（`SESSION_TTL_SEC=86400`，即 1 天）、token 上限（`DEFAULT_MAX_TOKEN=6000`、`MAX_COMPLETION_TOKEN=1000`）等固化在 `llmsdk/utils/constants.py`。

---

## 六、启动命令

### 方式 1：本地虚拟环境运行

```powershell
# 创建并激活虚拟环境（PowerShell）
python -m venv ai-env
.\ai-env\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt

# 启动开发服务（自动 reload）
uvicorn api_server.main:app --host 0.0.0.0 --port 8000 --reload
```

> 也可直接 `python -m api_server.main`，入口文件内置了 `uvicorn.run(..., reload=True)`。

启动后访问：

- Swagger UI：<http://127.0.0.1:8000/docs>
- ReDoc：<http://127.0.0.1:8000/redoc>

### 方式 2：Docker Compose（本地开发推荐）

当前 `docker-compose.yml` 中内置 Redis 容器已注释，`chat-api` 通过 `host.docker.internal` 连接**宿主机 Redis**：

```bash
docker compose up --build -d          # 构建并后台启动
docker compose logs -f llm-chat-api  # 查看日志
docker compose down                   # 停止
```

如需使用 compose 内置 Redis，取消 `redis:` 服务块注释、打开 `depends_on`，并把 `REDIS_URL` 改为 `redis://redis:6379/0`。

### 方式 3：裸 Docker

```bash
docker build -t llm-chat-api:v1 .

docker run -d \
  -p 8000:8000 \
  -e DOUBAO_API_KEY=your-key \
  -e DOUBAO_ENDPOINT=https://ark.cn-beijing.volces.com/api/v3/chat/completions \
  -e DOUBAO_MODEL=your-model \
  -e REDIS_URL=redis://host.docker.internal:6379/0 \
  --name llm-chat-api \
  llm-chat-api:v1
```

---

## 七、统一响应格式

所有接口（含错误）HTTP 状态码恒为 **200**，业务结果通过响应体 `code` 判断：

```json
{
  "code": 0,
  "msg": "ok",
  "data": {}
}
```

### 业务错误码一览

| code | 常量 | 含义 |
|---|---|---|
| `0` | `CODE_OK` | 成功 |
| `422` | `CODE_VALIDATE_ERROR` | Pydantic 请求体校验失败 |
| `500` | `CODE_SERVER_ERROR` | 未捕获的服务器内部错误 |
| `1001` | `ERR_ENV_CONFIG` | 环境变量/配置缺失 |
| `1002` | `ERR_LLM_NETWORK` | LLM 网络超时/连接失败 |
| `1003` | `ERR_LLM_HTTP` | 上游 LLM 返回 4xx/5xx 或 content 为空 |
| `1004` | `ERR_SSE_PARSE` | SSE 分片解析失败 |
| `4001` | `ERR_JSON_PARSE` | JSON 解析失败 |
| `4002` | `ERR_PYDANTIC_VALIDATE` | Pydantic 字段校验失败 |
| `4003` | `ERR_VALUE` | 数值非法 / WATCH 乐观锁冲突 |
| `4005` | `ERR_MSG_VALIDATE` | 消息列表格式校验失败 |
| `4006` | `ERR_CLIENT_DISCONNECT` | 客户端主动断开（正常结束） |
| `4007` | `ERR_PARAM_TOO_LONG` | prompt/system_prompt 字符超长 |
| `5001/5002/5003` | `ERR_REDIS_*` | Redis 连接/超时/操作异常 |
| `40001` | `ERR_HTTP_BAD_HTTP_STATUS` | 404/405 等 HTTP 层异常（原始状态码在 `data.http_status`） |
| `42900` | `ERR_RATE_LIMIT` | 触发限流 |

404/405 响应示例：

```json
{
  "code": 40001,
  "msg": "http请求异常：Not Found",
  "data": {"http_status": 404}
}
```

限流响应示例：

```json
{
  "code": 42900,
  "msg": "请求过于频繁，请稍后重试",
  "data": null
}
```

---

## 八、接口文档

所有接口前缀 `/chat`。

### 8.1 `POST /chat/single` — 单轮问答

无会话上下文，不读写 Redis。

**请求体 `SingleChatRequest`**

| 字段 | 类型 | 必填 | 约束 | 说明 |
|---|---|---|---|---|
| `prompt` | string | ✅ | 长度 1–4000 | 用户提问 |
| `system_prompt` | string | ❌ | 最长 2000 | 系统提示词 |
| `temperature` | float | ❌ | 0–2，默认 0.7 | 随机度 |
| `stream` | bool | ❌ | 默认 false | 预留字段 |

```json
{
  "prompt": "你好，用一句话介绍你自己",
  "system_prompt": "你是一个简洁的助手",
  "temperature": 0.7
}
```

**响应 `data`（`SingleChatResponse`）**

```json
{
  "answer": "你好，我是你的AI助手。",
  "prompt_tokens": 18,
  "completion_tokens": 12,
  "total_tokens": 30
}
```

---

### 8.2 `POST /chat/async_session` — 多轮对话（主，httpx 异步）

通过 `session_id` 在 Redis 维护上下文；`system_prompt` **仅首次会话生效**，后续传入会被忽略并告警。

**请求体 `SessionChatRequest`**（禁止多余字段，`extra="forbid"`）

| 字段 | 类型 | 必填 | 约束 | 说明 |
|---|---|---|---|---|
| `session_id` | string | ✅ | 长度 1–64 | 会话 ID |
| `prompt` | string | ✅ | 长度 1–4000 | 本轮提问 |
| `system_prompt` | string | ❌ | 最长 2000 | 仅首次会话生效 |
| `temperature` | float | ❌ | 0–2，默认 0.7 | 随机度 |

```json
{
  "session_id": "u_10001",
  "prompt": "我叫张三，记住我的名字",
  "system_prompt": "你是一个 helpful 助手"
}
```

**响应 `data`（`SessionChatResponse`）**

```json
{
  "session_id": "u_10001",
  "answer": "好的，张三，我记住了。",
  "history_count": 2,
  "prompt_tokens": 35,
  "completion_tokens": 15,
  "total_tokens": 50
}
```

> `history_count` 为本轮写入后会话总消息条数（首轮 = `user + assistant` 共 2 条；若首轮带 `system_prompt` 则为 3 条）。

---

### 8.3 `POST /chat/session` — 多轮对话（兜底，requests 同步）

入参/出参与 8.2 完全一致，底层为同步 `requests`，通过 `asyncio.to_thread` 调度，建议生产优先使用 `/chat/async_session`。

---

### 8.4 `POST /chat/session_stream_httpx` — SSE 流式（主，httpx 异步）

请求体同 `SessionChatRequest`。返回 `Content-Type: text/event-stream`。

> ⚠️ 若预处理阶段（参数校验、prompt 超长、Redis 读会话失败等）出错，**直接返回 `application/json` 错误体**，不会进入 SSE 流。

**SSE 事件**

| event | data | 说明 |
|---|---|---|
| `message` | 分片文本字符串 | LLM 逐 token 增量输出 |
| `done` | JSON：`{"full_answer": "...", "msg_count": 2}` | 流式正常结束，`full_answer` 为完整回答，`msg_count` 为会话总条数 |
| `error` | JSON：`{"code": 4007, "msg": "..."}` | 流式运行时异常（LLM 错误、Redis 写入失败等） |

事件样例：

```
event: message
data: 你

event: message
data: 好

event: done
data: {"full_answer": "你好", "msg_count": 2}
```

客户端断开连接时服务端静默丢弃本轮会话，不推送 `error` 事件。

---

### 8.5 `POST /chat/session_stream_requests` — SSE 流式（兜底，requests 同步）

事件协议与 8.4 一致，底层为同步 `requests` + `asyncio.to_thread`。

---

### 8.6 `GET /chat/health` — 健康检查

```json
{
  "code": 0,
  "msg": "ok",
  "data": {"status": "ok", "service": "llm-api"}
}
```

---

## 九、两层参数校验说明

| 层级 | 位置 | 限制 | 失败错误码 |
|---|---|---|---|
| Pydantic 入参层 | `chat_models.py` | `prompt ≤ 4000`，`system_prompt ≤ 2000`，`session_id ≤ 64` | `422` |
| Service 业务层 | `chat_service._validate_input_length` | `prompt ≤ 2000`（`PROMPT_MAX_CHARS`），`system_prompt ≤ 1000`（`SYSTEM_PROMPT_MAX_CHARS`） | `4007` |

> 业务层阈值比 Pydantic 更严，因此**单字段在 2001–4000 之间时不会被 Pydantic 拦截，会进入 service 并返回 `4007`**。上下文总 token 由 `ChatSession.trim_messages_for_context` 在 `DEFAULT_MAX_TOKEN=6000` 内自动截断。

---

## 十、限流说明

- 算法：Redis ZSET 滑动窗口，按客户端 IP 计数（key：`rate_limit:{ip}`）
- 配置：窗口 `WINDOW_SECONDS=60` 秒，窗口内最多 `MAX_REQUEST_PER_WINDOW=20` 次
- 实现：`pipeline.multi()` 事务包裹 `ZREMRANGEBYSCORE → ZCARD → ZADD → EXPIRE`，避免竞态
- 命中：抛出 `HTTPException(429)`，经全局异常处理器转为 `code=42900`
- 降级：Redis 异常时记录日志并放行，不阻断业务

---

## 十一、测试

> ⚠️ E2E 用例会真实调用大模型产生 token 费用；运行前需先启动服务与 Redis。

```bash
# 全部用例
pytest tests/test_chat_api.py -v -s

# 单条用例
pytest tests/test_chat_api.py::test_stream_httpx_normal_flow -v -s

# 指定被测服务地址
$env:API_BASE_URL="http://127.0.0.1:8000"; pytest tests/test_chat_api.py -v
```

覆盖场景：

- 单轮问答成功 / 缺少必填字段 422 / prompt 超长
- 多轮首次会话 / 上下文记忆 / 跨接口（非流式→流式）会话互通
- SSE 分片输出、`done` 事件与分片拼接一致性
- Pydantic 入参超长时 SSE 接口直接返回 JSON（不进流）
- Redis 滑动窗口限流触发
- 404 / 405 全局兜底返回统一 JSON

> 限流用例依赖 Redis 状态，重复执行前可清理 key：
>
> ```redis
> del rate_limit:127.0.0.1
> del rate_limit:172.18.0.1
> ```

---

## 十二、生产部署注意事项

1. 所有密钥通过环境变量注入，`.env` 已在 `.gitignore` 中，禁止提交真实 Key。
2. 生产环境关闭 `--reload`，使用多进程 uvicorn/gunicorn worker；会话 Redis 建议配置密码与持久化。
3. 会话写入使用 Redis `WATCH` 乐观锁，同一 `session_id` 并发请求冲突时返回 `4003`，业务侧应对同一会话串行化或做重试。
4. SSE 前端需区分两类错误：HTTP 响应为 JSON（预处理阶段）vs `text/event-stream` 中的 `event: error`（运行时）。
5. 限流为单机 Redis 计数，多实例共享同一 Redis 即可全局生效；Redis 故障时限流自动放行，需结合监控告警。

---
