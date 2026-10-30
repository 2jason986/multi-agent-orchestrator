# orchestra 中文说明

一个只依赖 Python 标准库的多智能体编排框架，用客服工单作为落地场景。

四个角色分工合作：规划、检索、回复、审核。每条工单走完流程都会留下一份
可读的执行轨迹，回复在审核通过之前不会发出去。

[English README](README.md)

```bash
git clone https://github.com/2jason986/multi-agent-orchestrator.git
cd multi-agent-orchestrator
python -m orchestra.cli demo
```

不需要 API key，不需要联网，也不需要装任何第三方包。

## 它解决什么问题

单靠一个提示词"帮我回复工单"只能做演示。真正能交给客户的流程需要三样东西：

一是事实来源。回复里的每句话都必须能追溯到知识库条目，避免模型编造政策、价格和时效。

二是否决机制。写回复的智能体和审核回复的智能体必须是两个独立上下文，自己审自己几乎总是通过。

三是兜底。审核反复不通过时不能无限循环，到达修订上限就转人工，并且把原因写进轨迹。

这个仓库把这三件事做成了默认行为，而不是可选项。

## 四个角色

| 角色 | 职责 | 输出约定 |
|---|---|---|
| planner | 判断优先级，生成检索词，给出处理计划 | `{"priority", "queries", "plan", "subject"}` |
| researcher | 调用 `kb_search` 工具，拿到证据后用带引用的要点总结 | `{"summary", "citations"}` |
| responder | 只用检索结果写客户回复 | `{"draft", "citations"}` |
| reviewer | 审核引用是否完整、是否有编造内容、是否给了一个明确的下一步 | `{"verdict", "feedback"}` |

智能体之间不直接通信，数据流由 orchestrator 掌管，所以控制流程是线性的，能直接写断言。

## 运行效果

处理仓库自带的三条工单：

```
provider      : mock
tickets       : 3
approved      : 3
needs a human : 0
review rounds : 6
```

每条工单都经历一次驳回和一次修订，第二轮审核通过。轨迹里的 `verdict=revise`
到 `verdict=approve` 就是这个项目想展示的东西。

## 接真实模型

默认用 `mock`，它是脚本化的确定性客户端，让整条链路可以离线跑。接真实模型只需要三个环境变量：

```bash
export ORCHESTRA_API_KEY=sk-...
export ORCHESTRA_BASE_URL=https://api.deepseek.com/v1
export ORCHESTRA_MODEL=deepseek-chat
python -m orchestra.cli --provider openai demo
```

任何兼容 `/chat/completions` 的服务都能用：OpenAI、DeepSeek、Moonshot、通义、vLLM、Ollama。
没有安装厂商 SDK，HTTP 请求用 `urllib` 直接发。

## 目录结构

```
src/orchestra/
  llm.py            模型客户端协议与 OpenAI 兼容实现
  mock_llm.py       确定性脚本模型，离线演示和测试用
  agents.py         角色提示词、JSON 契约、工具循环
  orchestrator.py   规划 → 检索 → 回复 → 审核的编排
  knowledge.py      TF-IDF 检索，中文按字二元组切分
  builtin_tools.py  kb_search 与 ticket_lookup 两个工具
  tickets.py        工单模型与内存存储
  tools.py          工具注册表与统一错误处理
  trace.py          执行轨迹，输出 Markdown 和 JSON
  cli.py            命令行入口
data/
  knowledge_base/   六篇知识库文档
  tickets.json      三条示例工单
tests/              34 个单元测试与端到端测试
```

## 测试

```bash
pip install -e ".[dev]"
python -m unittest discover -s tests -t . -v
ruff check .
```

全部测试都跑在 mock 模型上，不联网、不花钱、结果稳定。CI 在 Python 3.10、3.11、3.12 上各跑一遍。

## 设计取舍

用严格的 JSON 契约而不是自由文本，是为了让编排层能直接依据 `verdict` 做路由，代价是模型偶尔要重试一次。

审核单独成角色而不是在同一个上下文里自我检查，是因为模型对自己刚写的内容几乎没有判断力，独立上下文的否决权才能真正挡住编造。

修订轮次有上限，默认两轮后转人工。无限自我修正既烧钱又不收敛。

检索用自己实现的 TF-IDF 而不是向量库，是为了保证 clone 下来十秒内能跑起来。真实部署时这块要换成 embedding 检索，接口只需要替换 `KnowledgeBase.search`。

mock 模型是一等公民而不是测试替身，它让测试断言编排行为，而不是断言模型今天心情好不好。

## 许可证

MIT，见 [LICENSE](LICENSE)。
