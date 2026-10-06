"""ai大模型总入口"""

from langchain.agents import create_agent   #从 LangChain 导入创建 Agent 的函数。
# 以前我们用：llm.bind_tools(ai_tools)是让 LLM具备调用 Tool 的能力。
# 现在：create_agent(...)是直接创建一个完整的 Agent。

from ai.client import llm       #导入模型
from ai.tools import ai_tools   #导入 供agent可使用的Tools


### 1、 创建 Agent
# 创建一个能够“思考 → 调用Tool → 获取结果 → 再思考 → 再调用Tool → 最终回答”的执行器。
# Agent 本质上就是把“LLM + Tools + 自动循环”组合起来，让 LLM 自己决定下一步该做什么
agent = create_agent(
    model=llm,              #agent可使用的模型（用谁思考）
    # model：负责思考和决定下一步做什么； tools：Agent可以调用的工具
    tools=ai_tools,         #agent可使用的工具（能用什么）
    system_prompt="""你是一个Todo任务管理助手。    
    你的任务是帮助用户查询、新增、修改和删除Todo任务。
    当用户明确要求执行Todo操作时，应优先选择能够直接完成该操作的Tool。
    如果目标Todo不明确，可以先使用查询Tool确认目标。"""
    )                        #给Agent加上系统提示词（Agent整体应该遵守什么规则）
"""
"system_prompt="相当于以前我们手动写的：
SystemMessage(content="你是一个Todo助手，负责帮助用户管理Todo任务。")
区别只是：
    以前：SystemMessage → 手动放进 messages → llm.invoke()
    现在：system_prompt → create_agent() → Agent内部自动管理
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
创建一个 Agent，并告诉它“你使用这个大模型，并且可以使用这些工具”。
    Agent
     ├── 大脑：llm
     └── 工具箱：ai_tools
           ├── 查询标题
           ├── 查询全部
           ├── 查询状态
           ├── 查询截止时间
           └── 高级查询
"""

# system_prompt → Agent的长期规则 ； message → 本次用户请求（下面）

### 2、AI todo总入口
def todo_by_AI(message):
    """
    AI Todo 总入口。
    用户输入自然语言后：
        用户问题
            ↓
        Agent
            ↓
        Agent自己决定是否调用Tool
            ↓
        Tool执行数据库操作
            ↓
        Agent根据Tool结果继续判断
            ↓
        最终回答
    """
    result = agent.invoke({        #调用Agent，并把用户输入作为第一条消息传给Agent
        "messages": [
            {"role": "user",
            "content": message}    #{"role": "user", "content": "..."} → HumanMessage(...)
        ]
    })
    # Agent流程：用户 → LLM → 调用Tool → Tool结果 → LLM → 再调用Tool → Tool结果 → LLM → 最终答案
    # 所以Agent需要保存整个消息过程，这就是："messages": [...]
    """
    普通LLM调用：
    SystemMessage + HumanMessage
            ↓
    llm.invoke(messages)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Agent调用：
    system_prompt 放在→ create_agent()
    user消息    放在→ agent.invoke()
    """
    """
    print(result)输出的核心结构如下：（大量细节先去掉）
        result
           └── messages
               ├──①HumanMessage:
               │                ↓
               │                HumanMessage(content="查询所有未完成的任务")
               │
               ├──②AIMessage:Agent决定调用哪个工具以及参数
               │                ↓
               │                AIMessage(content='',
               │                       tool_calls=[
               │                           {'name': 'search_todos_by_status',
               │                           'args': {'status': 'pending',
               │                                   'page': 1,
               │                                   'page_size': 50}}
               │                               ])   
               │
               ├──③ToolMessage：Tool 执行完成，把结果返回给 Agent。
               │                     ↓
               │                ToolMessage(
               │                           content="{'data': [...], 'total': 13, ...}",
               │                           name='search_todos_by_status',
               │                           tool_call_id='...' )
               │
               └── ④ AIMessage：Agent 最终给用户的回答
                                    ↓
                                AIMessage(content='以下是所有未完成的任务（共 **13** 条）：...' )
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    用户 → HumanMessage → AIMessage（决定调用Tool）→  ToolMessage（Tool返回结果）→ AIMessage（最终回答）
    """

    # Agent执行过程中会产生多条消息，最后一条消息就是最终AI回答
    final_message = result["messages"][-1]  #①拿到完整消息历史：②取最后一条：AIMessage也就是最终回答
    # 即从Agent完整执行结果中，只拿最后一条AI消息的文本内容
    """
    result里面有很多消息，可以理解成：Agent这一次任务执行完成后留下的完整结果。
    例如：
        第1条：HumanMessage
        第2条：AIMessage → Tool1
        第3条：ToolMessage → Tool1结果
        第4条：AIMessage → Tool2
        第5条：ToolMessage → Tool2结果 
           ……(因为 Agent 可能不是只调用一次 Tool)
        第n条：AIMessage → 最终答案
        
    所以：result["messages"][-1]意思就是：取messages列表中的最后一条消息。
    最后一条就是Agent的最终回答。
    """

    # 返回最终回答文本
    return final_message.content
    # final_message是一个AIMessage，.content取出AIMessage中的最终回答文本

"""
流程：
    用户输入
       ↓
    todo_by_AI(message)
       ↓
    agent.invoke()
       ↓
    Agent思考
       ↓
    选择Tool（以及确定tools的执行顺序）
       ↓
    执行Tool（Prompt + Parser → 辅助Tool内部的AI业务处理）
       ↓
    得到Tool结果
       ↓
    Agent继续思考
       ↓
    最终AIMessage
       ↓
    final_message.content
       ↓
    返回给用户
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Agent → 负责决定“调用哪个Tool、什么时候调用、是否继续调用”
而：Prompt + Parser → 辅助Tool内部的AI业务处理

⭐⭐⭐Agent负责“调度”，Tool负责“执行”，Prompt + Parser负责帮助Tool完成需要AI理解的具体业务。
"""