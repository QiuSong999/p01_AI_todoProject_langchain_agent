"""ai大模型总入口"""

from langchain.agents import create_agent   #从 LangChain 导入创建 Agent 的函数。
# 以前我们用：llm.bind_tools(ai_tools)是让 LLM具备调用 Tool 的能力。
# 现在：create_agent(...)是直接创建一个完整的 Agent。

from ai.client import llm       #导入模型
from ai.tools import ai_tools   #导入 供agent可使用的Tools

#⭐ V3 Chain：程序员自己设计执行流程，Chain 按我们写好的流程运行。
#⭐ V4 Agent：程序员提供工具和规则，由 Agent 根据用户需求决定调用哪个工具、传什么参数，以及是否需要继续调用工具。
"""
可以去掉的内容
--------------------------------------------------------------------------------------------------
之前版本中的内容	                              该版本是否需要	原因
searchtodo_by_AI() 统一意图判断函数	            可以去掉	    Agent 负责判断用户意图
if/elif 判断用户要查询、新增、修改还是删除	        可以去掉	    Agent 根据工具描述选择 Tool
手动判断查询条件并选择查询函数	                    可以去掉	    Agent 直接选择 5 个查询工具之一
ask_deepseek_tool() 这种手动请求模型选择工具的函数	可以去掉	    LangChain Agent 负责模型与工具的交互
parse_ai_search_arguments() 手动解析工具参数的函数	可以去掉	    LangChain Tool Calling 已经提供结构化参数
execute_search_tool() 手动分发工具的函数	        可以去掉	    Agent 负责执行选中的工具
format_search_result() 公共查询结果处理函数	        不需要	    每个查询 Tool 内部直接整理并返回结果
--------------------------------------------------------------------------------------------------
注意：
    1、parse_ai_search_arguments() 不再需要，是因为 V4 中大模型负责根据用户语言提取工具参数，
        LangChain 负责解析和传递工具调用参数，不需要我们再手动使用 json.loads() 转换。
    2、format_search_result()不需要，不是因为Agent代替了，而是写到每一个工具里面了


我们只需要：
    1. 编写好每个 Tool 的业务逻辑。
    2. 把 Tool 注册给 Agent。
    3. 把用户消息交给 Agent。
剩下的工具选择、参数处理、工具分发、工具执行以及多轮工具调用流程，由 Agent 和底层框架协同处理。
注意：大模型负责的是理解用户意图、生成参数；LangChain 负责的是处理工具调用数据。两者不是同一件事。
仍然需要保留的内容
--------------------------------------------------------------------------------------------------
内容	                    为什么仍然需要
@tool	                把 Python 函数包装成 LangChain Tool
ai_tools	            集中管理并绑定给 Agent 的工具列表
addtodo_by_AI()	        封装新增业务流程
updatetodo_by_AI()	    封装修改业务流程
deletetodo_by_AI()	    封装删除业务流程
5 个查询 Tool	        提供不同的查询能力
match_todo_id_by_AI()	修改、删除时，根据自然语言匹配 Todo ID
Prompt	                指导大模型提取参数或匹配 Todo
PydanticOutputParser	解析和校验大模型输出
db.py	                执行 MySQL 数据库操作
--------------------------------------------------------------------------------------------------
以前的版本：                       现在的版本：
      用户输入                            用户输入
          ↓                               ↓       
      Python 判断意图               Agent + 大模型判断意图      
          ↓                               ↓        
      if / elif 选择函数                选择 Tool          
          ↓                               ↓          
      手动解析参数                     Agent 执行 Tool           
          ↓                               ↓     
      执行数据库操作                 Tool 内部完成业务流程
--------------------------------------------------------------------------------------------------
                    
"""

### 1、 创建 Agent
# 创建一个能够“思考 → 调用Tool → 获取结果 → 再思考 → 再调用Tool → 最终回答”的执行器。
# Agent 本质上就是把“LLM + Tools + 自动循环”组合起来，让 LLM 自己决定下一步该做什么
agent = create_agent(   #创建一个可以根据模型决策调用工具的 Agent 执行组件
    model=llm,              #model为agent可使用的模型（用谁思考）
                # model：负责思考和决定下一步做什么； tools：Agent可以调用的工具
    tools=ai_tools,         #tools 是可以供模型选择调用的工具集合（能用什么）
    # ⭐ ai_tools现在有8个工具，增删改查所有操作函数都注册成工具

    system_prompt=
    """
    你是一个Todo任务管理助手。    
    你的任务是帮助用户查询、新增、修改和删除Todo任务。
    当用户明确要求执行Todo操作时，应优先选择能够直接完成该操作的Tool。
    如果目标Todo不明确，可以先使用查询Tool确认目标。
    """
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
     ├── 大脑：model
     ├── 系统提示词：system_prompt
     └── 工具箱：tools
           ├── 查询标题
           ├── 查询全部
           ├── 查询状态
           ├── 查询截止时间
           ├── 高级查询
           ├── 增删改3个
"""
# system_prompt → Agent的长期规则 ； message → 本次用户请求（下面）

### 2、调用agent (AI todo总入口)
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
        result（dict）
           └── messages（list）
               ├──①HumanMessage:用户输入
               │                ↓
               │                HumanMessage(content="查询所有未完成的任务")
               │
               ├──②AIMessage:Agent决定调用哪个工具以及参数
               │                ↓
               │                AIMessage(
               │                       content='',
               │                       tool_calls=[ 
               │                           {'name': 'search_todos_by_status',
               │                           'args': {'status': 'pending','page': 1,'page_size': 50}
               │                           'id':'call_xxx',       
               │                           "type": "tool_call"}
               │                                    ],   #当只有一个工具时，tool_calls列表里面就一个大字典元素  
               │                        additional_kwargs={……},
               │                        response_metadata={……},
               │                        usage_metadata={……},
               │                        ……          
               │                            )   
               │                                 
               │
               ├──③ToolMessage：Tool 执行完成，把结果返回给 Agent。
               │                     ↓
               │                ToolMessage(
               │                           content="{'data': [...], 'total': 13, ...}",
               │                           name='search_todos_by_status',
               │                           tool_call_id='...' )
               │
               │                    ↕ (关键!)
               │
               │                 【Agent 根据工具返回结果继续判断】 
               │                         ├── 需要继续调用工具 → 重复②和③ ⤴
               │                         └── 不需要继续调用工具 → 进入④
               │                                ↓
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
"""
ai/agent.py
Agent 根据用户需求决定调用什么工具
    ↓
ai/tools.py
集中注册工具，并将工具提供给 Agent
    ↓
ai/todo.py
工具的具体实现：解析参数、匹配目标、调用业务函数
    ↓
db.py → MySQL
执行实际的数据查询、新增、修改和删除
"""

#agent能做的事情：
"""
结合本版本的项目的代码，Agent 的 6 个作用分别通过以下机制实现。
     作用                     如何实现        
     1. 理解用户意图        大模型根据用户输入和上下文理解需求。     
     2. 选择合适的工具      create_agent(model=llm, tools=ai_tools)将模型和工具集合交给 Agent，由模型决定调用哪个工具。     
     3. 提取工具参数        模型根据工具的名称、描述、参数定义，从用户语句中生成调用参数。     
     4. 调用工具并获取结果   LangChain Agent 执行模型提出的工具调用，运行对应的 Python 函数，获得 ToolMessage。     
     5. 根据结果继续决策     Agent 将工具执行结果交给模型，模型判断是否还需要调用其他工具；需要时继续调用。     
     6. 生成最终回答        模型根据用户请求和工具执行结果生成最终回答，代码通过 result["messages"][-1].content 提取文本。     
"""
