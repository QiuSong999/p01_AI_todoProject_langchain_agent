"""
    tools.py
        ↓
    告诉 Agent：
    “你现在拥有这些工具”
★★★Agent负责确定 Tool 的执行顺序，Prompt 和 Parser 辅助 Tool 工作
"""

from langchain_core.tools import tool   #让我们能够使用 LangChain 的 @tool

from ai.todo import addtodo_by_AI , updatetodo_by_AI, deletetodo_by_AI
"""
从LangChain工具模块导入tool装饰器。

@tool作用：
1. 把普通Python函数转换成LangChain可以识别的AI工具。
2. 自动读取函数名称作为工具name。
3. 自动读取函数注释(docstring)作为工具description。
4. 自动根据函数参数生成parameters参数结构。

例如：
@tool
def search_todos_by_title(title,page,page_size):
#
LangChain会自动转换成AI可以调用的工具：
{
    name: "search_todos_by_title",
    description: "根据函数注释生成",
    parameters: {
        title,
        page,
        page_size
    }
}

后续通过：
tools=ai_tools[工具列表]
将这些工具提供给agent，让agent决定调用哪个工具。
"""

"""
⭐⭐⭐
1、为什么不直接把 db.py 的函数全部注册成 Tool供agent调用操作？
    现在项目中有两种函数，
    一种是db文件中直接操作数据库的函数
        def delete_todo(todo_id: int):
            # 根据任务 ID 删除数据库中的记录
    另一种是ai/todo文件中的AI 业务工具函数
        @tool
        def deletetodo_by_AI(message: str):
            todo_id = match_todo_id_by_AI(message)  # 根据用户的自然语言描述，找到目标任务
            return delete_todo(todo_id)     # 调用数据库函数，执行删除
            #★这个工具接收用户的自然语言，负责找到目标任务，再调用数据库函数
    如果直接把数据库函数注册给 Agent，会发生什么？
    假如用户输出删除买牛奶的事项，Agent 看到这个工具需要：delete_todo(todo_id: int)
    但是用户提供的是“买牛奶”，而不是数字 ID。
    此时，Agent 必须想办法确定 todo_id。它可能从已有上下文获取 ID，也可能尝试查询，
    但如果没有对应的查询工具和正确的调用流程，就无法可靠地完成删除。
    
    当前 V4 agent的设计如何解决？
        用户输入
        “删除买牛奶”
            ↓
        Agent
        选择 deletetodo_by_AI 工具
            ↓
        ai/todo.py
        调用 match_todo_id_by_AI()，根据描述匹配目标 ID
            ↓
        db.py
        调用 delete_todo(todo_id) 执行数据库删除
    这样，Agent 不需要直接处理底层数据库函数的细节。
    你把自然语言处理和业务流程封装在一个工具中，Agent 只需要知道这个工具的用途和参数。
2、查询工具为什么可以直接注册？
    查询工具本身可能已经是适合 Agent 使用的函数。
    例如：
        @tool
        def search_todos_by_status(status: str, page: int = 1, page_size: int = 50):
            '''根据任务状态查询待办事项。'''
        ...

    它接收 status、page 等参数，能够直接完成一项查询需求，因此可以注册给 Agent。
    不过，这个 Tool 内部仍然会调用数据库函数获取数据。
    所以，你当前 V4 中同时存在两种情况：
    查询工具：Agent 可以直接调用已经封装好的查询 Tool。
    新增、修改、删除工具：先通过 AI 解析用户描述或匹配目标，再调用数据库函数。
    两种设计并不矛盾，关键是 Tool 的输入参数是否适合直接从用户请求中获得，以及工具是否封装好了完成业务所需的步骤。
    
    实际上，两种设计都可以。
    
    设计方式        Agent 需要调用什么                                    谁负责组织操作流程
直接注册数据库函数   查询、新增、修改、删除等底层工具                            Agent 负责组合多个工具
注册业务函数      addtodo_by_AI、updatetodo_by_AI、deletetodo_by_AI 等   业务函数负责内部流程，Agent 负责选择业务工具

	
例如，完全可以把 delete_todo(todo_id) 直接注册为 Tool，再让 Agent 先查询标题、获取 ID，然后调用删除工具。
因此，V4 并不是因为 Agent 做不到多步操作，才封装增删改。 
而是选择把复杂的业务流程放进业务函数，让 Agent 调用时更简单，也更容易集中控制。

你只需要先记住一个原则：数据库函数负责具体的数据操作，业务函数负责组织业务步骤，Agent 负责理解用户需求并选择合适的工具。 
查询可以直接注册数据库函数，增删改也可以直接注册；是否封装，取决于业务流程的复杂程度和你希望由谁来控制流程。
"""

from db import (
    get_todo_title,
    get_todo_title_count,
    get_todos,
    get_todo_count,
    get_todo_status,
    get_todo_status_count,
    get_todo_deadline,
    get_todo_deadline_count,
    get_todo_advanced,
    get_todo_advanced_count
)

#以下5个工具都是LangChain Tool
### 1、定义函数，用来当AI调用的工具
"""
Tool的三个部分记成：
    Tool
    ├── name          → 我叫什么
    ├── description   → 我能干什么、什么时候用我
    └── args_schema   → 调用我需要什么参数
~~~~~~~~~~~~~
Agent 的一次 Tool 调用实际上经历:
    用户自然语言
          ↓
    LLM理解用户意图
          ↓
    选择Tool
          ↓
    生成Tool参数
          ↓
    Agent执行Tool
          ↓
    Tool返回结果
    
每次 Tool 返回结果后，Agent 都会让 LLM 再判断一次“下一步做什么”。
Chain 是程序决定流程，Agent 是 LLM 在运行过程中决定下一步。
"""

# 1.1 定义根据title查询数据的函数（工具1）
@tool
def search_todos_by_title(title, page, page_size):
    """
    给AI使用的工具函数。
    注意：这个函数不是给用户直接调用的。
    它的作用：
        把一个复杂数据库查询能力，
        包装成AI能够理解的工具。

    根据标题关键词查询Todo。
    """
    todos = get_todo_title(title, page, page_size)      #调用db文件中的“根据title查询数据的函数”
    total = get_todo_title_count(title)

    return {
        "data": todos,
        "total": total,
        "page": page,
        "page_size": page_size
    }

# 1.2 定义查询所有Todo数据的函数 （工具2）
@tool
def search_all_todos(page, page_size):
    """
    给AI使用的工具函数。
    作用：
        查询数据库中的所有Todo。
    """
    todos = get_todos(page, page_size)
    total = get_todo_count()

    return {
        "data": todos,
        "total": total,
        "page": page,
        "page_size": page_size
    }

# 1.3 定义根据status查询函数（工具3）
@tool
def search_todos_by_status(status, page, page_size):
    """
    根据任务状态查询Todo。

    参数：
        status:
            pending表示未完成任务。
            completed表示已完成任务。

        page:
            查询页码。

        page_size:
            每页数量。
    """
    todos = get_todo_status(status, page, page_size)    #引用db中的get_todo_status()查询函数
    total = get_todo_status_count(status)

    return {
        "data": todos,
        "total": total,
        "page": page,
        "page_size": page_size
    }

# 1.4 定义根据deadline查询数据的函数（工具4）
@tool
def search_todos_by_deadline(start_time, end_time, page, page_size):
    """
    给AI使用的工具函数。
    作用：
        根据截止时间范围查询Todo。
    """
    todos = get_todo_deadline(
        start_time,
        end_time,
        page,
        page_size
    )

    total = get_todo_deadline_count(start_time,end_time)

    return {
        "data": todos,
        "total": total,
        "page": page,
        "page_size": page_size
    }

# 1.5 复合查询 （工具5）
# 上面4个：一个工具 = 一个条件。例如：status → search_todos_by_status
# 现在：一个工具 = 多个可选条件。例如：title + status + deadline
@tool
def search_todos_advanced(
        title=None,
        status=None,
        start_time=None,
        end_time=None,
        page=1,
        page_size=10):
    """
    根据多个条件组合查询Todo。

    支持条件：
        title:
            标题关键词。

        status:
            pending或completed。

        start_time:
            开始时间。

        end_time:
            结束时间。

        page:
            查询页码。

        page_size:
            每页数量。
    """

    todos = get_todo_advanced(      #根据多个可选条件查询具体的 Todo 数据
        title,
        status,
        start_time,
        end_time,
        page,
        page_size
    )

    total = get_todo_advanced_count(    #get_todo_advanced筛选出来的数据的个数
        title,
        status,
        start_time,
        end_time
    )

    return {"data": todos,
        "total": total,
        "page": page,
        "page_size": page_size}

"""
LangChain 的：
    @tool
    def xxx():
会自动生成：工具的name、description、parameters schema，不需要自己写 JSON schema。
"""
#⭐⭐选择让 Agent 自己选择查询方式，而不是让 searchtodo_by_AI() 在内部选择查询方式。
#所以是把searchtodo_by_AI()对应的5种情况执行的函数都注册成工具


### 2、定义AI工具列表：用于把Python中的查询工具“介绍”给大模型
ai_tools = [
    search_todos_by_title,
    search_all_todos,
    search_todos_by_status,
    search_todos_by_deadline,
    search_todos_advanced,

    addtodo_by_AI,      #相较于前面的版本，本版本“增删改”也注册成工具
    updatetodo_by_AI,
    deletetodo_by_AI
]

# LangChain工具列表

