# 从 LangChain 模型客户端模块导入 llm。
# llm 是 ChatOpenAI 创建的模型对象，用于调用 DeepSeek API。

from ai.client import llm       #LangChain，所以改为导入ChatOpenAI创建的llm对象。
from langchain_core.messages import ToolMessage    #ToolMessage：保存Python执行工具后的结果，第二次发送给AI。
"""
ToolMessage:保存Python执行工具后的结果，第二次发送给AI。
SystemMessage:已经由ChatPromptTemplate.invoke()自动生成，不需要手动导入。
HumanMessage:已经由ChatPromptTemplate.invoke()自动生成，不需要手动导入。
"""



# 从 tools 模块导入 AI 查询工具函数
# 这些函数是真正执行数据库查询的业务工具
from ai.tools import (          #从 ai.tools 模块导入AI查询Todo时使用的5个工具函数。
    search_todos_by_title,       # 根据标题关键词查询Todo
    search_all_todos,            # 查询全部Todo
    search_todos_by_status,      # 根据状态查询Todo
    search_todos_by_deadline,    # 根据截止时间范围查询Todo
    search_todos_advanced,        # 根据多个条件组合查询Todo（标题、状态、时间范围）
    ai_tools                     #导入大模型可用的工具列表
)

from ai.prompt import search_prompt,search_user_prompt,search_result_prompt     #从 prompt 模块导入查询流程中使用的三个 prompt
# search_prompt：告诉AI如何选择查询工具
# search_user_prompt：提供当前时间和用户的查询需求
# search_result_prompt：告诉AI如何根据工具执行结果生成最终回答


from schema import AITodoSearch     # 导入Pydantic模型，用于校验AI返回的查询参数
from pydantic import ValidationError

# FastAPI异常类，用于返回HTTP错误
from fastapi import HTTPException


# 获取当前时间：用于让AI理解“今天、明天、后天”等时间表达
from datetime import datetime

### 1、公共辅助函数(把下面的重复代码抽取出来)
# 把AI返回的查询参数解析和校验逻辑抽取出来。避免 search_todos_by_title、search_all_todos、
# search_todos_by_status 三个分支重复写相同代码。

## 1.1 公共参数解析函数
def parse_ai_search_arguments(tool_call):
    # 这个参数tool_call：是第一次调用AI后，返回的工具调用信息。包括 name：AI选择的工具名称。args：AI生成的工具参数
    try:
        arguments = tool_call["args"]
        #LangChain的LangChain已经自动把AI生成的工具参数解析成Python字典。例如：{"title":"牛肉","page":1,"page_size":10}

        return AITodoSearch(**arguments)    # 使用Pydantic中的AITodoSearch校验AI生成的查询参数。
                                            #**arguments 是把字典拆成关键字参数。
    # 原来arguments = {"title": "牛奶","page": 1,"page_size": 10}，经过**arguments变成
    # AITodoSearch(title="牛奶",page=1,page_size=10)

    except ValidationError as e:
        #如果AI生成的参数不符合AITodoSearch定义，Pydantic会抛出ValidationError。
        raise HTTPException(
            status_code=400,
            detail="ai query parameters invalid"
        ) from e

## 1.2 执行AI选择的查询工具函数（真正的查询数据动作）
#第一次调用AI后，获取意图，然后根据意图，通过此函数来让不同的意图执行相应的查询函数
"""
作用：
    根据第一次DeepSeek返回的tool_call，
    判断AI选择了哪个查询工具，
    然后执行对应的数据库查询函数。

流程：
    第一次DeepSeek
            ↓
    返回tool_call
            ↓
    execute_search_tool()
            ↓
    读取工具名称(tool_name)
            ↓
    解析AI生成的查询参数
            ↓
    调用对应查询函数
            ↓
    返回MySQL查询结果
 """
"""
    例如：
    用户：
        "查询未完成的任务"
    第一次AI返回：
    {
        "name": "search_todos_by_status",
        "arguments": {
            "status":"pending",
            "page":1,
            "page_size":10
        }
    }
    
    execute_search_tool()
            ↓
    识别工具：
    search_todos_by_status
            ↓
    执行：
    search_todos_by_status(
        "pending",
        1,
        10
    )
            ↓
    返回数据库结果
"""
def execute_search_tool(tool_call):
    """
    根据AI选择的工具，执行对应数据库查询
    参数:
        tool_call:
            第一次AI返回的工具调用信息
    返回:
        数据库查询结果
    """
    #result.tool_calls中的每一个工具调用信息是dict结构
    tool_name = tool_call["name"]     #获取AI第一次返回的工具名称，如"search_todos_by_status"

    search_data = parse_ai_search_arguments(tool_call)  #利用###1定义的函数，解析AI生成的工具参数
    #LangChain已经把工具参数转换成Python字典。这里通过AITodoSearch(**arguments)，把字典转换成Pydantic对象，并校验参数格式。

    #工具1
    if tool_name == "search_todos_by_title":    #如果是根据title查询相关数据
        # ⭐⭐下面的Tool.invoke() = 执行工具；LLM.invoke() = 调用模型；Prompt.invoke() = 填充模板
        return search_todos_by_title.invoke(    #search_todos_by_title：根据title查询数据的函数（工具1）
            {
                "title": search_data.title,     #search_data为AITodoSearch对象
                "page": search_data.page,
                "page_size": search_data.page_size
            } )
    # from ai.tools import search_todos_by_title现在导入的已经不是普通函数。
    #因为在tools文件中，search_todos_by_title函数经@tool 修饰后变成StructuredTool对象

    # 工具2
    elif tool_name == "search_all_todos":

        return search_all_todos.invoke(
            {"page": search_data.page,
             "page_size": search_data.page_size}
        )

    # 工具3
    elif tool_name == "search_todos_by_status":

        return search_todos_by_status.invoke(
            {
                "status": search_data.status,
                "page": search_data.page,
                "page_size": search_data.page_size
            } )

    # 工具4
    elif tool_name == "search_todos_by_deadline":

        return search_todos_by_deadline.invoke(
            {
                "start_time": search_data.start_time,
                "end_time": search_data.end_time,
                "page": search_data.page,
                "page_size": search_data.page_size
            }
        )

    # 工具5
    elif tool_name == "search_todos_advanced":

        return search_todos_advanced.invoke(
            {
                "title": search_data.title,
                "status": search_data.status,
                "start_time": search_data.start_time,
                "end_time": search_data.end_time,
                "page": search_data.page,
                "page_size": search_data.page_size
            }
        )

    else:
        raise HTTPException(
            status_code=400,
            detail="unknown search tool"
        )

## 1.3 整理数据库查询结果函数
def format_search_result(query_result):
    # query_result 就是：410行代码，即真正执行查询工具后得到的结果，传达该函数中
    """
    整理数据库查询结果。

    作用：
        将不同查询工具返回的数据，
        统一转换成第二次AI需要的格式。

    输入：
        数据库查询结果

    输出：
        包含：
        当前页数据
        总数量
        当前页
        每页数量
        总页数
    """

    todos = query_result["data"]             # 当前页查询到的Todo数据
    total = query_result["total"]            # 符合条件的总数量

    page = query_result["page"]              # 当前页码
    page_size = query_result["page_size"]    # 每页数量

    total_pages = (total + page_size - 1) // page_size
    # 计算总页数。例如：total=7，page_size=5。结果：2页

    return {
        "data": todos,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages
    }


### 2、AI工具选择函数
# 作用：第一次调用AI，让AI根据用户需求选择应该执行哪个工具，并生成工具参数
def ask_deepseek_tool(messages):  # 让AI选择工具
    # 这里的message是343行代码，已经准备好的聊天消息，其中
    # SystemMessage：告诉AI有哪些查询工具，以及每个工具什么时候使用 ； HumanMessage：用户真正的查询要求
    """
    第一次调用 DeepSeek。
    作用：
        不是直接回答用户问题，
        而是让 AI 判断：
        1. 用户想查询什么
        2. 应该调用哪个工具
        3. 调用工具时需要传什么参数

    流程：
        用户问题
            ↓
        DeepSeek
            ↓
        选择工具 + 生成参数
            ↓
        Python执行对应函数
    """
    try:
        llm_with_tools = llm.bind_tools(ai_tools)       # 普通AI + 可使用的工具 = 具有工具调用能力的AI
        #.bind_tools()是LangChain提供的方法。作用：把查询工具告诉大模型，让AI可以选择调用哪个工具。
        response = llm_with_tools.invoke(messages)      #LangChain统一使用invoke()调用模型。
        # 把问题和工具一起打包发给DeepSeek，让AI来做“选择题”——判断需不需要用工具、用哪个工具、以及提取什么参数。
        # 返回一个AIMessage对象

        return response
        # 返回AI生成的消息对象。里面可能包含：
        # .content 普通回答
        # .tool_calls AI选择的工具

    except Exception as e:
        print("ai service error:", e)
        raise Exception(f"ai service error: {e}") from e
"""
用户：
"帮我查一下没完成的任务"
        ↓
searchtodo_by_AI()
        ↓
      调用
        ↓
ask_deepseek_tool()
        ↓
    DeepSeek判断：
    假如要调用：
    get_pending_todos
    参数：
    {
     page:1,
     page_size:10
    }
        ↓
返回给 searchtodo_by_AI()
        ↓
Python执行真正查询
        ↓
      MySQL
~~~~~~~   
所以：ask_deepseek_tool = AI决策阶段
"""

### 3、 定义AI查询Todo/数据 的业务函数  （上面的函数都是从本函数单独抽出来的）
def searchtodo_by_AI(message):
    """
    AI查询Todo主流程。
    完整流程如下：
        用户：
            "查找买相关的待办"
                ↓
        第一次调用AI：
            判断调用哪个工具
                ↓
        Python执行工具：
            查询MySQL
                ↓
        第二次调用AI：
            把查询结果整理成人类语言
                ↓
        返回用户
    """
    now_time = datetime.now()

    # 使用ChatPromptTemplate.invoke()给模板传入真实参数返回LangChain标准HumanMessage对象
    search_user_message = search_user_prompt().invoke(      #search_user_prompt()：查询用户Prompt模板（未填充，含占位符）
        {"now_time": now_time,                              #.invoke({...}) ：把真实数据填入 Prompt 模板
         "message": message}
        ).messages[0]
    # 改提示词是让AI理解用户输入的今天，昨天，明天等词汇，下面的是告诉AI如何根据用户输入的话选择相应的工具执行
    """上面转换流程
    ChatPromptTemplate
    ↓ invoke()
    ChatPromptValue
    ↓.messages
    list[BaseMessage]
    ↓ [0]
    HumanMessage / SystemMessage
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       变量	                        是什么
    search_user_prompt()	    用户提示词模板（把当前时间和用户的问题传给AI，帮助AI理解“今天、明天、后天”等时间表达）
    invoke()	                给模板填真实参数
    .messages[0]	            取出生成的第一条消息
    search_user_message	        最终得到的 HumanMessage
    """

    system_message = search_prompt().invoke({}).messages[0]     #search_prompt() 里面没有变量。故invoke内传了个{}
    #search_prompt()模板告诉AI：你有哪些查询工具，用户提出查询需求时应该选择哪个工具。
    # invoke()执行后返回ChatPromptValue对象。
    # .messages取出ChatPromptValue里面的消息列表  ；  [0]取出第一个消息，因为这里的第一个消息是SystemMessage。
    # 最终system_message保存的是LangChain标准SystemMessage对象。

    # 第一次调用AI的消息列表这里直接使用invoke生成的Message对象,不需要再次包装SystemMessage和HumanMessage
    messages = [
        system_message,         # SystemMessage：查询规则（告诉AI：你有哪些查询工具，用户提出查询需求时应该选择哪个工具。）
        search_user_message     # HumanMessage：用户查询内容
        ]       #上面System和Human两个message构成完整的提示词

    result = ask_deepseek_tool(messages)  #执行AI工具选择函数。 上面message填充后作为参数传给ask_deepseek_tool，即。
    # 调用ask_deepseek_tool —— AI工具选择函数。第一次调用AI.目的：让AI根据用户需求选择工具。
    #返回：AIMessage对象，里面包含tool_calls。 注意：这里还没有查询数据库。

    """
    result是LangChain返回的AIMessage对象。
    例如：

    AIMessage(
        content='',
        tool_calls=[
            {
                'name':'search_todos_by_status',
                'args':{
                    'status':'pending',
                    'page':1,
                    'page_size':10
                },
                'id':'call_xxx'
            }     ]   )
    其中：
    tool_calls保存AI选择的工具信息。
    args保存AI生成的工具参数。
    """

    print("AI返回：", result)

    #从ask_deepseek_tool()的输出结果中，通过result.tool_calls取工具列表，再判断
    if not result.tool_calls:       #如果生成的工具列表为空
        raise HTTPException(
            status_code=400,
            detail="无法识别查询需求"
        )  # 没有工具：[] → 返回400错误：无法识别查询需求


    print("AI选择的工具：", result.tool_calls)  # 第一次调用AI是让AI得到需要用到的Tools
    # result.tool_calls 即从AI返回的response 中取出tool_calls(所用的工具)
    """
    tool_calls大概是下面的样子：
        tool_calls = [
            {
                'name': 'search_todos_by_status',
                'args': {'status': 'pending', 'page': 1, 'page_size': 10},
                'id': 'call_xxx'
            }, ……         # 理论上可以有多条，比如 AI一次想调多个工具  
            ]
    """

    tool_call = result.tool_calls[0]  # tool_calls = 工具调用列表，根据列表索引取值（该工具列表是由字典组成的）
    # result.tool_calls = [工具调用1,工具调用2,工具调用3]      #这种结构

    print("工具名称：", tool_call["name"])             #tool_call是dict。工具名称直接通过"name"获取。
    print("工具参数：", tool_call["args"])             #打印所用工具的参数


    # -------------根据AI选择的工具执行不同函数-------------
    #把第一次 AI 返回的工具调用信息传给 execute_search_tool()，由这个函数根据 AI 选择的工具执行对应的函数对数据库查询，并返回查询结果。
    query_result = execute_search_tool(tool_call)       #真正的查询动作
    """
    execute_search_tool()会根据tool_call中的工具名称：
        1. 判断AI选择的是哪个查询工具
        2. 解析AI生成的查询参数
        3. 调用对应的数据库查询函数
        4. 返回MySQL查询结果
    ~~~~~~~~~~
    根据AI选择的工具，执行对应的数据库查询
    流程：
        AI选择工具
              ↓
        execute_search_tool()
              ↓
        调用对应查询函数
              ↓
        MySQL查询
              ↓
        返回查询结果
    """

    ##5个查询工具执行完成后，统一处理查询结果
    search_result = format_search_result(query_result)      # 调用整理数据库查询结果函数，转换成统一格式，方便后续发送给第二次AI生成自然语言回答
    # 把Python执行数据库查询后的原始结果，整理成适合AI阅读的格式。
    # query_result里面包含数据库返回的数据、总数量、分页信息等。
    # 格式化后发送给第二次AI，让AI根据查询结果生成自然语言回答。

    # ~~~~~~~~~~~~~~~~~~上面第一次调用AI，让AI确定执行哪个函数（工具），并得到返回结果~~~~~~~~~~~~~~~~~~
    # 第一次调用AI：让 AI 决定“调用哪个工具”（这里必须把 tools 告诉 AI。所以第一次需要：messages + tools）
    # 第二次：已经没有“选工具”这个任务了，根据查询结果回答用户即可（不需要再传 tools）
    # ~~~~~~~~~~~~~~~~~~~~~~~~下面第二次调用AI，让AI把查询到的结果组织给我们~~~~~~~~~~~~~~~~~~~~~~~~~~

    # 创建工具执行结果消息，Python执行AI选择的工具后，把查询数据库的结果包装成 AI 能理解的tool消息。
    tool_message = ToolMessage(
        content=str(search_result),     #工具返回的数据。（把 Python 对象转换成字符串）
        tool_call_id=tool_call["id"]
    )
    # 把执行数据库查询的结果，包装成AI能读懂的"工具返回消息"，好在第二次调用AI时喂给它

    # 使用ChatPromptTemplate.invoke()生成LangChain标准SystemMessage对象
    result_system_message = search_result_prompt().invoke({}).messages[0]
        #search_result_prompt():告诉AI如何根据查询结果生成最终回答; invoke({})因为模板没有变量，所以传空字典
        #.messages[0]:取出模板生成的第一条消息。 最终得到SystemMessage对象

    """
        变量                         类型                     作用
    search_result_prompt()       ChatPromptTemplate       第二次AI回答模板
    invoke({})                  ChatPromptValue           执行模板
    .messages[0]                SystemMessage             获取第一条消息
    result_system_message       SystemMessage             第二次AI系统提示
    """


    messages = [
        result_system_message,      # 第二次AI的系统提示词
#search_prompt().invoke({}).messages[0]已经生成：SystemMessage(...)，
# 所以不用SystemMessage(content=result_system_message),不然又嵌套一层SystemMessage( content=SystemMessage(...) )
        result,                     # 第一次AI返回的工具选择结果
        tool_message                # Python执行工具后的查询结果
    ]
    """
        变量              是什么                             什么时候产生 
      result          第一次AI返回的工具选择结果        ask_deepseek_tool()之后 
    query_result      Python执行工具后的数据库结果     execute_search_tool()之后
    second_response   第二次AI生成的人类回答           llm.invoke(messages)之后 
    """

    """
    SystemMessage
        | 告诉AI任务
        ↓
    AIMessage(result)
        | AI刚才选择工具
        ↓
    ToolMessage
        | 工具执行结果
        ↓
    DeepSeek
        |
        ↓
    生成最终回答
    """

    second_response = llm.invoke(messages)

    # 这次不负责选择工具。只根据tool_message里的查询结果，整理成人类能看懂的回答。

    final_message = second_response.content
    print("第二次AI最终回答：", final_message)


    return {"message": final_message,                       # 第二次AI生成的人类可读回答
            "page" :search_result["page"],                  # 当前查询页
            "page_size":  search_result["page_size"],       # 每页多少条
            "total": search_result["total"],                # 符合查询条件的全部数据有多少条
            "total_pages": search_result["total_pages"],    # 全部数据一共多少页
            "data": search_result["data"]}                  # 当前这一页的数据

"""
①searchtodo_by_AI(): 负责整个查询流程，相当于“总经理”。
②ask_deepseek_tool(): 负责问 AI：“用户想干什么？应该用哪个工具？”
③search_todos_by_title(): 负责真正干活：“去数据库查未完成任务”。

它们的调用关系：
    用户问题 → searchtodo_by_AI() → ask_deepseek_tool() → AI选择工具 →
    → search_todos_by_title() →  db.py  →  MySQL
"""
"""
AI查询Todo流程：

    用户输入
        ⬇️
    searchtodo_by_AI()
        # AI查询总入口。
        # 负责组织完整查询流程：
        # 生成Prompt、调用AI、执行工具、生成最终回答。
        ⬇️
    ask_deepseek_tool()
        # 第一次调用大模型。
        # 作用：
        # 让AI根据用户需求选择需要使用的查询工具。
        # 返回AIMessage对象，其中包含tool_calls。
        ⬇️
    llm.bind_tools(ai_tools)
        # 将ai/tools.py中的查询工具绑定给大模型。
        # AI通过工具名称、描述、参数结构判断调用哪个工具。
        ⬇️
    AI返回tool_call
        # AI不会直接查询数据库。
        # 只返回：
        # 1.选择的工具名称
        # 2.调用工具需要的参数
        ⬇️
    execute_search_tool()
        # 接收AI返回的tool_call。
        # 根据工具名称选择对应的LangChain工具，
        # 并执行查询。
        ⬇️
    search_todos_xxx.invoke()
        # 调用LangChain封装后的查询工具。
        # 工具内部执行真正的业务逻辑。
        ⬇️
    ai/tools.py
        # LangChain工具层。
        # 负责把AI工具调用转换成数据库查询请求。
        ⬇️
    db.py
        # 数据库操作层。
        # 调用SQL语句查询MySQL。
        ⬇️
    MySQL
        # 执行SQL并返回查询结果。
        ⬇️
    第二次llm.invoke()
        # 第二次调用大模型。
        # 输入：
        # 1.第一次AI返回的信息
        # 2.Python执行工具后的查询结果
        #
        # 作用：
        # 将数据库结果整理成人类能理解的自然语言。
        ⬇️
    返回用户

"""
