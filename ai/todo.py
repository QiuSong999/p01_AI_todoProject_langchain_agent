"""AI操作单个Todo数据的业务逻辑: AI新增/修改/删除"""

from ai.client import llm
from db import (
    add_todo,
    update_all,
    delete_todo,
    get_todo_id,
    get_all_titles
)

from ai.prompt import (     #从prompt模块导入AI提示词生成函数
    add_prompt,             #生成AI新增Todo的提示词
    update_prompt,          #生成AI提取修改字段的提示词
    match_todo_prompt       #生成AI匹配目标Todo id的提示词
)


from datetime import datetime   #导入时间，让AI知道现在的时间，助于其理解我们说的明天、后天
from pydantic import ValidationError
from fastapi import HTTPException


from schema import TodoAdd, AITodoUpdate,AITodoMatch

from langchain_core.output_parsers import PydanticOutputParser  # 导入LangChain的Pydantic输出解析器。
# 作用：
# 1. 根据指定的Pydantic模型（例如TodoAdd、AIAction）检查AI输出格式。
# 2. 将AI返回的JSON结果直接转换成对应的Pydantic对象
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### 0、 定义AI匹配Todo id的公共辅助函数
def match_todo_id_by_AI(message):
    """
    根据用户自然语言描述，
    使用AI匹配数据库中对应的Todo id。
    例如：
    用户：
        删除买牛奶
    数据库：
        [
            {"id":1,"title":"学习python"},
            {"id":2,"title":"买牛奶"}
        ]
    AI返回：
        id=2
    """

    todos = get_all_titles()
    # 获取数据库已有Todo标题列表。提供给AI，让AI知道有哪些任务可以匹配。

    prompt = match_todo_prompt()    # 创建匹配Todo的Prompt模板。


    parser = PydanticOutputParser(
        pydantic_object=AITodoMatch)
    # 创建Pydantic解析器。要求AI返回：{ "id": 数字}

    format_instructions = parser.get_format_instructions()
    # 根据AITodoMatch模型生成AI输出格式要求。

    chain = prompt | llm | parser
    # LCEL链：Prompt  → DeepSeek  →  Pydantic解析 →  最终得到AITodoMatch对象

    result = chain.invoke(
        {"message": message,         # 用户自然语言需求
        "todos": todos,             # 数据库已有Todo列表
        "format_instructions": format_instructions      # 告诉AI返回格式
        } )

    return result.id        # 返回匹配到的Todo id



### 1、定义AI添加 Todo/数据 的业务函数
def addtodo_by_AI(message):
    try:
        now_time = datetime.now()

        prompt = add_prompt()       #AI新增Todo的Prompt模板,这里返回的是PromptTemplate对象，不是最终发送给AI的完整提示词。真实数据会在chain.invoke()时传入。
        parser = PydanticOutputParser(pydantic_object=TodoAdd)          #创建Pydantic输出解析器。
        # 作用：1. 按照TodoAdd模型检查AI返回的数据结构。2. 将AI返回的JSON结果直接转换成TodoAdd对象。

        format_instructions = parser.get_format_instructions()
        #将TodoAdd模型生成的JSON格式要求传给Prompt，让AI按照TodoAdd结构返回数据。

        chain = prompt | llm | parser           # 创建LCEL Chain
        # 执行流程：PromptTemplate → ChatOpenAI(llm) → PydanticOutputParser →  TodoAdd对象

        todo = chain.invoke(
            {"now_time": now_time,   #对应add_prompt()中的{now_time}： 用于让AI理解“今天、明天”等时间表达。
             "message": message,     # 对应add_prompt()中的{message}：用户输入的自然语言，例如："明天上午10点提醒我买牛奶"

             "format_instructions": format_instructions  #对应add_prompt()中的{format_instructions},
                                # 将PydanticOutputParser根据TodoAdd模型生成的格式要求传入Prompt，用于约束大模型输出结构。
            } )

        new_id = add_todo(todo.title, todo.deadline)     #添加到数据库（上行代码创建的TodoAdd对象可以通过.的方式来取响应的值）
        return get_todo_id(new_id)      #返回新增后的完整数据

    #AI输出的数据不符合TodoAdd要求。 例如：{"title":"买牛奶"}，缺少deadline
    except ValidationError as e:
        raise Exception("ai output data validation failed") from e


    #其他未知错误.例如：大模型调用失败、Pydantic解析失败、MySQL操作失败。
    except Exception as e:
        # raise Exception(f"ai创建todo失败：{e}")    如果错误里面包含：api key 用户信息 数据库信息 可能泄露。
        raise Exception("ai todo operation failed") from e
"""
AI新增流程：
    用户输入：
    "明天上午10点提醒我买牛奶"
            ↓
    addtodo_by_AI()
            ↓
    add_prompt()
            ↓
    生成PromptTemplate模板
            ↓
    PydanticOutputParser(TodoAdd)
            ↓
    生成输出格式要求
            ↓
    chain.invoke()
            ↓
    ChatPromptTemplate填充：
        now_time
        message
        format_instructions
            ↓
     ChatOpenAI(llm)
            ↓
        AI返回JSON
            ↓
    PydanticOutputParser解析
            ↓
       TodoAdd对象
            ↓
        add_todo()
            ↓
          MySQL
"""

### 2、定义AI修改 Todo/数据 的业务函数
def updatetodo_by_AI(message):
    try:
        now_time = datetime.now()

        prompt = update_prompt()        #获取AI修改Todo的Prompt模板。
        # 获取AI修改Todo的ChatPromptTemplate模板。返回的是ChatPromptTemplate对象，不是最终发送给AI的消息。真实数据会在chain.invoke()时传入。

        parser = PydanticOutputParser(pydantic_object=AITodoUpdate)       # 创建Pydantic输出解析器。
        # 作用：1. 按照AITodoUpdate模型检查AI返回的数据结构。2. 将AI返回结果直接转换成AITodoUpdate对象。

        format_instructions = parser.get_format_instructions()    # 根据AITodoUpdate模型自动生成输出格式要求。
        # 作用：根据AITodoUpdate模型自动生成输出格式说明。告诉大模型返回结果需要符合什么数据结构。

        chain = prompt | llm | parser
        # 创建LCEL Chain。
        # 执行流程：ChatPromptTemplate → ChatOpenAI(llm) → PydanticOutputParser → AITodoUpdate对象

        todo = chain.invoke(
            {"now_time": now_time,   # 对应update_prompt()中的{now_time}，用于理解今天、明天等时间表达。
              "message": message,    # 对应update_prompt()中的{message}，用户输入的修改要求。


              "format_instructions": format_instructions
              # 对应update_prompt()中的{format_instructions}将AITodoUpdate格式要求传给大模型。
            } )
        """
        第一次 AI 流程:
            用户：把买牛奶改成完成
            ↓
            update_prompt()
            ↓
            ChatPromptTemplate
            ↓
            llm
            ↓
            PydanticOutputParser
            ↓
            AITodoUpdate(status="completed")
        """

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # 第二次调用AI：作用：根据用户描述 + 数据库已有todo列表，找到用户真正想修改的是哪一条数据
        # 根据用户描述，AI匹配目标Todo id
        todo_id = match_todo_id_by_AI(message)      #match_todo_id_by_AI为AI匹配Todo id的公共辅助函数

        # 根据AI找到的id，利用查询数据函数找到对应的数据
        todo_data = get_todo_id(todo_id)    #返回找到的数据

        #~~~~~~~~~~~~~~~~~~~~~~~
        if todo_data is None:
            raise HTTPException(
                status_code=404,
                detail="todo not found"
            )

        # 如果AI返回了新的status，就使用AI的；如果AI没有返回status，就保留数据库原来的status
        status = (
            todo.status
            if todo.status is not None
            else todo_data["status"]
        )

        # 如果AI返回了新的deadline，就使用AI的；如果AI没有返回deadline，就保留数据库原来的deadline
        deadline = (
            todo.deadline
            if todo.deadline is not None
            else todo_data["deadline"]
        )

        # 根据数据库真实id修改数据
        result = update_all(        #update_all()返回修改了几条数据
            todo_data["id"],
            status,
            deadline
        )

        if result == 0:
            raise Exception("todo update failed")

        # 修改完成后重新查询最新数据返回
        return get_todo_id(todo_data["id"])

    # HTTPException需要原样抛出
    except HTTPException:
        raise

    # AI返回格式错误
    except ValidationError as e:
        raise Exception(
            "ai output data validation failed"
        ) from e

    # 其他错误
    except Exception as e:
        raise Exception(
            "ai todo operation failed"
        ) from e
"""
用户输入
    ↓
updatetodo_by_AI()
    ↓
第一次AI
    ↓
update_prompt()
    ↓
AITodoUpdate
    ↓
得到修改字段
(status/deadline)
    ↓
第二次AI
    ↓
match_todo_prompt()
    ↓
AITodoMatch
    ↓
得到目标todo_id
    ↓
get_todo_id()
    ↓
获取数据库原数据
    ↓
合并新字段 + 原字段
    ↓
update_all()
    ↓
返回最新todo
"""


### 3、定义AI删除数据函数
def deletetodo_by_AI(message):
    try:
        todo_id = match_todo_id_by_AI(message)

        result = delete_todo(todo_id)
        # 根据AI找到的id删除数据库中的Todo。
        # delete_todo()返回删除的数据数量。

        if result == 0:
            raise HTTPException(
                status_code=404,
                detail="todo not found"
            )

        return {
            "id": todo_id,
            "message": "todo deleted"
        }

    except HTTPException:
        raise

    except ValidationError as e:
        raise Exception(
            "ai output data validation failed"
        ) from e


    except Exception as e:
        raise Exception(
            "ai todo operation failed"
        ) from e

"""
AI删除Todo流程：

    用户自然语言
        ↓
    deletetodo_by_AI()
        ↓
    获取数据库已有Todo标题列表
        ↓
    创建match_todo_prompt()
        ↓
    创建PydanticOutputParser(AITodoMatch)
        ↓
    chain.invoke()
        ↓
    ChatPromptTemplate填充：
        message
        todos
        format_instructions
        ↓
    llm调用DeepSeek
        ↓
    PydanticOutputParser解析
        ↓
    得到AITodoMatch对象
        ↓
    match_result.id获取目标Todo id
        ↓
    delete_todo(todo_id)
        ↓
    返回删除结果
"""


