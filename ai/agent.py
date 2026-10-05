"""ai大模型总入口"""

from ai.client import llm
# 从 ai.client 导入LangChain的ChatOpenAI模型对象。后面LCEL Chain会直接使用这个llm调用DeepSeek。

from ai.prompt import intent_prompt     #从ai.prompt模块导入意图识别提示词函数
# intent_prompt()负责创建“判断用户操作(add/update/delete/search)”的PromptTemplate模板。

from langchain_core.output_parsers import PydanticOutputParser
# 导入PydanticOutputParser。作用：把AI返回的JSON结果直接解析并验证成Pydantic对象。


from schema import AIAction     # AIAction是我们定义的意图识别输出结果模板。

from ai.todo import (       # 从ai.todo模块导入 AI操作Todo数据的业务函数。
    addtodo_by_AI,          #根据用户自然语言，调用AI解析后新增Todo。
    updatetodo_by_AI,       #根据用户自然语言，调用AI解析后修改Todo。
    deletetodo_by_AI        #根据用户自然语言，调用AI匹配后删除Todo。
)
from ai.search import searchtodo_by_AI  #根据用户自然语言，调用AI匹配后查询Todo。

"""
项目中的AI调用方式：

1、LCEL Chain模式：
    ChatPromptTemplate
            ↓
    ChatOpenAI(llm)
            ↓
    PydanticOutputParser
            ↓
    Pydantic对象
    用于：
        - 用户意图识别(agent.py)


2、Tool Calling模式：
    llm.bind_tools(ai_tools)
            ↓
    AI返回tool_calls
            ↓
    执行工具函数
    用于：
        - 查询Todo(search.py)


其他Todo操作(add/update/delete)
均已经改为LCEL Chain模式：

    Prompt模板
        ↓
    ChatOpenAI(llm)
        ↓
    PydanticOutputParser
        ↓
    Pydantic对象

其中：
    查询(search)
仍然使用Tool Calling模式。
"""

### 1、定义AI判断用户意图函数
def ai_chat(message):
    prompt= intent_prompt()
    #获取意图识别的ChatPromptTemplate模板，这里只创建模板，不填充用户输入。{message}会在chain.invoke()时自动替换。
    #chain = prompt | llm | parser中的prompt也不是最终提示词，它是一个可执行组件 Runnable。

    parser = PydanticOutputParser(
        pydantic_object=AIAction)     #"pydantic_object="是告诉 Parser：按照 AIAction 这个模型规则解析和校验大模型输出。
#创建Pydantic输出解析器。作用：把AI返回的JSON结果直接解析成AIAction对象，同时按照AIAction模型检查返回的数据格式。

    format_instructions = parser.get_format_instructions()   #.get_format_instructions()	解析器的方法，返回一段格式说明字符串
    #虽然已经告诉 Parser：“你最后要按照 AIAction 来解析和校验。”但是大模型本身并不知道AIAction这个Python类。所以需要把AIAction转给大模型听
    #把AIAction的格式要求生成一段文字，然后放入 Prompt，提前告诉大模型应该怎么输出。
    #其实就是从解析器里取出"格式说明书"，一段告诉模型该按什么格式输出的文字。
    """
    注意与上面pydantic_object=AIAction的作用的区别：
    
    代码	                                作用	               对象
    pydantic_object=AIAction	定义解析规则、校验规则	      Parser
    format_instructions	        生成输出格式提示，让AI遵守	  大模型
    ~~~~~
    PydanticOutputParser → 解析 + 校验
    get_format_instructions() → 生成Prompt格式说明
    """

    chain = prompt | llm | parser
    # 创建LCEL Chain，执行流程: ChatPromptTemplate → ChatOpenAI → PydanticOutputParser

    action = chain.invoke(      #执行chain
        {"message": message,
    #左边的 "message"对应Prompt 模板（intent_prompt()）里面的占位符。右边的 message：这个是 ai_chat() 函数接收到的真实参数。
         "format_instructions": format_instructions}  #左边的"format_instructions"对应Prompt 模板（intent_prompt()）里面的占位符
        #右边的format_instructions，是由format_instructions = parser.get_format_instructions()得到
    )
    """
    执行Chain。
     用户输入的message会传入ChatPromptTemplate，替换模板中的{message}变量。
     返回结果已经经过PydanticOutputParser解析，以这里直接得到AIAction对象。
    """

    return action   #action是经过PydanticOutputParser解析后的AIAction对象
    #如用户输入message为"明天上午提醒我买牛奶"，最终返回的是 ：action="add",content="明天上午提醒我买牛奶"
    #action数据类型为：<class 'schema.AIAction'>
    """
    
    AIAction模型(schema.py)
         ↓
    PydanticOutputParser
          ↓
    format_instructions
         ↓
    ChatPromptTemplate
         ↓
    ChatOpenAI
          ↓
    PydanticOutputParser
         ↓
    AIAction对象
    """



### 2、AI总入口函数
def todo_by_AI(message):
    action = ai_chat(message)   #调用意图识别函数。AI判断意图，得出用户想操作的动作（增/删/改/查），并返回AIAction对象。
    # 例如ai_chat(message)输出：action = {"action": "search", "content": "帮我查一下没完成的任务"}

    # action = add / update / delete / search
    if action.action == "add":
        result = addtodo_by_AI(action.content)      #AI增
        #把用户输入的消息，当作参数传给addtodo_by_AI(用户输入的消息)

    elif action.action == "update":
        result = updatetodo_by_AI(action.content)   #AI改

    elif action.action == "delete":
        result = deletetodo_by_AI(action.content)   #AI删

    elif action.action == "search":
        result = searchtodo_by_AI(action.content)  #AI查

    else:
        raise ValueError("unknown action")


    return result

"""
    用户输入自然语言
        ↓
    todo_by_AI()
        ↓
    ai_chat()
        ↓
    intent_prompt()
        ↓
    ChatPromptTemplate
        ↓
        llm
        ↓
PydanticOutputParser
        ↓
    AIAction对象
        ↓
    根据action分发业务

add → addtodo_by_AI()
update → updatetodo_by_AI()
delete → deletetodo_by_AI()
search → searchtodo_by_AI()

"""


















