"""DeepSeek连接ask_deepseek()"""

"""
V4 agent版本的整体执行流程：
    用户输入自然语言
    例如：先查找未完成的买牛奶任务，再将它标记为完成
           ↓
    main.py / api.py
    接收请求，进入后端业务流程
           ↓
    ai/agent.py
    调用 LangChain Agent，交给模型判断下一步操作
           ↓
    ai/tools.py
    向 Agent 提供可以调用的工具列表
           ↓
    ai/todo.py → db.py
    执行业务逻辑，访问 MySQL 查询或修改 Todo
           ↓
    Agent 返回最终结果
    模型根据工具执行结果组织自然语言回答
    
⭐⭐关键是 Agent 能够根据用户需求，自主决定调用哪个工具，并根据工具执行结果决定是否继续调用工具。

例如，用户要求先查询再修改，Agent 可以先调用查询工具，获得查询结果后，再调用修改工具。
这个过程不需要我们提前把所有步骤写死在固定流程里。
"""


import os
from dotenv import load_dotenv              #用于加载.env文件
from langchain_openai import ChatOpenAI     #LangChain提供的模型调用类，用来替代原来的OpenAI客户端

load_dotenv()      # 读取 .env 文件，把里面的环境变量加载到程序中。

"""

"""

### 1、创建一个LangChain模型对象
llm = ChatOpenAI(
    model="deepseek-flash",                  # 指定使用的模型
    api_key=os.getenv("deepseek_api_key"),   # 从.env中读取DeepSeek API Key
    base_url="https://api.deepseek.com"      # DeepSeek兼容OpenAI接口，所以仍然使用这个地址
)
# Python程序 → llm(ChatOpenAI) → DeepSeek API服务器 → 大模型

### 2、把调用 DeepSeek 的代码封装成函数（自己封装的“提问函数”。）
def ask_deepseek(message):
    try:
        response = llm.invoke(message)      #调用LangChain模型
        # message会自动转换成模型需要的消息格式
        # 返回的是LangChain的AIMessage对象
        """
        response是LangChain返回的AIMessage对象。
            例如：
                AIMessage(
                    content='',
                    tool_calls=[
                               {'name':'search_todos_by_status',
                                'args':{'status':'pending','page':1,'page_size':10},
                                'id':'call_xxx'  
                                "type": "tool_call"} 
                                ],          #当只有一个工具时，tool_calls列表里面就一个大字典元素
                    additional_kwargs={……},
                    response_metadata={……},
                    usage_metadata={……},
                    ……          
                        )
        其中AIMessage里面可通过.方式取出的(属性)有：
            1、content:消息内容
            2、tool_calls:保存AI选择的工具信息。
                （tool_calls里面的args保存AI生成的工具参数）
            3、response_metadata:模型响应元数据
            4、usage_metadata:Token 使用统计                      
        """

        return response.content      #AIMessage对象中的content属性，就是模型返回的文本内容

    except Exception as e:
        print("ai service error:", e)
        raise Exception(f"ai service error: {e}") from e

"""
agent.py文件中 result = ask_deepseek(prompt)：拿到的（返回的）是content字符串
    例如{"action":"search","content":"查询未完成任务"}，然后json.loads(result)继续处理
    
但search.py没有使用ask_deepseek()，而是用llm.bind_tools(ai_tools)原因是：查询需要完整的AIMessage
    因为里面有tool_calls，如果变成：response.content工具调用信息会丢失。这就是为什么项目现在分两种调用
    
普通JSON            Tool Calling      
    |                  | 
ask_deepseek()     llm.invoke()    
    |                  |        
返回content字符串    返回AIMessage      
#注：Tool Calling必须保留完整 AIMessage
"""
