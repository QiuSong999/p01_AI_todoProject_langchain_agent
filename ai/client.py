"""DeepSeek连接ask_deepseek()"""


"""
ai/client.py文件中
    方式1：OpenAI SDK(以前的) ：代码 →  OpenAI客户端(client) →  HTTP请求 → DeepSeek API → 返回JSON   
        创建：
            client = OpenAI(    
                api_key=os.getenv("deepseek_api_key"),
                base_url="https://api.deepseek.com" )     
        调用：  
            response = client.chat.completions.create(
                model="deepseek-flash",
                messages=[
                    {"role":"user",
                    "content":"你好"}
                ])
        返回：
            response = client.chat.completions.create(
                model="deepseek-flash",
                messages=[
                    {"role":"user",
                    "content":"你好"}
                ])                              #******返回的是ChatCompletion对象
        所以你需要自己取响应文本：response.choices[0].message.content
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
    方式2：langchain(现在的)：代码  → LangChain → ChatOpenAI →  DeepSeek（langchain的方式）
        创建： 
            llm = ChatOpenAI(
                model="deepseek-flash",                  
                api_key=os.getenv("deepseek_api_key"), 
                base_url="https://api.deepseek.com"  
        调用
            response = llm.invoke(message)      #******返回的是AIMessage对象
        返回
            llm.invoke(message.content)
        
~~~~~~~~~~~~
#langchain方式先安装pip install langchain langchain-openai 两个包是为了让你的项目具备 LangChain 调用大模型的能力。
0、LangChain可以使用：OpenAI、DeepSeek、通义、Moonshot、其他兼容OpenAI接口的模型
1、langchain包：核心框架。提供：
     PromptTemplate（提示词模板）
     Chain（链）
     Agent
     Tool
     Memory（后续学习）
2、langchain-openai包：
    连接 OpenAI 兼容模型的接口。
    虽然名字叫 openai，但是它不只支持 OpenAI。
    因为 DeepSeek 的 API 兼容 OpenAI 格式。
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
        AIMessage(
             content='',        #AI的回复文本
             tool_calls=[
               {
                name:'search_all_todos',
                args:{
                  page:1,
                  page_size:10
                } 
                }
             ]
            )
            这个就是LangChain格式。
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
