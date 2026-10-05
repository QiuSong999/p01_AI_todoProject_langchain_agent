"""各种prompt"""
"""

LangChain Prompt模板。
作用：
    管理发送给大模型的提示词。

主要类型：

ChatPromptTemplate:
    用于聊天模型。
    可以生成：
        SystemMessage
        HumanMessage

PromptTemplate:
    用于普通字符串模板。
"""

from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
#导入LangChain的提示词模板类。作用：把以前手写字符串prompt，改成可复用的模板
"""
ChatPromptTemplate：用于聊天模型的消息格式Prompt。
# 例如：system消息 + human消息。

PromptTemplate：用于普通字符串Prompt模板。
# 例如：固定提示词 + {变量}。
"""

"""⭐下面定义的函数均不直接接收参数。因为LangChain的设计是：先创建模板 → 后面调用format()填充真实数据"""
#~~~~~~~~~~~~~~~~~~意图识别类提示词~~~~~~~~~~~~~~~~~~
### 1、创建“判断用户操作(add/update/delete/search)”的聊天提示词模板
# 作用：创建一个LangChain ChatPromptTemplate对象。
# ChatPromptTemplate用于构建聊天消息形式的提示词。system：告诉AI角色和需要遵守的规则。 human：放用户实际输入。
# {message}：变量占位符，调用模板时再填入真实用户输入。
#模板中预留 {message} 变量,真正的用户输入会在调用模板时传入。
def intent_prompt():     #创建一个LangChain提示词模板对象
    return ChatPromptTemplate.from_messages([
        ("system",              #system消息：告诉AI应该做什么
        """
        你是一个todo助手。
        判断用户想执行什么操作。
    
        action只能是add、update、delete、search。
        content返回用户原话。
        
        输出格式要求
        {format_instructions} 
        """),
            # format_instructions}：PydanticOutputParser根据AIAction模型自动生成的格式说明。
            # 作用：告诉大模型最终返回结果必须符合AIAction的数据结构。
            #其中AIAction是schema.py中定义的Pydantic模型。
           # Parser = PydanticOutputParser(pydantic_object=AIAction)使用该模型作为输出校验和解析规则。

        # human消息：{message}会在调用模板时被替换成真实用户输入
        ("human", "用户输入：{message}")])     #这里的{message}是变量占位符

# 函数 → 返回ChatPromptTemplate模板 → chain.invoke()执行时传入变量 → 自动替换{message}和{format_instructions} → 生成聊天消息列表，发送给ChatOpenAI
#与PromptTemplate的区别：PromptTemplate → 主要生成一段字符串  ； ChatPromptTemplate → 生成聊天消息列表



#~~~~~~~~~~~~~~~~~~查询类提示词~~~~~~~~~~~~~~~~~~
### 2、负责生成“AI查询Todo时选择工具”的提示词模板。
#作用：创建一个PromptTemplate。这个模板告诉AI：你有哪些查询工具，用户提出查询需求时应该选择哪个工具。
#这里不查询数据库。这里只负责告诉AI如何选择工具。
def search_prompt():
    # 使用ChatPromptTemplate创建聊天消息形式的Prompt。
    # 后面调用.invoke()后，可以通过.messages获取SystemMessage。
    return ChatPromptTemplate.from_messages([
        ("system",      # system消息：告诉AI身份、任务以及工具选择规则。
        """
            你是一个Todo查询助手，根据用户需求选择合适的查询工具。
            
            规则：
                1. 如果用户查询标题关键词，调用 search_todos_by_title。
                2. 如果用户查询全部任务，调用 search_all_todos。
                3. 如果用户查询状态，例如未完成、已完成，调用 search_todos_by_status。
                4. 如果用户查询某个时间范围，例如：
                   - 今天有哪些任务
                   - 明天有哪些任务
                   - 后天有哪些任务
                   - 某日期有哪些任务
                    必须调用 search_todos_by_deadline。
                5. 如果用户同时提供多个查询条件：
                   调用 search_todos_advanced。

                   例如：
                   - 查询未完成的买牛肉任务
                   - 查询明天未完成任务
                   - 查询9月份pending任务
                
                deadline 查询需要生成：
                start_time:
                开始时间 YYYY-MM-DD 00:00:00
                
                end_time:
                结束时间 YYYY-MM-DD 23:59:59
                
                不要直接回答用户。
                必须选择工具。
            """)
        ])
    # ChatPromptTemplate.from_messages()创建聊天消息模板。
    # 这里只有一条system消息，所以后面：search_prompt().invoke({}).messages[0]
    # 就可以取得这个SystemMessage。


### 3、生成本次AI查询的用户提示词模板
#作用：把当前时间和用户的问题传给AI，帮助AI理解“今天、明天、后天”等时间表达
def search_user_prompt():
    # ⭐这里不直接接收message和now_time。因为LangChain的设计是：先创建模板 → 后面调用填充真实数据。
    return ChatPromptTemplate.from_messages([
        ("human",
            """
            你是一个Todo查询助手。

            当前时间：
            {now_time}

            请根据当前时间理解用户日期表达。

            用户说：
            {message}
            """
        )
    ])
# 后续调用：prompt.invoke(now_time="2026-09-26 21:00:00",message="帮我查明天的任务")
#LangChain会自动替换：{now_time} → 当前时间 ；{message} → 用户问题


### 4、查询结果返回Prompt模板
# 工具执行完成后，告诉AI如何根据查询结果生成最终回答

def search_result_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """
            工具已经执行完成。

            不要再次调用任何工具。

            只根据工具返回的数据回答用户。

            如果 total 为 0，
            明确告诉用户没有找到符合条件的待办事项。
            """
        )
    ])

#~~~~~~~~~~~~~~~~~~Todo操作类提示词(增、删、改)~~~~~~~~~~~~~~~~~~
### 5、负责生成AI新增Todo提示词模板
#作用：让AI从用户自然语言中提取：title deadline，后续交给TodoAdd校验，然后写入数据库。
def add_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",       # system消息：告诉AI新增Todo时应该如何提取数据。
            """
            你是一个todo新增助手。
            请根据用户的要求创建一个todo。

            返回内容必须包含：
            title：
            Todo任务名称

            deadline：
            Todo截止时间

            deadline必须返回完整的日期时间。
            格式必须是：
            YYYY-MM-DD HH:MM:SS

            deadline禁止直接返回：
            明天
            下午3点
            后天

            当前时间：
            {now_time}

            输出格式要求：
            {format_instructions}
            """),
        ("human","用户要求：{message}")      # human消息：放入用户实际提出的新增Todo要求。
    ])
    # ChatPromptTemplate.from_messages()创建聊天消息模板。
    # {now_time}：调用chain.invoke()时传入当前时间，
    # 让AI能够把“明天”“下午3点”等自然语言转换成具体时间。
    # {format_instructions}：PydanticOutputParser根据TodoAdd模型自动生成的输出格式要求。
    # 用来告诉AI最终返回的数据应该符合TodoAdd模型。
    # {message}：调用chain.invoke()时传入用户真实的Todo新增要求。


### 6、负责生成AI修改Todo提示词模板
#作用：创建AI修改Todo的ChatPromptTemplate模板。第一次调用AI。只负责提取用户想修改的字段。
#不负责判断修改哪一条todo。找具体todo由第二次AI完成。
def update_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",      #system消息：告诉AI身份、任务规则以及输出要求。
            """
            你是一个todo修改助手。
            请根据用户要求，提取需要修改的todo字段。
            
            可以修改的字段：
            title:
            任务名称
            deadline:
            任务截止时间
            status:
            任务状态，只能是pending或者completed

            注意：
            1. 用户没有要求修改的字段，不要返回。
            2. 不要返回解释。
            3. deadline需要根据当前时间理解“今天、明天、下午”等时间表达。

            当前时间：
            {now_time}

            输出格式要求：
            {format_instructions}
            """),       #{now_time}：占位符，调用chain.invoke()时传入当前时间。
                        #{format_instructions}：PydanticOutputParser根据AITodoUpdate模型自动生成的格式说明。
                        # 用于约束AI返回符合Pydantic模型的数据结构。
        ("human","用户要求：{message}")  #human消息：放入用户真实输入。{message}：占位符，调用chain.invoke()时传入用户修改要求。
    ])
"""
update_prompt()
        ↓
ChatPromptTemplate模板
        ↓
chain.invoke填充：
    now_time
    message
    format_instructions
        ↓
        llm
        ↓
PydanticOutputParser
        ↓
AITodoUpdate对象
"""


### 7、负责AI匹配目标Todo提示词模板（可充当删除提示词模板）
#作用：修改Todo、删除Todo时，根据用户描述，从数据库已有Todo列表里面找到目标id。
#这里只负责生成匹配提示词。不查询数据库。不删除、不修改数据。

def match_todo_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",      #system消息：告诉AI身份、任务目标以及输出规则。
            """
            你是一个todo匹配助手。

            根据用户描述，
            从todo列表中找到用户想操作的那一个。

            不要返回解释。

            输出格式要求：
            {format_instructions}
            """),   # {format_instructions}：PydanticOutputParser根据AITodoMatch模型自动生成的格式说明。
                    # 作用：告诉AI必须按照AITodoMatch的数据结构返回结果。
        ("human",
            """
            用户描述：
            {message}

            todo列表：
            {todos}
            """)
    ])
        # human消息：放入本次匹配任务的具体数据。
        # {message}：占位符。调用chain.invoke()时传入用户想修改/删除的描述。
        # {todos}：占位符。调用chain.invoke()时传入数据库查询出来的Todo列表。


