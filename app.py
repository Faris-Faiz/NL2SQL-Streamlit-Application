from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import os

def init_db(user: str, password: str, host: str, name:str) -> SQLDatabase:
    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
    return db

def get_sql_chain(db, api_key, model_choice, model_provider):
    template = """
    You are a data analyst. Generate SQL queries based on user questions using the provided database schema. Ensure queries are directly executable in an SQL database and do not include any formatting such as backticks.

    <SCHEMA>{schema}</SCHEMA>

    Conversation History:
    {chat_history}

    Example:
    Question: List all customers in France with a credit limit over 20,000.
    SQL Query: SELECT * FROM customers WHERE country = 'France' AND creditLimit > 20000;
    Question: Get the highest payment amount made by any customer.
    SQL Query: SELECT MAX(amount) FROM payments;
    Question: Show product details for products in the 'Motorcycles' product line.
    SQL Query: SELECT * FROM products WHERE productLine = 'Motorcycles';
    Question: Retrieve the names of employees who report to employee number 1002.
    SQL Query: SELECT firstName, lastName FROM employees WHERE reportsTo = 1002;
    Question: List all products with a stock quantity less than 7000.
    SQL Query: SELECT productName, quantityInStock FROM products WHERE quantityInStock < 7000;
    Question: what is price of `1968 Ford Mustang`
    SQL Query: SELECT `buyPrice`, `MSRP` FROM products  WHERE `productName` = '1968 Ford Mustang' LIMIT 1;

    Your task: Based on the user's question below, generate a clean SQL query without any formatting:
    Question: {question}
    SQL Query: 
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Fetch API key safely
    api_key = api_key
    model_choice = model_choice 
    model_provider = model_provider

    if model_provider == 'OpenAI':
        llm = ChatOpenAI(model=model_choice, api_key=api_key)
    else:
        llm = ChatGroq(model=model_choice, api_key=api_key, temperature=0)

    def get_schema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query:str, db:SQLDatabase, chat_history: list, api_key, model_choice, model_provider):
    sql_chain = get_sql_chain(db, api_key, model_choice, model_provider)

    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.

    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}
    """

    prompt = ChatPromptTemplate.from_template(template)
    api_key = api_key

    if model_provider == 'OpenAI':
        llm = ChatOpenAI(model=model_choice, api_key=api_key)
    elif model_provider == 'Groq':
        llm = ChatGroq(model=model_choice, api_key=api_key, temperature=0)

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm an SQL assistant. Ask me anything about your database."),
    ]

load_dotenv()

db_user = os.environ.get("db_user")
db_password = os.environ.get("db_password")
db_host = os.environ.get("db_host")
db_name = os.environ.get("db_name")

st.set_page_config(page_title="Chat with MySQL", page_icon=":speech_bubble:")

st.title("Chat with MySQL")

with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using MySQL. Connect to the database and start chatting.")

    # Button to clear the chat history
    if st.button("Clear Chat"):
        # Reset the chat history to the initial welcome message or an empty list
        st.session_state.chat_history = [
            AIMessage(content="Chat history cleared. Ask me a new question!")
        ]

    st.text_input("User", value=db_user, key="User")
    st.text_input("Password", value=db_password, type="password", key="Password")
    st.text_input("Host", value=db_host, key="Host")
    st.text_input("DB Name", value=db_name, key="DB Name")
    
    model_provider = st.selectbox("Choose the model provider", ("OpenAI", "Groq"))
    if model_provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password")
        model_choice = st.selectbox("Model", ("gpt-4-0125-preview", "gpt-4-turbo", "gpt-3.5-turbo-0125"))
    else:
        api_key = st.text_input("Groq API Key", type="password")
        model_choice = st.selectbox("Model", ("llama2-70b-4096", "mixtral-8x7b-32768", "gemma-7b-it"))

    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_db(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["DB Name"],
            )
            st.session_state.db = db
            st.success("Connected to database!")

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = []
user_query = st.chat_input("Type a message...")

if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content = user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)
    
    with st.chat_message("AI"):
        # This retrieves the direct SQL query only
        #sql_chain = get_sql_chain(st.session_state.db)
        # response = sql_chain.invoke({
        #     "chat_history": st.session_state.chat_history,
        #     "question": user_query
        # })

        # This retrieves the natural language from the generated SQL Query
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history, api_key, model_choice, model_provider)
        st.markdown(response)

    st.session_state.chat_history.append(AIMessage(content=response))