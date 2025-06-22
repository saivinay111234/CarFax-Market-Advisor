# app.py
import streamlit as st
from agent_utils import crew, Task, carfax_agent, last_rag_context, format_agent_output, initialize_genai_model

st.set_page_config(page_title="Car Market Insights", layout="centered")

st.sidebar.title("🚘 Car Intelligence Options")
option = st.sidebar.radio("Choose a Mode", ["RAG Database", "AI Agent"])

st.title("🔍 Car Market Intelligence Assistant")

if option == "RAG Database":
    st.subheader("🔎 Query the Car Database (RAG)")
    user_query = st.text_input("Enter your query (e.g., 'SUVs under $25,000 with good mileage')", key="rag_input")

    if st.button("Submit RAG Query"):
        with st.spinner("Querying RAG and Gemini..."):
            from agent_utils import RAGRetriever
            rag_tool = RAGRetriever()
            context = rag_tool._run(user_query)

            if "❌" in context:
                st.warning("No relevant results found.")
                st.code(context)
            else:
                genai_model = initialize_genai_model()
                answer = genai_model.generate_content(
                    f"""Based on the following listings, answer this query: "{user_query}"\n\n{context}"""
                ).text
                formatted = format_agent_output(answer)

                st.markdown("### 📄 Context")
                st.code(context)

                st.markdown("### 🤖 Insights")
                st.write(formatted)

elif option == "AI Agent":
    st.subheader("🤖 Ask the AI Agent (RAG + LLM Tools)")
    agent_query = st.text_input("Enter your business query (e.g., 'Find hybrid cars with good highway mileage')", key="agent_input")

    if st.button("Submit AI Agent Query"):
        with st.spinner("Letting the AI Agent reason and respond..."):
            task = Task(
                description=agent_query,
                expected_output="Business-friendly summary with reasoning and insights.",
                agent=carfax_agent
            )
            crew.tasks = [task]
            response = crew.kickoff()

            context = last_rag_context.get("content", "❌ Context not available")
            answer = response.get("raw") if isinstance(response, dict) else response

            formatted_answer = format_agent_output(answer)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📄 Context")
                st.code(context, language="markdown")

            with col2:
                st.markdown("### 🤖 Agent Insights")
                st.write(formatted_answer)
