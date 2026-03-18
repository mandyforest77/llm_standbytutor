import streamlit as st
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

st.set_page_config(page_title="Email Drafter", layout="wide")
# 1. 도구 정의
def draft_email(content: str) -> str:
    """작성된 이메일 초안을 최종적으로 반환합니다."""
    return f"\n[최종 드래프트 결과]\n\n{content}"

# 4. Streamlit UI
st.title("📧 Email Drafter")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_area("받은 이메일 내용을 적어주세요", key="input_email")
user_need = st.text_area("원하는 답장 방향를 적어주세요", key="input_need")

def run_process(prompt_text):
    st.session_state.messages.append(HumanMessage(content=prompt_text))
    with st.spinner("AI가 이메일을 작성 중입니다..."):
        inputs = {"messages": st.session_state.messages}
        response = graph.invoke(inputs)
        # 전체 메시지 히스토리 업데이트 (도구 실행 결과 포함)
        st.session_state.messages = response["messages"]

if "user_api" in st.session_state:
    
    tools = [draft_email]
    tool_node = ToolNode(tools)

    # 2. 모델 설정 (tool_choice 설정으로 도구 사용 강제)
    model = ChatOpenAI(model="gpt-4o", temperature=0.7, openai_api_key=st.session_state["user_api"]).bind_tools(tools, tool_choice="required")

    # 3. 그래프 상태 및 노드 정의
    class AgentState(TypedDict):
        messages: Annotated[list[BaseMessage], lambda x, y: x + y]

    def agent_node(state: AgentState):
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", END)
    graph = workflow.compile()
    
    if st.button("이메일 초안 작성"):
        if user_input and user_need:
            prompt = f"""아래 정보를 바탕으로 정중한 이메일 초안을 작성해줘.
            반드시 draft_email 도구를 호출해서 content 인자에 작성한 이메일 전문을 넣어 제출해.

            [받은 이메일]
            {user_input}

            [답장 방향]
            {user_need}"""
            run_process(prompt)
        else:
            st.warning("내용과 방향을 모두 입력해주세요.")


    # 결과 출력 로직
    if st.session_state.messages:
        st.divider()
        st.subheader("작성된 초안")

        # 마지막 메시지들 중 ToolMessage(도구 실행 결과)를 찾아 내용 표시
        final_draft = ""
        for msg in reversed(st.session_state.messages):
            if isinstance(msg, ToolMessage):
                # 도구 함수가 반환한 문자열 추출
                final_draft = msg.content
                break

        if final_draft:
            st.info(final_draft)
        else:
            st.write("초안을 생성하는 중입니다...")

        # 피드백 섹션
        st.divider()
        feedback = st.text_input("수정 사항이 있나요?", placeholder="예: 좀 더 캐주얼한 톤으로 바꿔줘")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("수정 반영하기"):
                if feedback:
                    run_process(f"다음 수정 사항을 반영해서 다시 draft_email을 호출해줘: {feedback}")
                    st.rerun()
        with col2:
            if st.button("새로 만들기"):
                st.session_state.messages = []
                st.rerun()

# 2. 다운로드 버튼 추가
        with col3:
            st.download_button(
                label="텍스트 파일 다운로드",
                data=final_draft,
                file_name="draft.txt",
                mime="text/plain"
            )

else:
    st.info("API Key가 등록되지 않았습니다. 첫 페이지에서 등록해주세요.")
    
