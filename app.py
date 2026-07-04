import streamlit as st

from agent.react_agent import ReactAgent

# 标题
st.title("智扫通机器人智能客服")
st.divider()

# 初始化 session_state
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

# 渲染历史消息
for message in st.session_state["message"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    # 1. 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    # 2. 使用 st.chat_message + st.empty 实现流式打字效果
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            for chunk in st.session_state["agent"].execute_stream(prompt):
                full_response += chunk
                # 实时更新 markdown 内容
                placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"抱歉，服务暂时不可用，请稍后重试。（错误: {e}）"
            placeholder.error(full_response)

    # 3. 保存完整回复到历史消息
    st.session_state["message"].append({"role": "assistant", "content": full_response})
    st.rerun()
