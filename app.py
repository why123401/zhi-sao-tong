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

if "processing" not in st.session_state:
    st.session_state["processing"] = False

# 渲染历史消息
for message in st.session_state["message"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入提示词
prompt = st.chat_input(disabled=st.session_state["processing"])

if prompt and not st.session_state["processing"]:
    # 标记正在处理，禁用输入框
    st.session_state["processing"] = True

    # 1. 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    # 2. 创建助手回复占位符，流式输出
    with st.chat_message("assistant"):
        assistant_placeholder = st.empty()
        full_response = ""

        try:
            for chunk in st.session_state["agent"].execute_stream(prompt):
                full_response += chunk
                # 实时更新占位符，实现打字机效果
                assistant_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"抱歉，服务暂时不可用，请稍后重试。（错误: {e}）"
            assistant_placeholder.error(full_response)

    # 3. 保存完整回复到历史消息
    st.session_state["message"].append({"role": "assistant", "content": full_response})

    # 4. 重置处理标记
    st.session_state["processing"] = False

    # 5. 重新渲染页面，固定显示最终结果
    st.rerun()
