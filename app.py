import streamlit as st
import rag

st.set_page_config(page_title="茶語時光客服", page_icon="🧋")
st.title("🧋 茶語時光 客服小幫手")
st.caption("可以問我菜單、價格、營業時間、會員、過敏原等問題")

with st.sidebar:
    st.header("設定")
    mode = st.radio(
        "RAG 模式",
        ["傳統 RAG", "Agentic RAG"],
        help="傳統 RAG：固定檢索一次再回答。Agentic RAG：Claude 自己決定要查什麼、查幾次。",
    )

    st.divider()
    st.header("管理")
    if st.button("重建索引"):
        with st.spinner("建立索引中..."):
            n = rag.build_index()
        st.success(f"完成，共 {n} 塊")
    if st.button("清除對話"):
        st.session_state.messages = []
        st.rerun()


def show_details(m: dict) -> None:
    """在回答下方顯示模式與參考資料／查詢路徑"""
    st.caption(f"模式：{m['mode']}")
    if m["mode"] == "傳統 RAG":
        with st.expander("查看參考資料"):
            for i, text in enumerate(m["chunks"], start=1):
                st.markdown(f"**[{i}]** {text}")
    else:
        with st.expander(f"查看查詢路徑（{len(m['trace'])} 次工具呼叫）"):
            for i, t in enumerate(m["trace"], start=1):
                arg = next(iter(t["input"].values()), "")
                st.markdown(f"**{i}. 🔧 `{t['tool']}`**　{arg}")
                st.text(t["output"])


if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            show_details(m)

if question := st.chat_input("請輸入你的問題"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("查詢中..."):
            if mode == "傳統 RAG":
                chunks = rag.retrieve_with_sections(question, top_k=3)
                answer = rag.generate(rag.build_prompt(question, chunks))
                reply = {"role": "assistant", "content": answer, "mode": mode,
                         "chunks": [c["text"] for c in chunks]}
            else:
                answer, trace = rag.agent_ask(question)
                reply = {"role": "assistant", "content": answer, "mode": mode, "trace": trace}

        st.markdown(answer)
        show_details(reply)

    st.session_state.messages.append(reply)