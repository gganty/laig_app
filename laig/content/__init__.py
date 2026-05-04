from __future__ import annotations
import re
import streamlit as st


_REGISTRY: dict[str, dict] = {}


def register(items: list[dict]) -> None:
    for item in items:
        _REGISTRY[item["id"]] = item


def get_registry() -> dict[str, dict]:
    return _REGISTRY


def lookup(item_id: str) -> dict | None:
    return _REGISTRY.get(item_id)


LINK_RE = re.compile(r'\[(d|t):(\d+\.\d+)\]')


def linkify(text: str, registry: dict) -> str:
    def replace(match: re.Match) -> str:
        kind = match.group(1)
        num = match.group(2)
        item_id = f"{kind}:{num}"
        item = registry.get(item_id)
        prefix = "опр." if kind == "d" else "теор."
        if item is None:
            return f"_{prefix} {num}_"
        name = item.get("name", "")
        return f"_{prefix} {num} ({name})_"

    return LINK_RE.sub(replace, text)


def render_definition(d: dict, registry: dict) -> None:
    num = d["id"].split(":")[1]
    name = d["name"]

    with st.container(border=True):
        st.markdown(f"**{num}. {name}**")

        notation = d.get("notation", "")
        if notation:
            with st.expander("📖 Обозначения", expanded=False):
                st.markdown(linkify(notation, registry))

        statement = d.get("statement", "")
        if statement:
            st.markdown(linkify(statement, registry))

        remark = d.get("remark", "")
        if remark:
            with st.expander("💬 Замечание", expanded=False):
                st.markdown(linkify(remark, registry))


def render_theorem(t: dict, registry: dict) -> None:
    num = t["id"].split(":")[1]
    name = t["name"]

    st.markdown(f"#### {num}. {name}")
    st.markdown(linkify(t.get("statement", ""), registry))

    if t.get("idea"):
        with st.expander("💡 Идея доказательства", expanded=False):
            st.markdown(linkify(t["idea"], registry))

    if t.get("deps"):
        with st.expander("🔗 Зависит от", expanded=False):
            for dep_id in t["deps"]:
                dep = registry.get(dep_id)
                if dep:
                    kind = "опр." if dep_id.startswith("d:") else "теор."
                    dep_num = dep_id.split(":")[1]
                    st.markdown(f"- {kind} {dep_num}: **{dep['name']}**")
                else:
                    st.markdown(f"- {dep_id} (не найдено в реестре)")

    if t.get("proof"):
        with st.expander("📜 Полное доказательство", expanded=False):
            st.markdown(linkify(t["proof"], registry))
    elif t.get("idea"):
        st.caption("⏳ Полное доказательство в процессе. Пока есть только идея — см. выше.")

    if t.get("pitfalls"):
        with st.expander("⚠️ Подводные камни", expanded=False):
            st.markdown(linkify(t["pitfalls"], registry))


def filter_items(items: list[dict], query: str) -> list[dict]:
    if not query:
        return items
    q = query.lower()
    result = []
    for item in items:
        haystacks = [
            item.get("name", ""),
            item.get("statement", ""),
            item.get("notation", ""),
            item.get("remark", ""),
        ]
        if any(q in h.lower() for h in haystacks):
            result.append(item)
    return result
