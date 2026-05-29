from __future__ import annotations

from pathlib import Path

from .data import DEFAULT_DATA_PATH, load_draws
from .features import feature_table
from .web.actions import create_web_report, generate_candidate_rows, pymc_summary_rows
from .web.charts import (
    build_frequency_chart_spec,
    blue_frequency,
    red_frequency,
    red_sum_distribution,
    span_distribution,
    to_plotly_bar,
)
from .web.state import ACADEMIC_NOTICE, build_dashboard_summary, data_quality_rows


def _render_chart(st, spec) -> None:
    chart = to_plotly_bar(spec)
    if hasattr(chart, "to_dict"):
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.bar_chart({spec.x_label: spec.x, spec.y_label: spec.y})


def run_streamlit_app() -> None:
    try:
        import streamlit as st  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Streamlit 未安装。请在 Mac 上运行 scripts/install_and_run_macos.sh 或 python3 -m pip install -e '.[web]'") from exc

    st.set_page_config(page_title="双色球概率建模研究", page_icon="📊", layout="wide")
    st.title("双色球概率建模研究系统")
    st.warning(ACADEMIC_NOTICE)

    with st.sidebar:
        st.header("数据与运行参数")
        data_path = st.text_input("历史数据 CSV", value=str(DEFAULT_DATA_PATH))
        if st.button("重新加载数据"):
            st.cache_data.clear()

    @st.cache_data(show_spinner=False)
    def cached_load(path: str):
        return load_draws(Path(path))

    try:
        draws = cached_load(data_path)
    except Exception as exc:  # pragma: no cover - UI error path
        st.error(f"数据读取失败：{exc}")
        st.stop()

    tab_dashboard, tab_stats, tab_pymc, tab_candidates, tab_reports = st.tabs(["首页", "数据与统计", "PyMC 贝叶斯", "候选组合", "报告中心"])

    with tab_dashboard:
        summary = build_dashboard_summary(draws, data_path)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("样本期数", summary["sample_count"])
        c2.metric("最新期号", summary["latest_issue"])
        c3.metric("理论组合空间", summary["total_combinations"])
        c4.metric("Apple Silicon", "是" if summary["is_apple_silicon"] else "否")
        st.subheader("环境状态")
        st.table([
            {"项目": "Python", "状态": summary["python_version"]},
            {"项目": "PyMC", "状态": "可用" if summary["pymc_available"] else "未安装"},
            {"项目": "JAX", "状态": "可用" if summary["jax_available"] else "未安装"},
            {"项目": "Streamlit", "状态": "可用" if summary["streamlit_available"] else "未安装"},
        ])
        st.subheader("数据质量")
        st.table(data_quality_rows(draws))

    with tab_stats:
        st.subheader("红球频率")
        _render_chart(st, build_frequency_chart_spec(red_frequency(draws), title="红球历史边际频率", x_label="红球", y_label="每期出现频率"))
        st.subheader("蓝球频率")
        _render_chart(st, build_frequency_chart_spec(blue_frequency(draws), title="蓝球历史频率", x_label="蓝球", y_label="出现频率"))
        st.subheader("和值与跨度")
        col_a, col_b = st.columns(2)
        with col_a:
            _render_chart(st, build_frequency_chart_spec(red_sum_distribution(draws), title="红球和值分布", x_label="和值", y_label="期数"))
        with col_b:
            _render_chart(st, build_frequency_chart_spec(span_distribution(draws), title="红球跨度分布", x_label="跨度", y_label="期数"))
        st.subheader("派生特征预览")
        st.dataframe(feature_table(draws)[-20:], use_container_width=True)

    with tab_pymc:
        st.subheader("PyMC 后验模型")
        quick = st.checkbox("快速模式：仅使用最近 500 期", value=True)
        alpha = st.number_input("alpha", min_value=0.01, value=1.0, step=0.1)
        beta = st.number_input("beta", min_value=0.01, value=1.0, step=0.1)
        st.caption("默认按钮输出共轭后验解析均值；完整 MCMC 可先在 CLI 中运行 python3 -m ssq_model pymc-fit。")
        if st.button("运行 PyMC 快速后验均值"):
            summary = pymc_summary_rows(draws, quick=quick, alpha=alpha, beta=beta)
            col_r, col_b = st.columns(2)
            with col_r:
                st.write("红球后验均值 Top 10")
                st.table(summary["red"])
            with col_b:
                st.write("蓝球后验均值 Top 5")
                st.table(summary["blue"])

    with tab_candidates:
        st.subheader("候选组合生成")
        model_name = st.selectbox("模型", options=["bayesian", "pymc", "ensemble"], index=1)
        top_k = st.slider("Top-K", min_value=1, max_value=100, value=20)
        seed = st.number_input("随机种子", value=20260529, step=1)
        pool_size = st.slider("候选池大小", min_value=100, max_value=10000, value=2000, step=100)
        if st.button("生成候选组合"):
            rows = generate_candidate_rows(draws, model_name=model_name, top_k=top_k, seed=int(seed), pool_size=pool_size)
            st.dataframe(rows, use_container_width=True)
            csv_text = "序号,红球,蓝球,score\n" + "\n".join(f"{r['序号']},{r['红球']},{r['蓝球']},{r['score']}" for r in rows)
            st.download_button("下载候选 CSV", data=csv_text.encode("utf-8-sig"), file_name="candidates.csv", mime="text/csv")

    with tab_reports:
        st.subheader("报告中心")
        output_dir = st.text_input("输出目录", value="outputs")
        if st.button("生成 Web 报告"):
            report = create_web_report(draws, output_dir)
            st.success(f"报告已生成：{report}")
            st.code(str(report))


def main() -> None:
    run_streamlit_app()


if __name__ == "__main__":
    main()
