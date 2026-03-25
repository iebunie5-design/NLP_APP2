"""
프로그램명: 한국어 영화 리뷰 감성 분석 앱 (두 모델 동시 비교)
사용법: streamlit run app_day2.py
필요 라이브러리: pip install streamlit scikit-learn kiwipiepy transformers torch plotly
"""

import re
import warnings
import streamlit as st
import numpy as np
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="영화 리뷰 감성 분석",
    page_icon="🎬",
    layout="wide",
)

EXAMPLE_REVIEWS = [
    ("😊", "이 영화 진짜 재미있어요 배우들 연기도 최고의 명작"),
    ("😞", "시간 아까웠다 최악의 영화 다시는 안 봄"),
    ("😐", "나쁘지 않았어요 그냥 평범한 영화"),
    ("🤔", "연기는 좋았는데 스토리가 너무 별로"),
    ("😵", "감독이 관객을 바보로 아는 것 같은 영화"),
]

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Apple SD Gothic Neo', sans-serif;
}
.app-header {
    padding: 1.6rem 0 1.2rem;
    border-bottom: 1px solid #f0f0f0;
    margin-bottom: 1.6rem;
}
.app-title { font-size: 1.75rem; font-weight: 700; color: #1a1a2e; letter-spacing: -0.5px; margin: 0; }
.app-subtitle { font-size: 0.86rem; color: #999; margin-top: 4px; }
.section-label {
    font-size: 0.73rem; font-weight: 700; color: #bbb;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 6px;
}
textarea {
    border-radius: 12px !important; border: 1.5px solid #e8e8ee !important;
    font-size: 0.95rem !important; line-height: 1.7 !important;
    padding: 12px 14px !important; background: #fafafa !important;
}
textarea:focus { border-color: #5c6bc0 !important; background: #fff !important; }
div[data-testid="stButton"] > button[kind="primary"] {
    background: #3d4eac; color: white; border: none;
    border-radius: 10px; padding: 0.55rem 2rem;
    font-size: 0.92rem; font-weight: 600;
}
div[data-testid="stButton"] > button[kind="primary"]:hover { background: #2e3d8f; }

/* 모델 결과 컬럼 카드 */
.model-card {
    background: white;
    border-radius: 16px;
    border: 1.5px solid #ebebf2;
    padding: 1.4rem 1.5rem;
    height: 100%;
}
.model-card-header {
    display: flex; align-items: center; gap: 10px;
    padding-bottom: 12px;
    border-bottom: 1px solid #f3f3f3;
    margin-bottom: 14px;
}
.model-badge {
    font-size: 0.75rem; font-weight: 700; padding: 3px 10px;
    border-radius: 20px; letter-spacing: 0.04em;
}
.badge-tfidf { background: #eef0fb; color: #3d4eac; }
.badge-hf    { background: #fff4e0; color: #a05a00; }
.model-card-name { font-size: 0.9rem; font-weight: 600; color: #333; }

/* 결과 배너 */
.result-banner {
    display: flex; align-items: center; gap: 14px;
    padding: 1rem 1.2rem; border-radius: 12px; margin-bottom: 10px;
}
.result-banner.pos { background: #f0faf3; border: 1.5px solid #6dbe8d; }
.result-banner.neg { background: #fff3f3; border: 1.5px solid #f08080; }
.result-icon { font-size: 1.9rem; line-height: 1; }
.result-label { font-size: 1.35rem; font-weight: 800; line-height: 1.2; }
.result-label.pos { color: #1a5c30; }
.result-label.neg { color: #7a1515; }
.result-meta { font-size: 0.8rem; color: #999; margin-top: 2px; }
.result-meta b.pos { color: #27a84a; font-size: 0.88rem; }
.result-meta b.neg { color: #d43b3b; font-size: 0.88rem; }

/* 차트 관련 */
.chart-title { font-size: 0.88rem; font-weight: 700; color: #444; margin: 12px 0 2px; }
.chart-caption { font-size: 0.75rem; color: #bbb; margin-bottom: 2px; }

/* 비교 요약 배너 */
.compare-banner {
    background: #f7f7fc;
    border: 1px solid #e0e0f0;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 1rem 0;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 20px;
    font-size: 0.88rem;
}
.compare-agree   { color: #3d4eac; font-weight: 700; }
.compare-disagree{ color: #c05000; font-weight: 700; }

.info-box {
    background: #eef1fb; border-left: 3px solid #5c6bc0;
    border-radius: 0 8px 8px 0; padding: 9px 13px;
    font-size: 0.82rem; color: #3d4eac; margin: 10px 0; line-height: 1.55;
}
.example-head {
    font-size: 0.73rem; font-weight: 700; color: #bbb;
    letter-spacing: 0.08em; text-transform: uppercase; margin: 1.6rem 0 8px;
}
div[data-testid="stButton"] > button[kind="secondary"] {
    background: #f7f7f9; border: 1.5px solid #e8e8ee;
    border-radius: 22px; font-size: 0.82rem; color: #555;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover {
    background: #eef0fb; border-color: #5c6bc0; color: #3d4eac;
}
[data-testid="stSidebar"] { background: #f8f8fc; }
.sb-card {
    background: white; border-radius: 12px; padding: 1rem 1.1rem;
    margin-bottom: 10px; border: 1px solid #ebebf2;
}
.sb-model { font-size: 0.86rem; font-weight: 700; color: #3d4eac; margin-bottom: 7px; }
.sb-row {
    display: flex; justify-content: space-between;
    font-size: 0.78rem; color: #777; padding: 3px 0;
    border-bottom: 1px solid #f5f5f5;
}
.sb-row:last-child { border-bottom: none; }
.sb-val { font-weight: 600; color: #333; }
hr { border: none; border-top: 1px solid #f0f0f0; margin: 1.2rem 0; }
.block-container { padding-top: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 모델 로딩
# ============================================================
@st.cache_resource(show_spinner="Kiwi 형태소 분석기 초기화 중...")
def load_kiwi():
    from kiwipiepy import Kiwi
    return Kiwi()


@st.cache_resource(show_spinner="TF-IDF 모델 학습 중... (최초 1회, 약 3~5분)")
def load_tfidf_model():
    import urllib.request
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    kiwi = load_kiwi()
    try:
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
            "ratings_train.txt"
        )
        df = pd.read_csv("ratings_train.txt", sep="\t").dropna(subset=["document"])
    except Exception as e:
        return None, None

    n = 10000
    df_s = pd.concat([
        df[df["label"] == 1].sample(n, random_state=42),
        df[df["label"] == 0].sample(n, random_state=42),
    ]).reset_index(drop=True)

    def preprocess(text):
        text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", str(text))
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""
        tokens = kiwi.tokenize(text)
        valid = {"NNG", "NNP", "VV", "VA", "MAG"}
        return " ".join(t.form for t in tokens if t.tag in valid and len(t.form) > 1)

    df_s["processed"] = df_s["document"].apply(preprocess)
    df_s = df_s[df_s["processed"].str.len() > 0]

    vec = TfidfVectorizer(max_features=5000, min_df=3, ngram_range=(1, 2))
    X   = vec.fit_transform(df_s["processed"])
    y   = df_s["label"].values
    mdl = LogisticRegression(max_iter=1000, random_state=42)
    mdl.fit(X, y)
    return vec, mdl


@st.cache_resource(show_spinner="Korean Sentiment 모델 다운로드 중... (최초 1회, 약 1~2분)")
def load_hf_model():
    try:
        from transformers import pipeline
        return pipeline("text-classification", model="matthewburke/korean_sentiment", top_k=None)
    except ImportError:
        return None


def preprocess_korean(text, kiwi):
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    tokens = kiwi.tokenize(text)
    valid = {"NNG", "NNP", "VV", "VA", "MAG"}
    return " ".join(t.form for t in tokens if t.tag in valid and len(t.form) > 1)


# ============================================================
# 차트 함수
# ============================================================
def draw_gauge(prob_pos: float, label: str):
    bar_color = "#27a84a" if label == "긍정" else "#d43b3b"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob_pos * 100, 1),
        number={"suffix": "%", "font": {"size": 40, "color": bar_color}},
        gauge={
            "axis": {
                "range": [0, 100], "tickwidth": 0,
                "tickvals": [0, 50, 100],
                "ticktext": ["0", "50", "100"],
                "tickfont": {"size": 10, "color": "#ccc"},
                "tickcolor": "rgba(0,0,0,0)",
            },
            "bar": {"color": bar_color, "thickness": 0.26},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [0, 50],   "color": "#fff0f0"},
                {"range": [50, 100], "color": "#f0faf3"},
            ],
            "threshold": {
                "line": {"color": "#ccc", "width": 2},
                "thickness": 0.82, "value": 50,
            },
        },
    ))
    fig.update_layout(
        height=190,
        margin=dict(t=14, b=0, l=24, r=24),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def draw_contribution_chart(text, vectorizer, model, kiwi):
    processed = preprocess_korean(text, kiwi)
    if not processed:
        return None, []

    vec_t         = vectorizer.transform([processed])
    feature_names = vectorizer.get_feature_names_out()
    coef          = model.coef_[0]

    contribs = []
    for idx in vec_t.nonzero()[1]:
        contribs.append((feature_names[idx], float(vec_t[0, idx] * coef[idx])))

    contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    top = contribs[:8]
    if not top:
        return None, []

    top_s  = sorted(top, key=lambda x: x[1])
    words  = [w[0] for w in top_s]
    values = [round(w[1], 4) for w in top_s]

    bar_colors = ["#27a84a" if v >= 0 else "#d43b3b" for v in values]
    bg_colors  = ["rgba(39,168,74,0.10)" if v >= 0 else "rgba(212,59,59,0.10)" for v in values]
    txt_colors = ["#1a5c30" if v >= 0 else "#7a1515" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=words, orientation="h",
        marker=dict(color=bg_colors, line=dict(color=bar_colors, width=1.5)),
        text=[f"{'+'if v>0 else ''}{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(size=10, color=txt_colors),
        cliponaxis=False,
    ))
    fig.add_vline(x=0, line_width=1.5, line_dash="dot", line_color="#ddd")
    fig.update_layout(
        height=max(len(top_s) * 44 + 36, 160),
        margin=dict(t=4, b=4, l=4, r=64),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.04)", zeroline=False, showticklabels=False),
        yaxis=dict(tickfont=dict(size=12, color="#444"), showgrid=False),
        showlegend=False, bargap=0.38,
    )
    return fig, top


# ============================================================
# 메인
# ============================================================
def main():

    # 헤더
    st.markdown("""
    <div class="app-header">
      <p class="app-title">🎬 영화 리뷰 감성 분석</p>
      <p class="app-subtitle">TF-IDF 로지스틱 회귀 &nbsp;vs&nbsp; Korean Sentiment — 두 모델 동시 비교</p>
    </div>
    """, unsafe_allow_html=True)

    if "review_text" not in st.session_state:
        st.session_state.review_text = ""

    # ── 입력 영역 ──────────────────────────────────────────
    st.markdown('<p class="section-label">리뷰 입력</p>', unsafe_allow_html=True)
    review_input = st.text_area(
        "review", value=st.session_state.review_text, height=90,
        placeholder="예)  이 영화 진짜 재밌어요. 배우 연기도 최고!",
        label_visibility="collapsed",
    )

    col_btn, _ = st.columns([1.2, 5])
    with col_btn:
        analyze = st.button("분석하기", use_container_width=True, type="primary")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 분석 실행 ──────────────────────────────────────────
    if analyze and review_input.strip():

        kiwi = load_kiwi()

        # ① TF-IDF 예측
        vectorizer, lr_model = load_tfidf_model()
        tfidf_ok = vectorizer is not None

        processed = preprocess_korean(review_input, kiwi) if tfidf_ok else ""

        if tfidf_ok and processed:
            vec_t       = vectorizer.transform([processed])
            tfidf_pred  = lr_model.predict(vec_t)[0]
            tfidf_prob  = lr_model.predict_proba(vec_t)[0][1]
        else:
            tfidf_pred, tfidf_prob = 0, 0.5

        # ② HuggingFace 예측
        hf_model = load_hf_model()
        hf_ok    = hf_model is not None

        if hf_ok:
            result   = hf_model(review_input[:512])[0]
            scores   = {r["label"]: r["score"] for r in result}
            hf_prob  = scores.get("LABEL_1", 0)
            hf_pred  = 1 if hf_prob > 0.5 else 0
        else:
            hf_pred, hf_prob = 0, 0.5

        # ── 비교 요약 배너 ──────────────────────────────────
        t_label = "긍정" if tfidf_pred == 1 else "부정"
        h_label = "긍정" if hf_pred   == 1 else "부정"
        agree   = t_label == h_label

        if agree:
            summary_cls  = "compare-agree"
            summary_text = f"두 모델 모두 <b>{t_label}</b>으로 판단했습니다"
        else:
            summary_cls  = "compare-disagree"
            summary_text = f"모델 의견 불일치 — TF-IDF: <b>{t_label}</b> &nbsp;/&nbsp; Korean Sentiment: <b>{h_label}</b>"

        st.markdown(f"""
        <div class="compare-banner">
          <span class="{summary_cls}">{summary_text}</span>
        </div>
        """, unsafe_allow_html=True)

        # ── 두 컬럼 나란히 표시 ─────────────────────────────
        col_left, col_right = st.columns(2, gap="medium")

        # ───── 왼쪽: TF-IDF ──────────────────────────────
        with col_left:
            st.markdown("""
            <div class="model-card-header">
              <span class="model-badge badge-tfidf">TF-IDF</span>
              <span class="model-card-name">로지스틱 회귀</span>
            </div>
            """, unsafe_allow_html=True)

            if tfidf_ok and processed:
                cls  = "pos" if tfidf_pred == 1 else "neg"
                icon = "😊" if tfidf_pred == 1 else "😞"
                proc_disp = processed[:36] + ("…" if len(processed) > 36 else "")

                st.markdown(f"""
                <div class="result-banner {cls}">
                  <div class="result-icon">{icon}</div>
                  <div>
                    <div class="result-label {cls}">{t_label}</div>
                    <div class="result-meta">
                      긍정 확률 <b class="{cls}">{tfidf_prob*100:.1f}%</b>
                      &nbsp;·&nbsp; {proc_disp}
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                st.plotly_chart(draw_gauge(tfidf_prob, t_label), use_container_width=True)

                st.markdown('<p class="chart-title">📊 판단 근거 단어</p>', unsafe_allow_html=True)
                st.markdown('<p class="chart-caption">초록: 긍정 기여 &nbsp;/&nbsp; 빨강: 부정 기여</p>', unsafe_allow_html=True)

                fig_bar, top_words = draw_contribution_chart(review_input, vectorizer, lr_model, kiwi)
                if fig_bar:
                    st.plotly_chart(fig_bar, use_container_width=True)
                    with st.expander("상세 수치"):
                        for word, contrib in top_words:
                            d = "긍정↑" if contrib > 0 else "부정↓"
                            bar = "█" * int(abs(contrib) * 22)
                            st.code(f"{word:12s}  {contrib:+.4f} ({d})  {bar}")
                else:
                    st.info("분석 가능한 단어가 없습니다.")
            else:
                st.error("TF-IDF 모델 로딩 실패 또는 분석 가능한 단어 없음")

        # ───── 오른쪽: Korean Sentiment ──────────────────
        with col_right:
            st.markdown("""
            <div class="model-card-header">
              <span class="model-badge badge-hf">HuggingFace</span>
              <span class="model-card-name">Korean Sentiment</span>
            </div>
            """, unsafe_allow_html=True)

            if hf_ok:
                cls  = "pos" if hf_pred == 1 else "neg"
                icon = "😊" if hf_pred == 1 else "😞"

                st.markdown(f"""
                <div class="result-banner {cls}">
                  <div class="result-icon">{icon}</div>
                  <div>
                    <div class="result-label {cls}">{h_label}</div>
                    <div class="result-meta">
                      긍정 확률 <b class="{cls}">{hf_prob*100:.1f}%</b>
                      &nbsp;·&nbsp; 문맥 전체 이해
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                st.plotly_chart(draw_gauge(hf_prob, h_label), use_container_width=True)

                st.markdown("""
                <div class="info-box">
                  사전학습 모델로 <b>문장 전체 맥락</b>을 이해합니다.<br>
                  단어별 기여도는 제공되지 않습니다 (블랙박스 모델).
                </div>
                """, unsafe_allow_html=True)

                # 확률 차이 시각화
                prob_diff = abs(tfidf_prob - hf_prob)
                if prob_diff > 0.1 and tfidf_ok:
                    st.markdown(f"""
                    <div class="info-box" style="background:#fff8ee;border-color:#e8a020;color:#a05a00;">
                      두 모델의 확률 차이: <b>{prob_diff*100:.1f}%p</b><br>
                      {'Korean Sentiment 가 더 확신 — 문맥 이해 덕분일 수 있습니다.' if hf_prob > tfidf_prob else 'TF-IDF 가 더 확신 — 특정 단어 패턴에 민감한 경우입니다.'}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("transformers 패키지가 설치되지 않았습니다.")
                st.code("python -m pip install transformers torch")

    elif analyze and not review_input.strip():
        st.warning("리뷰를 입력해주세요.")

    # ── 예시 리뷰 ──────────────────────────────────────────
    st.markdown('<p class="example-head">예시 리뷰로 테스트</p>', unsafe_allow_html=True)
    cols = st.columns(len(EXAMPLE_REVIEWS))
    for i, (col, (emoji, review)) in enumerate(zip(cols, EXAMPLE_REVIEWS)):
        with col:
            if st.button(f"{emoji} {review[:8]}…", key=f"ex_{i}", use_container_width=True):
                st.session_state.review_text = review
                st.rerun()

    with st.expander("예시 리뷰 전체 보기"):
        for emoji, review in EXAMPLE_REVIEWS:
            st.write(f"{emoji}  {review}")

    # ── 사이드바 ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 모델 비교")
        st.markdown("""
        <div class="sb-card">
          <div class="sb-model">TF-IDF + 로지스틱 회귀</div>
          <div class="sb-row"><span>학습 데이터</span><span class="sb-val">NSMC 2만건</span></div>
          <div class="sb-row"><span>N-gram</span><span class="sb-val">uni+bigram</span></div>
          <div class="sb-row"><span>속도</span><span class="sb-val">⚡ 빠름</span></div>
          <div class="sb-row"><span>정확도</span><span class="sb-val">약 77%</span></div>
          <div class="sb-row"><span>기여도 확인</span><span class="sb-val">✅ 가능</span></div>
          <div class="sb-row"><span>문맥 이해</span><span class="sb-val">❌ 불가</span></div>
        </div>
        <div class="sb-card">
          <div class="sb-model">Korean Sentiment</div>
          <div class="sb-model" style="color:#a05a00">Korean Sentiment</div>
          <div class="sb-row"><span>모델</span><span class="sb-val">HuggingFace</span></div>
          <div class="sb-row"><span>구조</span><span class="sb-val">사전학습</span></div>
          <div class="sb-row"><span>속도</span><span class="sb-val">🐢 느림</span></div>
          <div class="sb-row"><span>정확도</span><span class="sb-val">약 85%+</span></div>
          <div class="sb-row"><span>기여도 확인</span><span class="sb-val">❌ 불가</span></div>
          <div class="sb-row"><span>문맥 이해</span><span class="sb-val">✅ 가능</span></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.caption("Day 2 실습  ·  NLP 감성 분석")


if __name__ == "__main__":
    main()