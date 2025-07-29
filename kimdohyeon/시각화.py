import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🔸 문제별 BERTScore F1 데이터 (동일)
f1_scores_all = {
    "Q2": {
        "RAG vs LLM": {
            ("gpt-4o", "gpt-4o"): 0.6755,
            ("gpt-4o", "gpt-3.5"): 0.6833,
            ("gpt-4o", "claude-opus-4"): 0.6882,
            ("gpt-4o", "claude-opus-3"): 0.6880,
            ("gpt-3.5", "gpt-4o"): 0.6713,
            ("gpt-3.5", "gpt-3.5"): 0.6783,
            ("gpt-3.5", "claude-opus-4"): 0.6849,
            ("gpt-3.5", "claude-opus-3"): 0.6764,
            ("claude-opus-4", "gpt-4o"): 0.6869,
            ("claude-opus-4", "gpt-3.5"): 0.7029,
            ("claude-opus-4", "claude-opus-4"): 0.7063,
            ("claude-opus-4", "claude-opus-3"): 0.6925,
            ("claude-opus-3", "gpt-4o"): 0.6774,
            ("claude-opus-3", "gpt-3.5"): 0.6733,
            ("claude-opus-3", "claude-opus-4"): 0.6881,
            ("claude-opus-3", "claude-opus-3"): 0.6891,
        }
    },
    "Q3": {
        "RAG vs LLM": {
            ("gpt-4o", "gpt-4o"): 0.7281,
            ("gpt-4o", "gpt-3.5"): 0.7340,
            ("gpt-4o", "claude-opus-4"): 0.6600,
            ("gpt-4o", "claude-opus-3"): 0.7091,
            ("gpt-3.5", "gpt-4o"): 0.6164,
            ("gpt-3.5", "gpt-3.5"): 0.6453,
            ("gpt-3.5", "claude-opus-4"): 0.5616,
            ("gpt-3.5", "claude-opus-3"): 0.5926,
            ("claude-opus-4", "gpt-4o"): 0.6678,
            ("claude-opus-4", "gpt-3.5"): 0.6741,
            ("claude-opus-4", "claude-opus-4"): 0.6540,
            ("claude-opus-4", "claude-opus-3"): 0.6869,
            ("claude-opus-3", "gpt-4o"): 0.6887,
            ("claude-opus-3", "gpt-3.5"): 0.7098,
            ("claude-opus-3", "claude-opus-4"): 0.6703,
            ("claude-opus-3", "claude-opus-3"): 0.7154,
        }
    },
    "Q4": {
        "RAG vs LLM": {
            ("gpt-4o", "gpt-4o"): 0.7929,
            ("gpt-4o", "gpt-3.5"): 0.7885,
            ("gpt-4o", "claude-opus-4"): 0.7247,
            ("gpt-4o", "claude-opus-3"): 0.7064,
            ("gpt-3.5", "gpt-4o"): 0.7306,
            ("gpt-3.5", "gpt-3.5"): 0.7069,
            ("gpt-3.5", "claude-opus-4"): 0.6746,
            ("gpt-3.5", "claude-opus-3"): 0.6664,
            ("claude-opus-4", "gpt-4o"): 0.6971,
            ("claude-opus-4", "gpt-3.5"): 0.6693,
            ("claude-opus-4", "claude-opus-4"): 0.6872,
            ("claude-opus-4", "claude-opus-3"): 0.6836,
            ("claude-opus-3", "gpt-4o"): 0.7201,
            ("claude-opus-3", "gpt-3.5"): 0.6967,
            ("claude-opus-3", "claude-opus-4"): 0.6780,
            ("claude-opus-3", "claude-opus-3"): 0.6833,
        }
    }
}

# 데이터프레임 생성 함수
def create_filtered_df(f1_data: dict) -> pd.DataFrame:
    rows = []
    for question, sections in f1_data.items():
        for section, scores in sections.items():
            for (m1, m2), score in scores.items():
                rows.append({
                    "Question": question,
                    "Section": section,
                    "Model1": m1,
                    "Model2": m2,
                    "F1": score
                })
    return pd.DataFrame(rows)

# 데이터프레임 생성
df_filtered = create_filtered_df(f1_scores_all)

# 시각화
sns.set(style="whitegrid")
g = sns.catplot(
    data=df_filtered,
    kind="bar",
    x="Model1", y="F1", hue="Model2",
    col="Question",
    height=5, aspect=1.4,
    palette="Set3"
)
g.set_titles("{col_name} - RAG vs LLM")
g.set_axis_labels("기준 모델", "BERTScore F1")
g._legend.set_title("비교 모델")

# y축 범위 조정 및 막대 위에 값 표시
for ax in g.axes.flat:
    ax.set_ylim(0.55, 0.82)  # 타이트하게 설정
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.002,
                f"{height:.3f}",
                ha='center',
                va='bottom',
                fontsize=8,
                rotation=90
            )

plt.tight_layout()
plt.show()
