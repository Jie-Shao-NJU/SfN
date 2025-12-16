import textwrap
import matplotlib.pyplot as plt

# 设置画布与适度扩大尺寸
fig, axs = plt.subplots(2, 2, figsize=(14, 7.2))  # 更大尺寸
plt.subplots_adjust(wspace=0.08, hspace=0.25)     # 稍宽间距

levels = [
    ("Level I: Nonsense",
     "Random tokens, and no structure.",
     "Q: Every morning, Aya does a 9 kilometer walk, and then finishes at the coffee shop. One day, she walks at s kilometers per hour, and the walk takes 4 hours, including $t$ minutes at the coffee shop. ... This morning, if she walks at s+1/2 kilometers per hour, how many minutes will the walk take?\n\n"
     "A: fjewl fj39qj qwokzz fj@#%... I can't think of a specific situation where I can't..."),

    ("Level II: Off-topic but Fluent",
     "Fluent sentence, but unrelated to context.",
     "Q: Every morning, Aya does a 9 kilometer walk, and then finishes at the coffee shop. One day, she walks at s kilometers per hour, and the walk takes 4 hours, including $t$ minutes at the coffee shop. ... This morning, if she walks at s+1/2 kilometers per hour, how many minutes will the walk take?\n\n"
     "A: Can you provide me with some data?"),

    ("Level III: In-context but Incorrect",
     "Context-aware, but incorrect answer.",
     "Q: Every morning, Aya does a 9 kilometer walk, and then finishes at the coffee shop. One day, she walks at s kilometers per hour, and the walk takes 4 hours, including $t$ minutes at the coffee shop. ... This morning, if she walks at s+1/2 kilometers per hour, how many minutes will the walk take?\n\n"
     "A: First, let's set up the equations...\n"
     "1. \( s(4 - \\frac{t}{60}) = 9 \)\n"
     "2. \( (s + 2)(2.4 - \\frac{t}{60}) = 9 \)\n"
     "→ Simplify and solve:\n"
     "144s - st - 2t = 252\n"
     "4s - \\frac{st}{60} = 9\n"
     "...\n"
     "**Final answer**\n"
     "\\boxed{12}"),

    ("Level IV: Correct Reasoning",
     "Context-aware and logically correct.",
     "Q: Every morning, Aya does a 9 kilometer walk, and then finishes at the coffee shop. One day, she walks at s kilometers per hour, and the walk takes 4 hours, including $t$ minutes at the coffee shop. ... This morning, if she walks at s+1/2 kilometers per hour, how many minutes will the walk take?\n\n"
     "A: ...\n"
     "9/s = 4 - t/60,\n"
     "9/(s + 2) = 2.4 - t/60\n"
     "→ Subtract:\n"
     "9(1/s - 1/(s + 2)) = 1.6 ⇒ s(s + 2) = 11.25 ⇒ s = 2.5\n"
     "Then: 3.6 + t/60 = 4 ⇒ t = 24\n"
     "Today: speed = 3 km/h → walk = 180 min, total = 180 + 24 = \\boxed{204} minutes.")
]

# 背景颜色（红→黄→蓝→绿）
colors = ['#fee0e0', '#fff9cc', '#dceefa', '#e2f7e2']

# 自动换行工具
def wrap_text(text, width):
    return textwrap.fill(text, width=width)

# Q / A 加粗换行处理（视觉强调 + 排版优化）
def format_qa(text):
    text = text.replace("Q:", "\n\nQ:").replace("A:", "\n\nA:")
    return textwrap.fill(text, width=105)

# 绘制四宫格内容
for i, (ax, (title, desc, example)) in enumerate(zip(axs.flat, levels)):
    ax.set_facecolor(colors[i])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    # 边框样式
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_color('#888888')

    # 标题
    ax.text(0.02, 0.92, title,
            transform=ax.transAxes, fontsize=13, fontweight='bold', ha='left', va='top')

    # 描述（斜体）
    ax.text(0.02, 0.82, wrap_text(desc, 65),
            transform=ax.transAxes, fontsize=11, style='italic', ha='left', va='top')

    # 示例（自动换行并加粗 Q/A）
    ax.text(0.02, 0.64, format_qa(example),
            transform=ax.transAxes, fontsize=10.7, ha='left', va='top', linespacing=1.35)

# 总标题
fig.suptitle("Examples of LLM Merge Output Levels", fontsize=16.5, fontweight='bold', y=1.04)

# 保存图像
plt.savefig("merge_stethoscope_grid_desc_qa_wrapped.png", dpi=300, bbox_inches='tight')
# plt.show()
