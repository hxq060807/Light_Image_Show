#!/usr/bin/env python3
"""
16×16 纯方格瀑布：每小格 1 px 红边框，无坐标轴，最新帧在底
← → 逐行，底部常驻大输入框回车跳转
"""
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

FILE_PATH = r'E:\Smart_Car\WitSD\LOG00083.TXT'  # 改这里
HEAD      = b'\x7F\x80'
TAIL      = b'\x80\x7F'
FRAME_LEN = 2 + 16 + 2

# ---------- 读帧 ----------
def read_frames(path):
    data = pathlib.Path(path).read_bytes()
    i, frames = 0, []
    while i + FRAME_LEN <= len(data):
        if data[i:i+2] == HEAD and data[i+18:i+20] == TAIL:
            frames.append(data[i+2:i+18])
            i += FRAME_LEN
        else:
            i += 1
    return frames

frames = read_frames(FILE_PATH)
if not frames:
    print('没找到任何完整帧！')
    exit()
print(f'共解析到 {len(frames)} 帧')

# ---------- 滑动缓存（最新在底） ----------
class Slide:
    buf = np.zeros((16, 16), dtype=np.uint8)
    row = 0
slide = Slide()

def update_buf(target_row):
    top = max(0, target_row - 15)
    for k in range(16):
        src = top + k
        slide.buf[15 - k] = np.frombuffer(frames[src], dtype=np.uint8) * 255 if 0 <= src < len(frames) else 0
    slide.row = target_row

# ---------- 绘图（纯方格 + 单像素红边框） ----------
def draw():
    ax.clear()
    
    ax.imshow(slide.buf, cmap='gray', vmin=0, vmax=255,
              extent=[0, 16, 16, 0],
              aspect='equal', interpolation='nearest')
    
    for i in range(17):
        ax.plot([i, i], [0, 16], 'pink', linewidth=1, alpha=1.0, solid_capstyle='round')
    
    for i in range(17):
        ax.plot([0, 16], [i, i], 'pink', linewidth=1, alpha=1.0, solid_capstyle='round')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.suptitle(f'Row {slide.row + 1} / {len(frames)}', y=0.92)
    fig.canvas.draw()

# ---------- 键盘事件 ----------
def on_key(event):
    if event.inaxes == ax:
        if event.key == 'right' and slide.row < len(frames) - 1:
            update_buf(slide.row + 1)
            draw()
        elif event.key == 'left' and slide.row > 0:
            update_buf(slide.row - 1)
            draw()

def jump_to_row(val=None):
    """跳转到指定行 - 修改：添加默认值None，解决Button回调参数问题"""
    try:
        num = int(entry_var.get())
        if 1 <= num <= len(frames):
            update_buf(num - 1)
            draw()
            canvas.draw()
    except ValueError:
        pass

def on_key_type(key):
    """模拟按键事件"""
    if key == 'Right' and slide.row < len(frames) - 1:
        update_buf(slide.row + 1)
        draw()
        canvas.draw()
    elif key == 'Left' and slide.row > 0:
        update_buf(slide.row - 1)
        draw()
        canvas.draw()

# ---------- 创建Tkinter窗口 ----------
root = tk.Tk()
root.title("16x16 方格瀑布显示器")

# 固定窗口大小，禁止调整
root.geometry("1000x1100")
root.resizable(False, False)

# 创建matplotlib图形
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.05, top=0.9)

update_buf(0)
draw()

# 将matplotlib图形嵌入Tkinter
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=20, pady=20)

# 定义字体列表，按优先级尝试
def get_safe_font(font_list):
    """返回系统中可用的字体"""
    for font_name in font_list:
        try:
            # 测试字体是否可用
            test_label = tk.Label(root, text="test", font=(font_name, 12))
            test_label.destroy()
            return font_name
        except:
            continue
    return "TkDefaultFont"  # 回退到默认字体

# 定义不同用途的字体优先级列表
label_fonts = ['钉钉进步体', 'Microsoft YaHei', 'SimHei', '黑体', 'Arial Unicode MS']
entry_fonts = ['钉钉进步体', 'Microsoft YaHei', 'SimHei', '黑体', 'Consolas']  
button_fonts = ['钉钉进步体', 'Microsoft YaHei', 'SimHei', '黑体', 'Arial']
help_fonts = ['钉钉进步体', 'KaiTi', 'STKaiti', 'SimKai', 'Georgia']

# 获取安全的字体
safe_label_font = get_safe_font(label_fonts)
safe_entry_font = get_safe_font(entry_fonts)
safe_button_font = get_safe_font(button_fonts)
safe_help_font = get_safe_font(help_fonts)

print(f"使用字体: 标签={safe_label_font}, 输入框={safe_entry_font}, 按钮={safe_button_font}, 说明={safe_help_font}")

# 创建输入框架
input_frame = tk.Frame(root)
input_frame.pack(side=tk.BOTTOM, pady=30)

# 标签 - 更大字体
label = tk.Label(input_frame, text="跳转到行:", font=(safe_label_font, 16))
label.pack(side=tk.LEFT, padx=10)

# 输入框变量和控件 - 更大
entry_var = tk.StringVar()
entry = tk.Entry(input_frame, textvariable=entry_var, width=15, 
                font=(safe_entry_font, 16), bd=3)
entry.pack(side=tk.LEFT, padx=15)

# 跳转按钮 - 修改：直接使用jump_to_row，不需要lambda包装
jump_button = tk.Button(input_frame, text="跳转", command=jump_to_row, 
                       # 注意：现在可以直接使用jump_to_row，因为函数接受可选参数val=None
                       font=(safe_button_font, 16, 'bold'), width=8, height=1, bd=3,
                       bg='lightblue', cursor='hand2')
jump_button.pack(side=tk.LEFT, padx=15)

# 添加导航按钮框架
nav_frame = tk.Frame(root)
nav_frame.pack(side=tk.BOTTOM, pady=10)

# 左箭头按钮
left_button = tk.Button(nav_frame, text="← 上一行", command=lambda: on_key_type('Left'),
                        font=(safe_button_font, 14), width=12, height=1, bd=3, bg='lightgray')
left_button.pack(side=tk.LEFT, padx=20)

# 右箭头按钮  
right_button = tk.Button(nav_frame, text="下一行 →", command=lambda: on_key_type('Right'),
                         font=(safe_button_font, 14), width=12, height=1, bd=3, bg='lightgray')
right_button.pack(side=tk.LEFT, padx=20)

# 绑定键盘事件
def on_tk_key(event):
    if event.keysym == 'Right' and slide.row < len(frames) - 1:
        update_buf(slide.row + 1)
        draw()
        canvas.draw()
    elif event.keysym == 'Left' and slide.row > 0:
        update_buf(slide.row - 1)
        draw()
        canvas.draw()

root.bind('<Key>', on_tk_key)
entry.bind('<Return>', jump_to_row)  # 修改：也可以直接使用jump_to_row

# 设置焦点到主窗口而不是输入框
root.focus_set()

# 添加使用说明
help_label = tk.Label(root, 
                     text="使用说明: 点击输入框输入行号后按回车或点击跳转按钮 | 也可使用方向键或下方按钮导航", 
                     font=(safe_help_font, 13),
                     fg='darkblue',
                     bg='lightyellow',
                     relief='raised',
                     bd=2,
                     padx=10,
                     pady=5)
help_label.pack(pady=15)

root.mainloop()