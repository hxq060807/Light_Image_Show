#!/usr/bin/env python3
"""
16×16 纯方格瀑布：每小格 1 px 红边框，无坐标轴，最新帧在底
← → 逐行，底部常驻大输入框回车跳转
新增：右侧显示状态信息 - 与网格精准对齐
终极优化版 - 修复边框显示问题
"""
import pathlib
import struct  
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
import time

# === 终极优化：使用numba JIT编译 ===
from numba import jit, prange, njit
import numba

# === 添加：抑制字体警告 ===
def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    if "findfont: Font family" in str(message) and "not found" in str(message):
        return
    warnings._showwarning_orig(message, category, filename, lineno, file, line)

warnings._showwarning_orig = warnings.showwarning
warnings.showwarning = custom_showwarning

FILE_PATH = r'E:\Smart_Car\WitSD\LOG00159.TXT'      # 💡修改为你的文件路径
HEAD      = b'\x7F\x80'
TAIL      = b'\x80\x7F'
FRAME_LEN = 2 + 18 + 2

# 状态映射 - 使用整数编码避免字符串问题
STATUS_NORMAL = 0
STATUS_LEFT = 1  
STATUS_RIGHT = 2
STATUS_STRAIGHT = 3
STATUS_GO_LEFT = 4
STATUS_GO_RIGHT = 5

# 状态文本映射函数 - 在Python层面处理
def get_status_text(status_code):
    status_map = {
        0: "Normal",
        1: "Left", 
        2: "Right",
        3: "Straight",
        4: "Go_Left", 
        5: "Go_Right"
    }
    return status_map.get(status_code, f"Unknown({status_code})")

# ---------- 优化的帧读取函数 ----------
def read_frames_optimized(path):
    """优化的帧读取函数"""
    data_bytes = pathlib.Path(path).read_bytes()
    data = np.frombuffer(data_bytes, dtype=np.uint8)
    
    start_time = time.time()
    frames = []
    i = 0
    data_len = len(data)
    
    while i + FRAME_LEN <= data_len:
        if data[i:i+2].tobytes() == HEAD and data[i+20:i+22].tobytes() == TAIL:
            frame_content = data_bytes[i+2:i+20]  # 直接使用bytes切片
            frames.append(frame_content)
            i += FRAME_LEN
        else:
            i += 1
    
    end_time = time.time()
    print(f"🚀 帧读取完成: {len(frames)} 帧, 耗时: {(end_time-start_time)*1000:.2f}ms")
    return frames

# ---------- 滑动缓存（最新在底） ----------
class Slide:
    buf = np.zeros((16, 16), dtype=np.uint8)
    row = 0
    status_data = []
slide = Slide()

# ---------- 终极优化：预计算所有常量 ----------
GRID_Y_POSITIONS = np.array([14.8, 13.9, 13.0, 12.1, 11.2, 10.3, 9.4, 8.5, 7.6, 6.7, 5.8, 4.9, 4.0, 3.1, 2.2, 1.3], dtype=np.float32)
STATUS_CODE_FONTS = ['钉钉进步体', 'Courier New', 'Monaco', '等线', 'Arial']
STATUS_TEXT_FONTS = ['钉钉进步体', 'Microsoft YaHei', 'SimHei', '黑体', 'Arial Unicode MS']
VALUE_FONTS = ['钉钉进步体', 'Arial Black', '微软雅黑', 'Microsoft YaHei', 'SimHei']

# ---------- numba加速的图像处理 ----------
@njit(nopython=True, fastmath=True, cache=True)
def process_single_frame(image_data, status_byte, error_byte):
    """处理单帧图像的numba函数"""
    processed_image = np.zeros(16, dtype=np.uint8)
    
    # 处理图像数据
    for i in range(16):
        if image_data[i] == 1:
            processed_image[i] = 255
        else:
            processed_image[i] = 0
    
    # 处理状态字节
    status_code = status_byte
    
    # 处理错误值 - 使用位运算优化
    if error_byte >= 128:  # 负数
        error_value = error_byte - 256
    else:  # 正数
        error_value = error_byte
    
    return processed_image, status_code, error_value

def process_images_batch_optimized(frames_batch, target_row):
    """优化的批处理函数"""
    batch_size = len(frames_batch)
    buf = np.zeros((16, 16), dtype=np.uint8)
    status_data = []
    
    for display_row in range(16):
        frame_idx = target_row - display_row
        
        if 0 <= frame_idx < batch_size:
            frame_content = frames_batch[frame_idx]
            image_data = np.frombuffer(frame_content[:16], dtype=np.uint8)
            
            # 使用numba处理单帧
            status_byte = frame_content[16] if len(frame_content) > 16 else 0
            error_byte = frame_content[17] if len(frame_content) > 17 else 0
            
            processed_image, status_code, error_value = process_single_frame(
                image_data, status_byte, error_byte
            )
            
            buf[display_row] = processed_image
            
            # Python层面处理状态文本
            status_text = get_status_text(status_code)
            status_data.append((status_code, status_text, error_value))
        else:
            buf[display_row, :] = 0
            status_data.append((0, "Normal", 0))
    
    return buf, status_data

# ---------- 预分配状态数据数组 ----------
def update_buf_ultimate(target_row):
    """终极优化的缓冲区更新函数"""
    start_time = time.time()
    
    # 使用优化的批处理
    buf, status_data = process_images_batch_optimized(frames, target_row)
    
    # 更新slide对象
    slide.buf = buf.copy()
    slide.status_data = status_data
    slide.row = target_row
    
    end_time = time.time()

# 为了保持接口一致，创建别名
update_buf = update_buf_ultimate

# ---------- 字体管理类 ----------
class FontManager:
    def __init__(self, root):
        self.root = root
        self.custom_fonts = {}
        self.cached_fonts = {}  # 字体缓存
        
    def add_custom_font(self, font_path, font_name=None):
        try:
            import matplotlib.font_manager as fm
            fm.fontManager.addfont(font_path)
            prop = fm.FontProperties(fname=font_path)
            actual_name = prop.get_name()
            self.custom_fonts[font_name or actual_name] = font_path
            print(f"✓ 成功加载自定义字体: {actual_name}")
            return actual_name
        except Exception as e:
            print(f"✗ 加载字体失败 {font_path}: {e}")
            return None
    
    def get_safe_font(self, font_list, size, weight='normal'):
        cache_key = (tuple(font_list), size, weight)
        if cache_key in self.cached_fonts:
            return self.cached_fonts[cache_key]
            
        for font_name in font_list:
            try:
                test_label = tk.Label(self.root, text="test", font=(font_name, size))
                test_label.destroy()
                result = (font_name, size, weight)
                self.cached_fonts[cache_key] = result
                return result
            except:
                continue
        result = ("TkDefaultFont", size, weight)
        self.cached_fonts[cache_key] = result
        return result

# ---------- 修复边框问题的绘图类 ----------
class FixedBorderPlotter:
    def __init__(self):
        self.grid_created = False
        self.last_ax_main = None
        
    def draw_with_fixed_borders(self, fig, slide_buf, slide_status_data, current_row, total_rows):
        """修复边框显示问题的绘图函数"""
        # 清除整个图形
        fig.clear()
        
        # 创建22列的子图：16列图像 + 6列状态信息
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(1, 2, width_ratios=[16, 6], figure=fig)
        
        # 左侧：16×16图像
        ax_main = fig.add_subplot(gs[0])
        self.last_ax_main = ax_main  # 保存引用
        
        # 性能优化：使用更快的图像显示参数
        ax_main.imshow(slide_buf, cmap='gray', vmin=0, vmax=255,
                       extent=[0, 16, 16, 0],
                       origin='upper',
                       aspect='equal', 
                       interpolation='nearest',
                       animated=True)
        
        # 关键修复：每次都重新创建或确保网格线可见
        self.ensure_gridlines_exist(ax_main)
        
        # 关闭坐标轴
        ax_main.set_xticks([])
        ax_main.set_yticks([])
        for spine in ax_main.spines.values():
            spine.set_visible(False)
        ax_main.set_title('16×16 Image', fontsize=12, pad=10)
        
        # === 主标题：在16×16网格正上方居中显示 ===
        main_title_text = f'Row {current_row + 1} / {total_rows}'
        
        ax_main.text(8, -1.2, main_title_text,
                    transform=ax_main.transData,
                    ha='center', va='bottom',
                    fontsize=14, fontweight='bold',
                    color='darkblue',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                             edgecolor='orange', alpha=0.8))
        
        # 右侧：状态信息
        ax_status = fig.add_subplot(gs[1])
        ax_status.axis('off')
        
        # === 右侧状态栏标题 ===
        status_title_text = f'Run_Mode/Error'
        ax_status.text(3, 16, status_title_text,
                      transform=ax_status.transData,
                      ha='center', va='bottom',
                      fontsize=14, fontweight='bold',
                      color='darkblue',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                               edgecolor='orange', alpha=0.8))
        
        ax_status.set_xlim(0, 6)
        ax_status.set_ylim(0, 16)
        
        # 使用预计算的位置
        status_y_positions = GRID_Y_POSITIONS - 0.1
        
        # 绘制当前状态信息
        for idx in range(min(16, len(slide_status_data))):
            status_code, status_text, value = slide_status_data[idx]
            y_pos = status_y_positions[idx]
            
            # 状态代码
            ax_status.text(0.1, y_pos, f"{status_code}", fontproperties=status_code_font_prop,
                          verticalalignment='center', horizontalalignment='left')
            
            # 状态文本
            ax_status.text(0.5, y_pos, status_text, fontproperties=status_text_font_prop,
                          verticalalignment='center', horizontalalignment='left')
            
            # 数值
            ax_status.text(2.5, y_pos, f"{value}", fontproperties=value_font_prop, color='red',
                          verticalalignment='center', horizontalalignment='left')
            
            # 分隔线
            if idx < 15:
                grid_line_y = GRID_Y_POSITIONS[idx] - 0.4
                ax_status.plot([0, 6], [grid_line_y, grid_line_y], 'k-', linewidth=0.5, alpha=0.3)
        
        # 使用draw_idle而不是draw
        fig.canvas.draw_idle()
    
    def ensure_gridlines_exist(self, ax_main):
        """确保网格线存在且可见 - 关键修复"""
        # 方法1：尝试找到现有的网格线
        existing_grid_lines = []
        for child in ax_main.get_children():
            if isinstance(child, plt.Line2D):
                xdata = child.get_xdata()
                ydata = child.get_ydata()
                # 检查是否是网格线（长度为16的直线）
                if (len(xdata) == 2 and len(ydata) == 2 and 
                    (abs(xdata[1] - xdata[0] - 16.0) < 0.1 or abs(ydata[1] - ydata[0] - 16.0) < 0.1)):
                    existing_grid_lines.append(child)
        
        if existing_grid_lines:
            # 方法2：使用现有的网格线，确保它们可见
            print("🔍 找到现有网格线，确保可见...")
            for line in existing_grid_lines:
                line.set_alpha(1.0)
                line.set_visible(True)
                line.set_linewidth(1.0)
        else:
            # 方法3：重新创建所有网格线
            print("🎯 重新创建网格线...")
            # 先清除可能存在的旧线条
            for child in ax_main.get_children():
                if isinstance(child, plt.Line2D):
                    child.remove()
            
            # 创建新的网格线
            for i in range(17):
                # 创建垂直线
                ax_main.plot([i, i], [0, 16], 'pink', linewidth=1, alpha=1.0, solid_capstyle='round')
                # 创建水平线  
                ax_main.plot([0, 16], [i, i], 'pink', linewidth=1, alpha=1.0, solid_capstyle='round')
        
        self.grid_created = True

# 创建绘图器实例
fixed_plotter = FixedBorderPlotter()

# ---------- 优化的绘图函数 ----------
def draw_ultimate():
    """终极优化的绘图函数 - 使用固定边框绘图器"""
    start_time = time.time()
    
    # 使用修复边框的绘图器
    fixed_plotter.draw_with_fixed_borders(
        fig, slide.buf, slide.status_data, slide.row, len(frames)
    )
    
    end_time = time.time()
    # print(f"🎨 绘图完成, 耗时: {(end_time-start_time)*1000:.2f}ms")

# 为了保持接口一致，创建别名
draw = draw_ultimate

# ---------- 优化的事件处理函数 ----------
def on_key_optimized(event):
    if hasattr(event, 'inaxes') and event.inaxes == ax_main:
        if event.key == 'right' and slide.row < len(frames) - 1:
            update_buf(slide.row + 1)
            draw()
        elif event.key == 'left' and slide.row > 0:
            update_buf(slide.row - 1)
            draw()

def jump_to_row_optimized(val=None):
    try:
        num = int(entry_var.get())
        if 1 <= num <= len(frames):
            update_buf(num - 1)
            draw()
    except ValueError:
        pass

def on_key_type_optimized(key):
    if key == 'Right' and slide.row < len(frames) - 1:
        update_buf(slide.row + 1)
        draw()
    elif key == 'Left' and slide.row > 0:
        update_buf(slide.row - 1)
        draw()

# 为了保持接口一致，创建别名
on_key = on_key_optimized
jump_to_row = jump_to_row_optimized
on_key_type = on_key_type_optimized

# ---------- 创建Tkinter窗口 ----------
root = tk.Tk()
root.title("16x16 方格瀑布显示器 - 终极优化版（修复边框显示）")

# 固定窗口大小
root.geometry("1300x1100")
root.resizable(False, False)

# 初始化字体管理器
font_manager = FontManager(root)

# 获取状态栏字体（使用缓存）
status_code_font_tuple = font_manager.get_safe_font(STATUS_CODE_FONTS, 14, 'bold')
status_text_font_tuple = font_manager.get_safe_font(STATUS_TEXT_FONTS, 10)
value_font_tuple = font_manager.get_safe_font(VALUE_FONTS, 14, 'bold')

# 转换为matplotlib字体属性
import matplotlib.font_manager as fm
status_code_font_prop = fm.FontProperties(family=status_code_font_tuple[0], 
                                         size=status_code_font_tuple[1], 
                                         weight=status_code_font_tuple[2])
status_text_font_prop = fm.FontProperties(family=status_text_font_tuple[0], 
                                         size=status_text_font_tuple[1])
value_font_prop = fm.FontProperties(family=value_font_tuple[0], 
                                   size=value_font_tuple[1], 
                                   weight=value_font_tuple[2])

print(f"状态栏字体: 代码={status_code_font_tuple[0]}, 文本={status_text_font_tuple[0]}, 数值={value_font_tuple[0]}")

# 创建matplotlib图形
fig = plt.figure(figsize=(12, 8))
plt.subplots_adjust(bottom=0.05, top=0.90)

# 加载所有帧
print("🚀 开始加载和预处理所有帧...")
start_total = time.time()

try:
    frames = read_frames_optimized(FILE_PATH)
except Exception as e:
    print(f"❌ 优化读取失败，使用基础方法: {e}")
    # 备用方案：基础读取方法
    data = pathlib.Path(FILE_PATH).read_bytes()
    frames = []
    i = 0
    while i + FRAME_LEN <= len(data):
        if data[i:i+2] == HEAD and data[i+20:i+22] == TAIL:
            frame_content = data[i+2:i+20]
            frames.append(frame_content)
            i += FRAME_LEN
        else:
            i += 1

print(f"✅ 总加载时间: {(time.time()-start_total):.2f}秒")

if not frames:
    print('没找到任何完整帧！')
    exit()
print(f'共解析到 {len(frames)} 帧')

# 预计算第一帧
update_buf(0)
draw()

# 获取当前的axes（需要在draw之后获取）
ax_main = fig.axes[0]

# 将matplotlib图形嵌入Tkinter
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=10, pady=10)

# 定义主界面字体列表
label_fonts = ['钉钉进步体', 'Microsoft YaHei', 'SimHei', '黑体', 'Arial Unicode MS']
entry_fonts = ['钉钉进步体', 'Microsoft YaHei', 'SimHei', '黑体', 'Consolas']  
button_fonts = ['钉钉进步体', 'Microsoft YaHei', 'SimHei', '黑体', 'Arial']
help_fonts = ['钉钉进步体', 'KaiTi', 'STKaiti', 'SimKai', 'Georgia']

# 获取主界面安全字体
safe_label_font = font_manager.get_safe_font(label_fonts, 16)
safe_entry_font = font_manager.get_safe_font(entry_fonts, 16)
safe_button_font = font_manager.get_safe_font(button_fonts, 16, 'bold')
safe_help_font = font_manager.get_safe_font(help_fonts, 14)

# 创建输入框架
input_frame = tk.Frame(root)
input_frame.pack(side=tk.BOTTOM, pady=20)

# 标签
label = tk.Label(input_frame, text="跳转到行:", font=safe_label_font)
label.pack(side=tk.LEFT, padx=10)

# 输入框
entry_var = tk.StringVar()
entry = tk.Entry(input_frame, textvariable=entry_var, width=15, 
                font=safe_entry_font, bd=3)
entry.pack(side=tk.LEFT, padx=15)

# 跳转按钮
jump_button = tk.Button(input_frame, text="跳转", command=jump_to_row, 
                       font=safe_button_font, width=8, height=1, bd=3,
                       bg='lightblue', cursor='hand2')
jump_button.pack(side=tk.LEFT, padx=15)

# 添加导航按钮框架
nav_frame = tk.Frame(root)
nav_frame.pack(side=tk.BOTTOM, pady=10)

# 左箭头按钮
left_button = tk.Button(nav_frame, text="← 上一行", command=lambda: on_key_type('Left'),
                        font=safe_button_font, width=12, height=1, bd=3, bg='lightgray')
left_button.pack(side=tk.LEFT, padx=20)

# 右箭头按钮  
right_button = tk.Button(nav_frame, text="下一行 →", command=lambda: on_key_type('Right'),
                         font=safe_button_font, width=12, height=1, bd=3, bg='lightgray')
right_button.pack(side=tk.LEFT, padx=20)

# 绑定键盘事件
def on_tk_key_optimized(event):
    if event.keysym == 'Right' and slide.row < len(frames) - 1:
        update_buf(slide.row + 1)
        draw()
    elif event.keysym == 'Left' and slide.row > 0:
        update_buf(slide.row - 1)
        draw()

root.bind('<Key>', on_tk_key_optimized)
entry.bind('<Return>', jump_to_row)

# 设置焦点到主窗口而不是输入框
root.focus_set()

# 添加使用说明
font_info = f"状态栏字体→ 代码:{status_code_font_tuple[0]} 文本:{status_text_font_tuple[0]} 数值:{value_font_tuple[0]}"
help_label = tk.Label(root, 
                     text=f"🚀使用说明: 点击输入框输入行号后按回车或点击跳转按钮 | 也可使用方向键或下方按钮导航", 
                     font=safe_help_font,
                     fg='darkblue',
                     bg='lightyellow',
                     relief='raised',
                     bd=2,
                     padx=10,
                     pady=5)
help_label.pack(pady=10)

root.mainloop()

# 程序结束时恢复原始警告设置
warnings.showwarning = warnings._showwarning_orig