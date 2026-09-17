# 网页版 GUI

`doublespike-gui.html` 是一个**单文件、完全离线**的双稀释剂工具箱界面：直接双击用浏览器打开即可，
不需要装 Python、不需要联网、不会上传任何数据。

主界面分三块：

| 区域 | 用途 |
|---|---|
| 左侧控制面板 | 选元素、改标样组成、挑反演同位素、搭稀释剂、设误差模型 |
| 「精度图」页 | 误差随混合比 / 稀释剂组成的变化曲线，以及 (p, q) 二维误差图 |
| 「最优稀释剂」页 | 穷举当前反演同位素下所有稀释剂对，按误差排序，点击即应用 |
| 「数据反演」页 | 粘贴实测束流 → 求 α、β、稀释剂比例；含蒙特卡洛验证 |
| 「原理与算法」页 | 公式、算法说明与性能对比 |

右上角「自检」按钮会运行 13 项内置检查（数值与 Python 库比对、四个页面渲染、示例数据反演），
用于确认当前浏览器里这份拷贝工作正常。

## 目录结构

```
webgui/
├── doublespike-gui.html     ← 构建产物，单文件，直接打开这个
├── template.html            ← 带占位符的页面骨架
├── build.py                 ← 把下面 4 个文件塞进 template，产出上面那个 html
├── make_data.py             ← 从 doublespike 生成 data.js 与 reference.json
└── src/
    ├── style.css            样式（深/浅两套主题）
    ├── data.js              39 个同位素体系的标样丰度、原子质量、可用单稀释剂
    ├── math.js              双稀释剂全部数值算法（误差传播 + 反演），Python 的逐行移植
    ├── app.js               界面逻辑与自绘图表
    └── reference.json       Python 库算出的参考值，供数值对照测试使用
```

## 重新构建

修改 `src/` 下任何文件后：

```bash
python webgui/build.py           # 只需要 Python 标准库
```

如果改了同位素数据（`src/doublespike/data/maininput.csv`）或改了 Python 库的数值算法，
需要先重新生成数据与参考值（这一步要用到 `doublespike` 本身）：

```bash
PYTHONPATH=src python webgui/make_data.py
python webgui/build.py
```

## 数值一致性

网页版的算法是 `doublespike` Python 包的逐行移植。`webgui/make_data.py` 会用 Python 算出
一批参考值（9 个体系的误差传播与反演），`webgui/test_math.js` 用 Node 把两边的结果
逐一比对：

```bash
node webgui/test_math.js
```

全部检查通过，相对偏差优于 10⁻¹²。

## 实现说明

* 图表是手写的 canvas 绘制，没有引入任何图表库，所以整个页面只有一个文件、零依赖。
* 「计算中」提示与渲染之间用 `setTimeout` 让出主线程；**没有**用 `requestAnimationFrame`，
  因为它在无头浏览器里不会触发、在后台标签页里会被完全节流，会导致页面永远不渲染。
* 二维误差图的分辨率可以在左侧「图形设置」里调。160×160 ≈ 2.6 万个点，在现代浏览器上
  大约零点几秒；200 以上会明显变慢。
* 二维图对整张图固定使用同一个分母同位素。分母的选择不影响物理结果，固定下来是为了让
  图上不会出现因为分母切换而产生的接缝。
