# Translation Guide (EN → 简体中文)

This guide is for translating the TensorMesh documentation from English (canonical source) into Simplified Chinese using the Sphinx gettext workflow. It covers the recommended scope, the FEM glossary, and an LLM prompt template you can drop straight into Claude / GPT.

## Workflow

```
1. Stabilize English .rst content (and Python docstrings, if you want the
   API reference translated — see "Scope" below).
2. cd docs && make intl-update-prose    # only refreshes prose .po files
3. Open docs/source/locale/zh_CN/LC_MESSAGES/**/*.po and fill in `msgstr`
   entries — feed each file to an LLM with the prompt template below.
4. Run the CJK-spacing pass (see "CJK + inline-markup spacing" below) so
   inline markup doesn't break against adjacent Chinese characters.
5. cd docs && make zh && ./serve.sh     # build Chinese site, review
6. Iterate on any awkward phrasings.
7. Commit the .po files. Deploy.
```

When the English source changes later, run `make intl-update-prose` again — `sphinx-intl` preserves existing translations and marks changed entries as `fuzzy` (you'll see `#, fuzzy` comments in the .po), so you only need to re-review those.

## Automated translation (recommended)

Steps 3–4 above (filling `msgstr`, then the CJK-spacing pass) are automated by `docs/scripts/translate_po.py`. It translates **only empty or `fuzzy`** entries with Claude — using the glossary and markup rules from this guide — then runs the escaped-space pass, so it never overwrites a human translation and is safe to re-run after every `make intl-update-prose`.

```bash
pip install -r requirements-translate.txt   # anthropic + polib
export ANTHROPIC_API_KEY=sk-ant-...

make intl-translate                          # refresh prose .po, then translate the gaps
# …or drive the engine directly:
python scripts/translate_po.py --dry-run     # list the gaps, call nothing
python scripts/translate_po.py               # translate all prose catalogs
python scripts/translate_po.py --files getting_started/index.po
```

It skips `api/` and `_archive/` automatically (same scope as below). Always finish with `make zh` and the quality checklist — the ZH build must add **zero** warnings over the EN baseline. The manual prompt-template path further down remains the reference for the glossary and the exact rules the script encodes.

**Provider.** The engine auto-detects the backend from whichever API key is set (or pass `--provider`):

- **Claude** (default) — `claude-sonnet-4-6`, set `ANTHROPIC_API_KEY` (`pip install anthropic`). Uses strict tool use for guaranteed-valid JSON.
- **Kimi** — Moonshot `kimi-k2.6` (strong for Chinese), set `MOONSHOT_API_KEY` (`pip install openai`; Kimi is OpenAI-API-compatible). Override the model with `KIMI_MODEL`, the endpoint with `MOONSHOT_BASE_URL` (e.g. `https://api.moonshot.cn/v1` for the China site), and optionally `KIMI_TEMPERATURE`. Kimi can't force a tool call, so this path uses plain JSON output with tolerant parsing.

## Scope

The `.po` files cover **every** translatable string Sphinx finds — including Python docstrings extracted by autodoc. There are 72 `.po` files total:

| Section | Files | Recommended |
|---|---|---|
| `index.po` | 1 | Translate |
| `getting_started/*.po` | 4 | Translate |
| `user_guide/*.po` | 10 | Translate |
| `example_gallery/*.po` | 21 | Translate |
| `performance/*.po` | 4 | Translate |
| `community/*.po` | 6 | Translate |
| `api/*.po` | 14 | **Skip** (leave `msgstr ""` — falls back to English) |
| `example_gallery/_archive/*.po` | 12 | **Skip** (archived/unpublished examples) |

**Why skip `api/`:** these are auto-extracted from Python docstrings. Translating them (a) is several times more work than translating the prose pages, (b) creates a maintenance burden every time a docstring changes, (c) most Chinese-speaking devs prefer to read API docs in English so the terminology lines up with the code they're calling. PyTorch's Chinese community follows this same pattern.

The `make intl-update-prose` target (in `docs/Makefile`) refreshes the prose sections only (`index`, `getting_started/`, `user_guide/`, `example_gallery/`, `performance/`, `community/`) — it does not touch `api/*.po`.

## Glossary

Use these consistently across all 46 prose `.po` files. Mixing translations of the same FEM term (e.g. "element" as both 单元 and 元素) reads as unprofessional.

| English | 中文 | Notes |
|---|---|---|
| finite element method (FEM) | 有限元方法 | |
| mesh | 网格 | |
| element | 单元 | NOT 元素 — that's "element" in CS sense |
| node | 节点 | |
| cell | 单元 / 网格单元 | Same as element in most contexts |
| facet / face | 面 | 3D face of a cell |
| edge | 边 | |
| connectivity | 连接性 / 单元拓扑 | |
| boundary | 边界 | |
| boundary condition (BC) | 边界条件 | |
| Dirichlet BC | 狄利克雷边界条件 | |
| Neumann BC | 诺伊曼边界条件 | |
| Robin BC | 罗宾边界条件 | |
| degree of freedom (DOF) | 自由度 | |
| basis function | 基函数 | |
| shape function | 形函数 | |
| weak form | 弱形式 | |
| variational form | 变分形式 | |
| assembly | 装配 | |
| assembler | 装配器 | |
| stiffness matrix | 刚度矩阵 | |
| mass matrix | 质量矩阵 | |
| load vector | 载荷向量 | |
| source term | 源项 | |
| residual | 残差 | |
| Jacobian | 雅可比矩阵 / 雅可比 | |
| gradient | 梯度 | |
| divergence | 散度 | |
| curl | 旋度 | |
| strain | 应变 | |
| stress | 应力 | |
| displacement | 位移 | |
| pressure | 压力 / 压强 | |
| sparse matrix | 稀疏矩阵 | |
| dense matrix | 稠密矩阵 | |
| solver | 求解器 | |
| backend | 后端 | |
| condensation (static) | 静态凝聚 | |
| condenser | 凝聚器 | |
| quadrature | 求积 / 数值积分 | |
| quadrature point | 求积点 / 高斯点 | |
| reference element | 参考单元 | |
| transformation | 变换 / 映射 | Geometric mapping |
| triangle / quadrilateral | 三角形 / 四边形 | |
| tetrahedron | 四面体 | |
| hexahedron | 六面体 | |
| prism / pyramid | 三棱柱 / 四棱锥 | |
| line element | 线单元 | |
| time integration | 时间积分 | |
| time step | 时间步 / 步长 | |
| explicit / implicit | 显式 / 隐式 | |
| ODE / PDE | 常微分方程 / 偏微分方程 | |
| Runge-Kutta | 龙格-库塔 | |
| Newton-Raphson | 牛顿-拉夫森 / 牛顿迭代 | |
| Poisson equation | 泊松方程 | |
| heat equation | 热方程 | |
| wave equation | 波动方程 | |
| linear elasticity | 线弹性 | |
| hyperelasticity | 超弹性 | |
| plasticity | 塑性 | |
| Neo-Hookean | 新胡克 | |
| J2 plasticity | J2 塑性 | |
| contact | 接触 | |
| topology optimization | 拓扑优化 | |
| inverse problem | 反问题 | |
| differentiable | 可微 / 可微分 | |
| automatic differentiation | 自动微分 | |
| forward / backward pass | 前向 / 反向传播 | |
| GPU acceleration | GPU 加速 | |
| batch | 批次 / 批量 | Keep "batch" untranslated when used as noun in code context |

**Untranslated on purpose** (keep in English): `PyTorch`, `NumPy`, `SciPy`, `gmsh`, `meshio`, `vmap`, `autograd`, `nn.Module`, `Tensor`, etc. — and obviously all class / function names.

## LLM Prompt Template

Paste this verbatim, then append the contents of one `.po` file at a time (or several at once if they fit in context). One file at a time gives more reliable output.

```
You are translating a Sphinx gettext .po file for TensorMesh, a PyTorch-based
finite element method (FEM) library, from English to Simplified Chinese.

Output ONLY the modified .po file content, no preamble, no explanation, no
markdown fences. Preserve every line that isn't a `msgstr`.

Rules:
1. Translate the value of each `msgstr ""` to Simplified Chinese. Leave `msgid`
   lines and all metadata (`#:`, `#,`, etc.) untouched.
2. Preserve reStructuredText / Sphinx inline markup verbatim, including the
   exact backtick / colon / asterisk count:
     - ``code``, ``:class:`Mesh```, ``:func:`assemble```, ``:doc:`installation```
     - `*italic*`, `**bold**`
     - `:math:`...`` and any LaTeX inside
     - URLs and `<links>`
3. Do not translate:
     - class / function / module names: Mesh, ElementAssembler, tensormesh.mesh,
       SparseMatrix, Condenser, etc.
     - PyTorch / NumPy / SciPy / Python API names
     - Code snippets (anything that looks like Python)
     - File paths and CLI commands
4. If a `msgid` is purely a code identifier or a single API name, the `msgstr`
   should be IDENTICAL to the `msgid` (no translation).
5. Use this glossary for consistency. The English term in column 1 must always
   map to the Chinese term in column 2:

   finite element method → 有限元方法
   mesh → 网格
   element → 单元   (NEVER 元素)
   node → 节点
   facet / face → 面
   degree of freedom → 自由度
   basis function → 基函数
   shape function → 形函数
   weak form → 弱形式
   assembly → 装配
   assembler → 装配器
   stiffness matrix → 刚度矩阵
   mass matrix → 质量矩阵
   sparse matrix → 稀疏矩阵
   dense matrix → 稠密矩阵
   Dirichlet boundary condition → 狄利克雷边界条件
   Neumann boundary condition → 诺伊曼边界条件
   strain / stress → 应变 / 应力
   displacement → 位移
   gradient / divergence / curl → 梯度 / 散度 / 旋度
   quadrature → 求积  (quadrature point → 求积点 or 高斯点)
   solver → 求解器  (backend → 后端)
   condensation → 静态凝聚  (condenser → 凝聚器)
   triangle / quadrilateral / tetrahedron / hexahedron / prism / pyramid →
     三角形 / 四边形 / 四面体 / 六面体 / 三棱柱 / 四棱锥
   ODE / PDE → 常微分方程 / 偏微分方程
   time integration → 时间积分
   explicit / implicit → 显式 / 隐式
   Runge-Kutta → 龙格-库塔
   Newton-Raphson → 牛顿-拉夫森
   Poisson / heat / wave equation → 泊松方程 / 热方程 / 波动方程
   linear elasticity → 线弹性  (hyperelasticity → 超弹性)
   plasticity → 塑性  (Neo-Hookean → 新胡克;  J2 plasticity → J2 塑性)
   topology optimization → 拓扑优化
   inverse problem → 反问题
   differentiable → 可微  (automatic differentiation → 自动微分)
   GPU acceleration → GPU 加速

6. Translate technical prose naturally — don't translate word-by-word. Aim for
   Chinese that a native FEM researcher would actually write in a paper.
7. Keep `PyTorch`, `NumPy`, `SciPy`, `gmsh`, `meshio`, `vmap`, `autograd`,
   `nn.Module`, `Tensor` and similar library / API names in English.
8. CJK spacing: wherever inline markup (``code``, **bold**, *em*,
   :role:`x`, :math:`...`, [Key]_) would end up directly touching a Chinese
   character or full-width punctuation, separate them with an escaped space —
   written `\\ ` (backslash-backslash-space) in the .po, which renders as
   nothing. E.g. msgstr "\\ ``Mesh``\\ 对象". Never nest a :role: inside
   **bold** (RST forbids it and the reference is lost). See "CJK +
   inline-markup spacing" in the guide.

Here is the .po file to translate:

<<<paste .po file content here>>>
```

## CJK + inline-markup spacing (the `\ ` escaped-space rule)

reStructuredText only recognises inline markup when its delimiters are bordered by whitespace or ASCII punctuation. A **Chinese character or full-width punctuation mark sitting directly against a markup delimiter** breaks that rule, so constructs like `` ``代码``中 ``, `**粗体**。`, or `` :class:`Mesh`的 `` are silently *not* parsed. Sphinx then emits `Inline ... start-string without end-string`, `reference target not found`, or `inconsistent term references` warnings — a fresh ZH build produced ~200 of these before this fix was applied.

The fix is to separate the markup from the adjacent CJK character with an **escaped space** `\ ` (a backslash followed by a space). It is a valid markup boundary but renders as nothing — no visible gap.

> **`.po` encoding caveat.** A literal backslash must be doubled inside a `.po` string, so to get the RST `\ ` you write **`\\ `** (two backslashes + a space) in the `msgstr`. Example:
>
> ```
> msgid  "the ``Mesh`` object"
> msgstr "\\ ``Mesh``\\ 对象"      # renders: Mesh对象  (no stray backslash, no gap)
> ```

Three cases the bulk rule does **not** cover; fix these by hand:

* **Two markup tokens back-to-back** — e.g. an English `**Label.** ``code`` ` whose separating space got dropped in translation, giving `` **标签。**``code`` ``. Put a `\ ` between them.
* **A role nested inside `**bold**`** — RST forbids nesting, so `` **:class:`Foo` 的说明** `` loses the reference entirely. Move the role *outside* the bold: `` :class:`Foo` 的 **说明** ``.
* **Citation / footnote refs `[Key]_`** whose `[` touches a full-width bracket: `` ）[Key]_ `` → `` ）\ [Key]_ ``.

### Post-processing helper

Catching every adjacency by hand is error-prone. After filling in the `.po` files, run this once to insert all the needed `\ ` automatically (needs `polib`: `pip install polib`). Run it from the `docs/` directory:

```python
import polib, re, glob

# Inline-markup tokens (role/math first so their interior isn't rescanned).
TOKEN = re.compile(
    r':[\w+-]+:`[^`]*`'        # :role:`...` and :math:`...`
    r'|``[^`]+``'              # ``literal``
    r'|`[^`]+`__?|`[^`]+`'     # `phrase`(opt _/__)
    r'|\*\*[^*]+\*\*'          # **strong**
    r'|\*[^*\s][^*]*\*'        # *emphasis*
)

def wide(c):                   # CJK ideographs + CJK / full-width punctuation
    return bool(c) and ord(c) >= 0x2E80

def fix(text):
    def repl(m):
        s, full, i, j = m.group(0), m.string, m.start(), m.end()
        pre = '\\ ' if i > 0 and wide(full[i - 1]) else ''
        suf = '\\ ' if j < len(full) and wide(full[j]) else ''
        return pre + s + suf
    return TOKEN.sub(repl, text)

for path in glob.glob('source/locale/zh_CN/LC_MESSAGES/**/*.po', recursive=True):
    if '/api/' in path or '/_archive/' in path:
        continue
    po = polib.pofile(path)
    for e in po:
        if e.msgstr:
            e.msgstr = fix(e.msgstr)
    po.save(path)
```

It only ever rewrites `msgstr` values (never `msgid`), is idempotent, and leaves math/code interiors untouched. Then rebuild and verify with the checklist below — the ZH build must add **zero** warnings over the EN baseline.

## After Translation: Quality Checklist

After running `make zh` and serving the Chinese site, spot-check:

- [ ] **The `make zh` build adds zero warnings over the `make html` baseline.** Build both with `-a -E -q` and diff the warning lists; any ZH-only warning (`Inline … start-string without end-string`, `reference target not found`, `inconsistent term references`) is almost always a CJK-adjacency miss — apply the escaped-space rule above.
- [ ] Code blocks render unchanged (Python should never be translated)
- [ ] Cross-references like `:class:`Mesh`` still link correctly (broken links = malformed inline markup)
- [ ] Math equations render (LaTeX inside `:math:` was preserved)
- [ ] No mixed half-English-half-Chinese paragraphs (those are `msgstr ""` entries the LLM missed; re-run on those files)
- [ ] FEM terminology is consistent across pages (no "单元" on one page and "元素" on another)
- [ ] Page navigation, search, "Next/Previous" labels work (these come from Sphinx's built-in zh_CN locale, no translation needed on your part)

## Updating Translations Later

When you update the English `.rst` files:

```bash
cd docs
make intl-update-prose    # diffs source → marks changed entries as fuzzy
                          # in the existing .po files; existing translations
                          # of unchanged strings are preserved.
```

Then grep for `#, fuzzy` in the `.po` files — those are the entries that need re-translation. You can either:
- Manually update the `msgstr` and remove the `#, fuzzy` line, or
- Re-feed the whole file to the LLM (it'll re-translate all entries; existing good translations get overwritten with equivalent ones, which is usually fine).
