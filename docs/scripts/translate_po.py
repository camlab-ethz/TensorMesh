#!/usr/bin/env python3
"""Auto-fill untranslated / fuzzy prose ``.po`` entries (EN -> zh_CN) via Claude.

This automates step 3 of ``docs/TRANSLATION_GUIDE.md`` — instead of pasting each
catalog into an LLM by hand, it walks the Sphinx gettext catalogs under
``source/locale/<lang>/LC_MESSAGES/``, finds entries whose ``msgstr`` is empty or
marked ``#, fuzzy``, translates them with the guide's FEM glossary + RST-markup
rules, writes the result back, and runs the CJK escaped-space post-process.

It is **incremental and idempotent**: only empty/fuzzy entries are touched, so a
re-run after ``make intl-update-prose`` translates just the newly-changed
strings (existing human translations are never overwritten). ``api/`` (autodoc
docstrings) and ``_archive/`` (archived examples) are skipped, matching the
guide's scope.

Usage (run from ``docs/``)::

    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/translate_po.py                 # translate every prose catalog
    python scripts/translate_po.py --dry-run       # list the gaps, call nothing
    python scripts/translate_po.py --files getting_started/index.po user_guide/meshes.po
    python scripts/translate_po.py --no-cjk-fix    # skip the spacing pass

Or via the Makefile (refreshes the catalogs first, then translates)::

    make intl-translate

Deps: ``pip install -r requirements-translate.txt`` (anthropic + polib).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import polib

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DOCS_DIR = Path(__file__).resolve().parent.parent          # docs/
LOCALE_DIR = DOCS_DIR / "source" / "locale" / "zh_CN" / "LC_MESSAGES"

# Skip auto-extracted API docstrings and archived examples (see TRANSLATION_GUIDE.md).
SKIP_PARTS = ("api", "_archive")

# The user explicitly chose Sonnet for this incremental-translation workload
# (good enough for glossary-constrained technical translation, far cheaper than
# Opus). Bare model string — do NOT append a date suffix.
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 8192
BATCH_SIZE = 40          # entries per API call; translation strings are short

SYSTEM = """\
You are a professional technical translator localizing the documentation of
TensorMesh -- a PyTorch-based finite element method (FEM) library -- from English
into Simplified Chinese (zh_CN).

You receive a JSON array of entries, each {"i": <int>, "en": "<English>"}.
Translate each "en" into natural, professional Simplified Chinese as a native FEM
researcher would write it, and return them through the submit_translations tool
as {"i": <same int>, "zh": "<translation>"}. Return exactly one zh per input i.

Rules:
1. Preserve reStructuredText / Sphinx inline markup verbatim, with the exact
   backtick / colon / asterisk counts:
     ``code``, :class:`Mesh`, :func:`assemble`, :doc:`installation`,
     *italic*, **bold**, :math:`...` (keep all LaTeX inside untouched),
     URLs, and `<links>`.
2. Do NOT translate: class / function / module names (Mesh, ElementAssembler,
   tensormesh.mesh, SparseMatrix, Condenser, ...), Python / NumPy / SciPy API
   names, code snippets, file paths, and CLI commands.
3. If an entry is purely a code identifier or a single API name, return it
   IDENTICAL (no translation).
4. Keep these in English: PyTorch, NumPy, SciPy, gmsh, meshio, vmap, autograd,
   nn.Module, Tensor -- and all class / function names.
5. Translate prose naturally; never word-by-word.
6. Use this glossary consistently (English -> 中文):
   finite element method -> 有限元方法; mesh -> 网格; element -> 单元 (NEVER 元素);
   node -> 节点; cell -> 单元/网格单元; facet/face -> 面; edge -> 边;
   boundary -> 边界; boundary condition -> 边界条件;
   Dirichlet/Neumann/Robin BC -> 狄利克雷/诺伊曼/罗宾边界条件;
   degree of freedom -> 自由度; basis function -> 基函数; shape function -> 形函数;
   weak form -> 弱形式; variational form -> 变分形式;
   assembly -> 装配; assembler -> 装配器;
   stiffness matrix -> 刚度矩阵; mass matrix -> 质量矩阵; load vector -> 载荷向量;
   source term -> 源项; residual -> 残差; Jacobian -> 雅可比矩阵;
   gradient/divergence/curl -> 梯度/散度/旋度; strain/stress -> 应变/应力;
   displacement -> 位移; pressure -> 压力;
   sparse/dense matrix -> 稀疏矩阵/稠密矩阵; solver -> 求解器; backend -> 后端;
   condensation -> 静态凝聚; condenser -> 凝聚器;
   quadrature -> 求积 (quadrature point -> 求积点/高斯点);
   reference element -> 参考单元; transformation -> 变换/映射;
   triangle/quadrilateral/tetrahedron/hexahedron/prism/pyramid ->
     三角形/四边形/四面体/六面体/三棱柱/四棱锥;
   line element -> 线单元; time integration -> 时间积分; time step -> 时间步;
   explicit/implicit -> 显式/隐式; ODE/PDE -> 常微分方程/偏微分方程;
   Runge-Kutta -> 龙格-库塔; Newton-Raphson -> 牛顿-拉夫森;
   Poisson/heat/wave equation -> 泊松方程/热方程/波动方程;
   linear elasticity -> 线弹性; hyperelasticity -> 超弹性; plasticity -> 塑性;
   Neo-Hookean -> 新胡克; J2 plasticity -> J2 塑性; contact -> 接触;
   topology optimization -> 拓扑优化; inverse problem -> 反问题;
   differentiable -> 可微; automatic differentiation -> 自动微分;
   forward/backward pass -> 前向/反向传播; GPU acceleration -> GPU 加速.

Do NOT insert CJK spacing escapes (the `\\ ` backslash-space between markup and
Chinese characters) -- a separate post-processing step handles that.
"""

TOOL = {
    "name": "submit_translations",
    "description": "Return the Simplified-Chinese translation for each given entry.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "i": {"type": "integer"},
                        "zh": {"type": "string"},
                    },
                    "required": ["i", "zh"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["translations"],
        "additionalProperties": False,
    },
}

# ---------------------------------------------------------------------------
# CJK + inline-markup spacing pass (copied verbatim from TRANSLATION_GUIDE.md)
# ---------------------------------------------------------------------------

_TOKEN = re.compile(
    r':[\w+-]+:`[^`]*`'        # :role:`...` and :math:`...`
    r'|``[^`]+``'              # ``literal``
    r'|`[^`]+`__?|`[^`]+`'     # `phrase`(opt _/__)
    r'|\*\*[^*]+\*\*'          # **strong**
    r'|\*[^*\s][^*]*\*'        # *emphasis*
)


def _wide(c: str) -> bool:
    """True for a CJK ideograph or CJK / full-width punctuation."""
    return bool(c) and ord(c) >= 0x2E80


def cjk_fix(text: str) -> str:
    """Insert RST escaped-spaces (``\\ ``) where markup touches a CJK character."""
    def repl(m: "re.Match[str]") -> str:
        s, full, i, j = m.group(0), m.string, m.start(), m.end()
        pre = "\\ " if i > 0 and _wide(full[i - 1]) else ""
        suf = "\\ " if j < len(full) and _wide(full[j]) else ""
        return pre + s + suf
    return _TOKEN.sub(repl, text)


# ---------------------------------------------------------------------------
# Catalog discovery + gap selection
# ---------------------------------------------------------------------------

def is_prose(path: Path) -> bool:
    parts = path.relative_to(LOCALE_DIR).parts
    return not any(p in SKIP_PARTS for p in parts)


def needs_translation(entry: polib.POEntry) -> bool:
    return bool(
        entry.msgid
        and not entry.obsolete
        and not entry.msgid_plural          # plurals are rare in Sphinx prose; skip
        and (not entry.msgstr.strip() or "fuzzy" in entry.flags)
    )


def discover(files: list[str] | None) -> list[Path]:
    if files:
        paths = [LOCALE_DIR / f for f in files]
    else:
        paths = [Path(p) for p in glob.glob(str(LOCALE_DIR / "**" / "*.po"), recursive=True)]
    return sorted(p for p in paths if p.is_file() and is_prose(p))


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate_batch(client, batch: list[polib.POEntry]) -> dict[int, str]:
    """Translate one batch; return {index_in_batch: chinese}."""
    payload = json.dumps(
        [{"i": idx, "en": e.msgid} for idx, e in enumerate(batch)],
        ensure_ascii=False,
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "disabled"},          # translation needs no reasoning
        output_config={"effort": "low"},        # Sonnet 4.6 defaults to high; rein it in
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        tools=[TOOL],
        tool_choice={"type": "tool", "name": "submit_translations"},
        messages=[{"role": "user", "content": "Translate these entries:\n\n" + payload}],
    )
    out: dict[int, str] = {}
    for block in resp.content:
        if block.type == "tool_use" and block.name == "submit_translations":
            for t in block.input.get("translations", []):
                out[int(t["i"])] = t["zh"]
    return out


def process_file(client, path: Path, do_cjk: bool) -> int:
    po = polib.pofile(str(path))
    todo = [e for e in po if needs_translation(e)]
    if not todo:
        return 0

    filled = 0
    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start:start + BATCH_SIZE]
        translations = translate_batch(client, batch)
        for idx, entry in enumerate(batch):
            zh = translations.get(idx)
            if not zh:
                print(f"    ! no translation for: {entry.msgid[:60]!r}", file=sys.stderr)
                continue
            entry.msgstr = cjk_fix(zh) if do_cjk else zh
            if "fuzzy" in entry.flags:
                entry.flags.remove("fuzzy")
            filled += 1

    if filled:
        po.save(str(path))
    return filled


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="*", help="catalogs relative to the locale dir (default: all prose)")
    ap.add_argument("--dry-run", action="store_true", help="report the gaps; make no API calls")
    ap.add_argument("--no-cjk-fix", action="store_true", help="skip the CJK escaped-space pass")
    args = ap.parse_args()

    if not LOCALE_DIR.is_dir():
        print(f"error: locale dir not found: {LOCALE_DIR}", file=sys.stderr)
        return 1

    catalogs = discover(args.files)
    if not catalogs:
        print("No prose catalogs found.", file=sys.stderr)
        return 1

    # Tally the gaps first so --dry-run needs neither anthropic nor a key.
    gaps = {p: sum(needs_translation(e) for e in polib.pofile(str(p))) for p in catalogs}
    total = sum(gaps.values())
    pending = {p: n for p, n in gaps.items() if n}

    print(f"Prose catalogs: {len(catalogs)}   entries to translate: {total}")
    for p, n in sorted(pending.items()):
        print(f"  {n:4d}  {p.relative_to(LOCALE_DIR)}")

    if args.dry_run:
        print("\n(dry run — nothing translated)")
        return 0
    if total == 0:
        print("\nNothing to translate. ✔")
        return 0
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nerror: set ANTHROPIC_API_KEY before translating.", file=sys.stderr)
        return 1

    import anthropic  # lazy: keeps --dry-run dependency-free

    client = anthropic.Anthropic()
    filled = 0
    for path in sorted(pending):
        print(f"\n-> {path.relative_to(LOCALE_DIR)}")
        filled += process_file(client, path, do_cjk=not args.no_cjk_fix)

    print(f"\nTranslated {filled}/{total} entries across {len(pending)} catalogs. ✔")
    print("Next: `make zh` and verify the ZH build adds zero warnings over `make html`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
