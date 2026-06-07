# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile gettext intl-update intl-update-prose intl-translate zh html-all

# Extract translatable strings to _build/gettext/*.pot
gettext:
	@$(SPHINXBUILD) -b gettext "$(SOURCEDIR)" "$(BUILDDIR)/gettext" $(SPHINXOPTS) $(O)

# Refresh source/locale/<lang>/LC_MESSAGES/*.po from the latest .pot files
intl-update: gettext
	@sphinx-intl update -p $(BUILDDIR)/gettext -l zh_CN

# Same as intl-update, but skips api/ (autodoc docstrings).
# Use this when you only want to refresh translations for prose pages
# (getting_started/, user_guide/, example_gallery/, performance/,
# community/, index.rst); the Chinese site will then fall back to English
# for the API reference. See docs/TRANSLATION_GUIDE.md.
intl-update-prose: gettext
	@rm -rf "$(BUILDDIR)/gettext/api"
	@sphinx-intl update -p $(BUILDDIR)/gettext -l zh_CN

# Auto-fill empty / fuzzy zh translations via Claude (needs ANTHROPIC_API_KEY).
# Refreshes the prose catalogs, translates the gaps, then runs the CJK-spacing
# pass. Deps: pip install -r requirements-translate.txt. See
# docs/TRANSLATION_GUIDE.md and docs/scripts/translate_po.py.
intl-translate: intl-update-prose
	@python scripts/translate_po.py

# Build the Chinese site into _build/html/zh/
zh:
	@$(SPHINXBUILD) -b html -D language=zh_CN "$(SOURCEDIR)" "$(BUILDDIR)/html/zh" $(SPHINXOPTS) $(O)

# Build both languages: English at _build/html/, Chinese at _build/html/zh/
html-all: html zh

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
