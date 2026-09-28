GitHub Issues
=============

The issue tracker is for **confirmed bugs** and **concrete feature
requests** — things a maintainer (or you) could in principle act on by
writing code. Open-ended questions, design discussion, and "how do I
do X" belong on :doc:`github_discussions` or :doc:`discord` instead;
the new-issue chooser will offer those routes alongside the templates.


Open an issue
-------------

.. raw:: html

   <p style="margin: 1.2em 0;">
     <a href="https://github.com/camlab-ethz/TensorMesh/issues/new/choose"
        style="background-color:#cf222e; border-color:#cf222e;
               color:white; padding:8px 16px; border-radius:6px;
               text-decoration:none; font-weight:600;">
       File a new issue &nbsp;&rarr;
     </a>
     &nbsp;&nbsp;
     <img alt="open issues"
          src="https://img.shields.io/github/issues/camlab-ethz/TensorMesh?style=for-the-badge&color=cf222e"
          style="vertical-align:middle;">
   </p>

Clicking the link above takes you to the *new-issue chooser* — a menu
with two templates and three "you probably want to go elsewhere"
shortcuts:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Choice
     - When to pick it
   * - 🐛 **Bug Report**
     - Something is broken or wrong. You can reproduce it.
   * - ✨ **Feature Request**
     - You want a specific, well-scoped addition or change.
   * - 💬 Usage question
     - Redirects to **Discussions Q&A** — not an issue.
   * - 💡 Idea or proposal
     - Redirects to **Discussions Ideas & RFCs** — not an issue.
   * - 🗨️ Real-time chat
     - Redirects to **Discord**.

Blank issues are disabled — every issue starts from a template, which
keeps triage tractable.


What each template captures
---------------------------

Both templates are GitHub *issue forms*: structured fields with
validation, rendered as a form rather than free-form Markdown. Required
fields are clearly marked.

**🐛 Bug Report** — required fields are *Description*, *Minimal
reproducing example*, *Expected behavior*, *Environment*, plus two
pre-flight checkboxes. The full traceback is optional (some bugs
produce a wrong result rather than an exception). The environment
field tells you exactly what to paste:

.. code-block:: bash

   python -c "import torch, torch_sla, tensormesh, platform; \
     print('torch:      ', torch.__version__); \
     print('torch_sla:  ', torch_sla.__version__); \
     print('tensormesh: ', tensormesh.__version__); \
     print('python:     ', platform.python_version()); \
     print('os:         ', platform.platform()); \
     print('cuda:       ', torch.version.cuda if torch.cuda.is_available() else 'n/a')"

If you can't fill out the *Minimal reproducing example*, that usually
means the problem isn't reduced enough yet — try to shrink it before
filing. A 200-line script and a five-line script take roughly the
same amount of human attention to read, and the five-line script is
ten times more likely to get a fast answer.

**✨ Feature Request** — only *Use case* is strictly required. Lead
with the **motivation** ("I'm trying to solve X") rather than a
proposed API ("please add ``foo``"). A clear use case lets the
maintainers and the community shape the API together; a pre-specified
API often anchors the discussion in the wrong place.


What happens after you file
---------------------------

The basic shape:

1. **Auto-label** — every new issue lands with the ``triage`` label
   plus its category (``bug`` or ``enhancement``), set by the
   template.
2. **Triage** — a maintainer reads it, possibly asks for more info,
   removes ``triage`` once the report is actionable, and adds topic
   labels (e.g. ``meshing``, ``solver``, ``cuda``) so similar issues
   cluster together.
3. **Disposition** — one of:

   * Confirmed and queued for work (someone will pick it up; you are
     welcome to too — see :doc:`contributing`).
   * Asked for clarification (please respond; stale issues with no
     follow-up may be closed).
   * Closed as a duplicate, invalid, or "won't fix" with an
     explanation.

This is a research-affiliated project with a small maintainer team —
response times are best-effort, not contractual. If your issue is
urgent for a paper deadline, mention it in the description; if it has
been silent for more than two weeks, a polite ping is welcome.


Before you file: search first
-----------------------------

A 30-second search avoids most duplicates and often turns up an
existing workaround. Useful entry points:

* Open + closed issues, scoped to this repo:
  ``https://github.com/camlab-ethz/TensorMesh/issues?q=<terms>``
  (drop the default ``is:open`` to include closed ones).
* Discussions — usage questions and design conversations live there,
  not here.
* Google with a ``site:`` filter:
  ``site:github.com/camlab-ethz/TensorMesh <your terms>``.

If you find a related issue but yours is genuinely different, link
to it from your new issue ("related to #NN, but X is different") —
that helps triage decide whether to merge them.


From Discussion to Issue
------------------------

Issues and Discussions cross over in two natural directions:

* **Q&A → Bug.** A "how do I…" question on Discussions Q&A turns
  out to be a real bug. Open an issue, reference the discussion
  thread for context (``Discussion: #DD``), and link the issue back
  from the discussion. Don't re-paste the whole transcript.
* **Idea & RFC → Feature.** After a design discussion converges,
  file an issue (or a PR) with the agreed design in the description,
  and link the discussion thread. The issue is then the *commitment*;
  the discussion is the *deliberation*.


Pull requests
-------------

This page is about *filing* issues. If you want to *fix* one, the PR
workflow — dev environment, branch naming, tests, doc updates — lives
in :doc:`contributing`. Linking the PR to its issue
(``Fixes #NN`` in the description) auto-closes the issue on merge,
which the triage flow above relies on.
