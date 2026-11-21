Discord
=======

Real-time chat with the TensorMesh community — quick help, design
discussions, and showing off what you've built.

The Discord server is the right place when you want a fast, conversational
back-and-forth. For long-form Q&A that future users will find via search,
prefer :doc:`github_discussions`. For bug reports and feature requests,
file an :doc:`github_issues` entry instead.


Join the server
---------------

.. raw:: html

   <div style="display: flex; flex-wrap: wrap; gap: 24px; align-items: flex-start; margin: 1.5em 0;">
     <iframe src="https://discord.com/widget?id=1501598713389514853&theme=dark"
             width="350" height="500"
             allowtransparency="true" frameborder="0"
             sandbox="allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts">
     </iframe>
     <div style="flex: 1; min-width: 240px;">
       <p><strong>Invite link:</strong>
       <a href="https://discord.gg/EC9kbHSnrx">https://discord.gg/EC9kbHSnrx</a></p>
       <p>The widget on the left shows who is online right now and lets you
       join in one click. The invite link is permanent.</p>
     </div>
   </div>

The server's primary language is **English**.


Channel guide
-------------

The server has four categories. The list below describes what each
channel is *for* — most channels are quiet right now because the
community is just getting started.

**Information** — read-only announcements and ground rules.

* ``#welcome`` — start here.
* ``#announcements`` — releases, breaking changes, and event notices.
  Maintainer-posted; turn on notifications if you depend on the library.
* ``#rules`` — server etiquette (see :ref:`etiquette` below).

**Community** — open conversation.

* ``#general`` — anything TensorMesh-adjacent that doesn't fit elsewhere:
  FEM discussion, PyTorch tricks, ML-for-PDE papers, related libraries.
* ``#showcase`` — share what you built. Plots, animations, papers,
  blog posts, course material — all welcome.

**Help** — getting un-stuck.

* ``#help`` — usage questions. "How do I assemble X?", "Why doesn't my
  mesh load?", "What's the right API for Y?". See :ref:`asking-well`
  below for what to include.
* ``#troubleshooting`` — installation issues, environment problems,
  CUDA/torch version conflicts, and torch-sla solver-backend setup.

**Development** — for people writing patches.

* ``#dev`` — design discussion, refactors-in-flight, RFC-style threads
  before opening a PR. Maintainers and contributors hang out here.
* ``#staff-only`` — private channel for maintainer coordination
  (release planning, moderation). Listed here for transparency; you
  won't see it unless you have the maintainer role.


.. _asking-well:

Asking a good help question
---------------------------

A few lines of context turn an unanswerable question into one a
maintainer can answer in a minute:

* **What you ran** — a minimal snippet, ideally something a reader can
  paste into a Python shell. Triple-backtick formatting is supported.
* **What you saw** — the full traceback, not a paraphrase. Wrap long
  output in a code block.
* **What you expected** — the shape, value, or behavior you were after.
* **Versions** — output of
  ``python -c "import torch, torch_sla, tensormesh; print(torch.__version__, torch_sla.__version__, tensormesh.__version__)"``,
  plus your OS and (if relevant) CUDA version.

If your question is longer than a few paragraphs or you'd like the
answer to be searchable later, post it to
:doc:`github_discussions` and drop a link in ``#help``.


.. _etiquette:

Server etiquette
----------------

A short list of norms that keep the server pleasant:

* Be kind and patient. Many users are learning FEM, PyTorch, or both.
* Keep technical disagreements technical.
* No spam, no promotion of unrelated products, no harassment.
* English is the default in public channels; please translate if you
  paste a non-English error so others can help.

Maintainers may remove messages or members that violate these norms.
If you witness a problem, ping ``@moderator`` or DM a maintainer.


Where chat doesn't fit
----------------------

Discord is great for live conversation, but the medium loses things:
messages scroll away, threads aren't always searchable, and Google
doesn't index them. So:

* **Bug?** → :doc:`github_issues`. Even a one-line repro on Discord
  should end up as an issue.
* **Design discussion you want to last?** → :doc:`github_discussions`.
* **Release announcements** live on the GitHub release page, not in
  chat history — subscribe there if you want notifications you can't
  miss.

See you on the server.
