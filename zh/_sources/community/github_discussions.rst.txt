GitHub Discussions
==================

Long-form, searchable conversation that future users will find via
Google — questions with reusable answers, design proposals before a
PR, and write-ups of work built with TensorMesh.

Use Discussions when you want the conversation to stick around.
For a fast back-and-forth, hop into :doc:`discord` instead; for a
confirmed bug or a concrete feature request, open a
:doc:`github_issues`.


Go to Discussions
-----------------

.. raw:: html

   <p style="margin: 1.2em 0;">
     <a class="sd-sphinx-override sd-btn sd-text-wrap sd-btn-primary"
        href="https://github.com/camlab-ethz/TensorMesh/discussions"
        style="background-color:#2da44e; border-color:#2da44e;
               color:white; padding:8px 16px; border-radius:6px;
               text-decoration:none; font-weight:600;">
       Open Discussions on GitHub &nbsp;&rarr;
     </a>
     &nbsp;&nbsp;
     <img alt="discussions"
          src="https://img.shields.io/github/discussions/camlab-ethz/TensorMesh?style=for-the-badge&color=2da44e"
          style="vertical-align:middle;">
   </p>

You'll need a GitHub account to post or comment. Reading is open to
everyone.


Category guide
--------------

There are five categories, each tuned for a different kind of post.
Pick the one that matches what you're writing; if you're unsure, **General**
is always safe — a maintainer will move it if needed.

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Category
     - Format
     - What goes here
   * - **Announcements**
     - Announcement (read-only)
     - Release notes, API deprecations, scheduled breaking changes,
       community events. Maintainer-posted. **Subscribe to this
       category** if you depend on the library in production or in
       a paper.
   * - **Q&A**
     - Q&A (with *Mark as answer*)
     - Usage questions, install / environment problems, "how do I do
       X with TensorMesh". The OP — or a maintainer — marks one reply
       as the accepted answer, which makes the thread useful to the
       next person who hits the same problem.
   * - **Ideas & RFCs**
     - Open discussion
     - Proposals for new features, API design, breaking changes —
       *before* you write the PR. Floating an idea here first usually
       saves a round of "we'd actually want this shaped differently"
       review later. Link the eventual issue / PR in a follow-up
       comment.
   * - **Show & Tell**
     - Open discussion
     - Papers that used TensorMesh, simulation animations, blog posts,
       teaching materials, comparison plots against other FEM stacks.
       Self-promotion is welcome here — that's the whole point of
       the category.
   * - **General**
     - Open discussion
     - Anything FEM-, PyTorch-, or PDE-adjacent that doesn't fit
       above. Catch-all.


Writing a good Q&A post
-----------------------

The same checklist from the Discord help-channel guide applies, and
matters even more here because future readers will arrive at your
thread via search — they only have what you wrote. See
:ref:`asking-well` for the full list. The short version:

* **What you ran** — a minimal, copy-pasteable snippet.
* **What you saw** — the full traceback, in a code block.
* **What you expected** — shape, value, or behavior.
* **Versions** — output of
  ``python -c "import torch, torch_sla, tensormesh; print(torch.__version__, torch_sla.__version__, tensormesh.__version__)"``,
  plus OS and CUDA version if relevant.

A useful habit: when the issue is resolved, **edit the OP** to add a
one-line "Resolution:" summary at the top, so future readers don't have
to scroll through the back-and-forth. Then mark the answering reply.


Discord vs Discussions vs Issues
--------------------------------

The three channels solve different problems. The decision is usually
simple once you know what you're trying to do:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - You want to …
     - Use
   * - chat in real time, get unstuck in a few minutes
     - :doc:`discord` (``#help`` / ``#troubleshooting``)
   * - ask a question whose answer should be searchable later
     - **Discussions → Q&A**
   * - propose a new feature or API change for discussion
     - **Discussions → Ideas & RFCs**
   * - share a paper, demo, or write-up
     - **Discussions → Show & Tell**
   * - report a confirmed bug with a repro
     - :doc:`github_issues`
   * - request a specific, actionable feature
     - :doc:`github_issues` (after a quick Discussions sanity check
       on bigger features)
   * - announce / be notified of releases
     - **Discussions → Announcements** (watch this category)

When in doubt: post to Discussions, link from Discord. Chat that
sounded one-off often turns out to be useful to others later, and
moving a Discord exchange into a Discussion thread is a maintainer-
friendly thing to do.


Labels
------

Maintainers may tag Q&A discussions with labels like ``installation``,
``cuda``, ``meshing``, ``assembly``, ``solver``, ``autograd``, or
``docs`` to make search filtering possible. You don't need to label
your own post — pick the right category, and the labels follow.


Etiquette
---------

The same norms that apply on :ref:`Discord <etiquette>` apply here:
be kind, keep disagreements technical, no spam. A couple of
Discussions-specific courtesies on top:

* **Search before posting.** Many questions have already been asked.
  GitHub's in-page search across Discussions is decent; a quick
  ``site:github.com/camlab-ethz/TensorMesh/discussions <your terms>``
  on Google is even better.
* **One topic per thread.** If you find yourself describing two
  unrelated problems in the same post, split it. Threaded replies
  on Q&A only work well when the question is single-shot.
* **Don't open a duplicate.** If your problem matches an existing
  thread, comment there instead of opening a new one — even if the
  existing thread is closed. A maintainer will reopen it or redirect.
