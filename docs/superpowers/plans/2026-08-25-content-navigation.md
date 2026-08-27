# Content Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compact floating contents menu that navigates meaningful blocks in the active page section while preserving the existing left step navigation.

**Architecture:** Add one reusable contents-menu shell to `ui/index.html`. `ui/app.js` discovers headings inside the active `.tab-pane`, assigns stable targets to their panels, renders grouped links, and tracks the visible panel with `IntersectionObserver`. `ui/styles.css` handles the fixed desktop popover, keyboard focus, collapsed groups, and mobile overlay.

**Tech Stack:** Existing semantic HTML, vanilla JavaScript, CSS, `IntersectionObserver`, `scrollIntoView`, no new dependencies.

---

## File map

- Modify: `ui/index.html` — contents-menu shell and navigation metadata.
- Modify: `ui/app.js` — discovery, rendering, scrolling, active-state tracking, and menu events.
- Modify: `ui/styles.css` — fixed popover, states, focus, and responsive layout.
- Verify: `node --check ui/app.js`, `git diff --check`, manual browser acceptance.

## Task 1: Mark navigation targets and add menu shell

**Files:** `ui/index.html`

- [ ] **Step 1: Preserve existing worktree changes.** Run `git diff -- ui/index.html ui/app.js ui/styles.css > .navigation-baseline.patch`; keep this local file untracked.

- [ ] **Step 2: Add trigger beside existing header actions.**

```html
<button id="contents-toggle" class="icon-btn contents-toggle" type="button" aria-expanded="false" aria-controls="contents-menu" title="Содержание">
  <span aria-hidden="true">☷</span>
  <span class="sr-only">Содержание</span>
</button>
```

- [ ] **Step 3: Add popover after `.workspace` opens.**

```html
<aside id="contents-menu" class="contents-menu" aria-label="Содержание страницы" hidden>
  <div class="contents-menu-head">
    <strong>Содержание</strong>
    <button id="contents-close" class="icon-btn" type="button" title="Закрыть содержание" aria-label="Закрыть содержание">×</button>
  </div>
  <div id="contents-list" class="contents-list"></div>
</aside>
```

- [ ] **Step 4: Add `data-content-group` to meaningful panels.** Use values `overview`, `datasets`, `studio`, `experiments`, `serving`, `resources`; use existing panel IDs where available and add stable `data-content-id` only when one panel contains several independent blocks. Do not change existing IDs or event hooks.

- [ ] **Step 5: Run `node --check ui/app.js`; expect exit code `0`.**

## Task 2: Implement menu discovery and navigation

**Files:** `ui/app.js`

- [ ] **Step 1: Add menu state and DOM references.** Extend `state` with `contentsCollapsed: new Set()`. Cache `contentsToggle`, `contentsMenu`, `contentsClose`, and `contentsList` beside existing navigation references.

- [ ] **Step 2: Add four small functions: `getContentItems`, `renderContentsMenu`, `setContentsOpen`, `observeContentPanels`.** `getContentItems()` reads only active `.tab-pane`, selects direct `.panel` elements, uses the first `h2/h3/h4` as label, ignores panels without headings, groups by `data-content-group`, and preserves DOM order.

`renderContentsMenu()` renders accessible `button type="button"` links with `data-content-target`; group buttons toggle `.is-collapsed` and `aria-expanded`. Item buttons call `scrollIntoView({ behavior: document.body.dataset.motion === 'off' ? 'auto' : 'smooth', block: 'start' })` and close the menu on narrow screens.

`setContentsOpen(open)` synchronizes `hidden`, `aria-expanded`, and a workspace class. Escape, close button, and outside clicks close it.

`observeContentPanels()` disconnects the previous observer, observes current targets with `rootMargin: '-72px 0px -55% 0px'`, and toggles `.is-current`. If `IntersectionObserver` is unavailable, menu navigation still works without active tracking.

- [ ] **Step 3: Call `renderContentsMenu()` and `observeContentPanels()` after active `.tab-pane` changes in `goToStep()`.** Preserve existing layout checks.

- [ ] **Step 4: Call `setupContentsNavigation()` once from `init()`.** Use event delegation on `contentsList` so re-rendering adds no duplicate listeners.

- [ ] **Step 5: Run `node --check ui/app.js`; expect exit code `0`.**

## Task 3: Style responsive menu and verify behavior

**Files:** `ui/styles.css`

- [ ] **Step 1: Add desktop styles.** Style `.contents-toggle`, `.contents-menu`, `.contents-menu-head`, `.contents-list`, `.contents-group`, `.contents-group-toggle`, and `.contents-item`. Use `position: fixed`, placement below sticky header, right alignment, high z-index, `max-height: calc(100vh - 72px)`, and internal scrolling.

- [ ] **Step 2: Add state/accessibility styles.** Style `[hidden]`, collapsed groups, `.is-current`, hover, `:focus-visible`, and dark theme with existing colors/radii. Add `.sr-only` only if not already present.

- [ ] **Step 3: Add responsive styles.** At `max-width: 900px`, make the menu a right overlay below the header with safe width. At `max-width: 600px`, make content buttons full width with 44px minimum touch targets and preserve no horizontal overflow.

- [ ] **Step 4: Manual acceptance.** Verify opening/closing on all six steps; correct scroll targets; active item while scrolling; group collapse; Escape/outside click; keyboard focus and Enter/Space; mobile overlay and close-after-navigation; unchanged left navigation, theme, forms, and dashboards.

- [ ] **Step 5: Review diff.** Run `git diff --check` and `git diff -- ui/index.html ui/app.js ui/styles.css`; expect no whitespace errors and only contents-menu markup, logic, and CSS changes.
