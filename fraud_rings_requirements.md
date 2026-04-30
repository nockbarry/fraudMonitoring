# Fraud Rings Hub — Requirements

A complete, implementation-ready specification for the standalone Fraud Rings
dashboard at `/fraud_rings_hub.html`. The document is organized so each section
of the report can be implemented independently from the rest given a shared
data model and chrome layer.

---

## 0. Global / framework

- **G1.** Single standalone HTML file. No build step. Plotly via CDN
  (`plotly-2.35.0.min.js`).
- **G2.** Self-contained synthetic data; deterministic across reloads via a
  seeded PRNG (`mulberry32` + Box-Muller `gauss()`).
- **G3.** Hash-based router: `#/<section>` for direct routes, `#/focus?r=R042`
  for the parameterized ring-focus route. Browser back/forward must work via
  `hashchange`.
- **G4.** All charts use a shared `baseLayout()` with Inter font (11px), white
  paper / plot background, and a horizontal legend below plot at y=−0.18 by
  default.
- **G5.** Common color tokens for every archetype, entity kind, and status:

  | Archetype       | Color     | Entity / status | Color     |
  | --------------- | --------- | --------------- | --------- |
  | Synthetic ID    | `#a855f7` | Account         | `#6366f1` |
  | Mule            | `#0ea5e9` | Card            | `#f59e0b` |
  | Card testing    | `#f59e0b` | Device          | `#10b981` |
  | Bust-out        | `#ef4444` | IP              | `#ec4899` |
  | ATO             | `#14b8a6` | Merchant        | `#94a3b8` |
  | Triangulation   | `#ec4899` | Sent            | `#10b981` |
  | Refund fraud    | `#8b5cf6` | Error           | `#94a3b8` |
  | First-party     | `#f97316` | Denied          | `#ef4444` |
  | Friendly fraud  | `#06b6d4` | active          | `#ef4444` |
  | Promo abuse     | `#84cc16` | monitoring      | `#f59e0b` |
  |                 |           | investigating   | `#6366f1` |
  |                 |           | blocked         | `#10b981` |

- **G6.** Helper functions required: `el(tag, cls, html)`, `pageHeader(title, desc)`,
  `chartCard(title, sub, height)`, `baseLayout(extra)`, `fmt$()`, `fmtN()`,
  `fmtP()`, `clip(x, lo, hi)`, `archPill(arch)`, `statusPill(status)`,
  `makeCallout(title, body, kind)`.
- **G7.** Pill kinds: `pill-good`, `pill-warn`, `pill-bad`, `pill-accent`, plus
  one per archetype (`pill-syn-id`, `pill-mule`, …, `pill-promo-abuse`) and one
  per ring status (`pill-status-active`, …, `pill-status-blocked`).
- **G8.** Callout kinds: `info` (indigo), `good` (green), `warn` (amber),
  `bad` (red).

---

## 1. Layout / chrome

- **L1.** Two-pane app: 280px dark sidebar (`#0f172a`), light main content area.
- **L2.** Sidebar contains: hub title with accent square, sub-label, four
  cross-hub links (Model Performance, Feature Analysis, Stability & Calibration —
  the rings hub is the only one without a self-link), nav tree, footer.
- **L3.** Nav tree of `direct` items only (no expandable sections). Each nav
  entry can carry an optional badge with a `kind` (accent / warn / bad / good).
- **L4.** Topbar (56px) with breadcrumbs left, comma-separated topstats right
  (rings count, active count, total $ exposure, capture %).
- **L5.** Page header pattern: 20px bold title + 13px description paragraph
  (max 880px wide, line-height 1.55, supports `<code>` and inline pills).
- **L6.** KPI grid: 4 columns at desktop, 2 at <1100px, optional left border
  accent (`accent` / `good` / `warn` / `bad`) plus optional inline border-left
  override for archetype tiles.
- **L7.** Chart card pattern: bordered white card with head (title + optional
  subtitle) and body div. Default height 360px; configurable.
- **L8.** Tables (`.rt-tbl`): uppercase 11px headers, cursor-pointer rows,
  hover-highlight `#f8fafc`, ring-id cells in monospace accent color, `.hl`
  class for emphasized rows.
- **L9.** Ring focus hero: gradient indigo background, monospace ring-id title,
  metadata pills, prev/next nav buttons.

---

## 2. Synthetic data model

### 2.1 Time grid

- **D1.** 70 days starting `START = 2026-02-19`. `dates` is an array of ISO
  date strings.

### 2.2 Archetypes

- **D2.** 10 archetypes in fixed `ARCH_ORDER` order:
  `[syn_id, mule, card_test, bust_out, ato, triangulation, refund, first_party, friendly, promo_abuse]`.
- **D3.** Each archetype has a template in `ARCH_TPL` with the following
  fields:
  - `count` — number of rings of this archetype to generate
  - `n_acct`, `n_card`, `n_dev`, `n_ip` — `[mean, std]` for entity counts
  - `$exp` — `[mean, std]` for total dollar exposure
  - `shareDev`, `shareIp` — fraction of devices/IPs that get reused inside the ring
  - `baseline_auc` — production-model AUC center for this archetype
  - `tx_per_acct` — `[mean, std]` for transactions per account over ring lifetime
- **D4.** Total ring count must be ≥240 with the canonical config.
  Reference distribution:

  | Archetype       | count | baseline_auc | $exp mean |
  | --------------- | ----- | ------------ | --------- |
  | Synthetic ID    | 35    | 0.72         | $85K      |
  | Mule            | 28    | 0.78         | $220K     |
  | Card testing    | 26    | 0.93         | $12K      |
  | Bust-out        | 18    | 0.65         | $340K     |
  | ATO             | 28    | 0.81         | $180K     |
  | Triangulation   | 22    | 0.74         | $195K     |
  | Refund fraud    | 20    | 0.58         | $62K      |
  | First-party     | 16    | 0.55         | $105K     |
  | Friendly fraud  | 20    | 0.62         | $42K      |
  | Promo abuse     | 30    | 0.71         | $28K      |

### 2.3 Per-archetype status fingerprint

- **D5.** `ARCH_STATUS_MIX[arch] = { sent, error, denied }` summing to 1.
  Reference values:

  | Archetype       | sent | error | denied |
  | --------------- | ---- | ----- | ------ |
  | Synthetic ID    | 0.74 | 0.06  | 0.20   |
  | Mule            | 0.85 | 0.04  | 0.11   |
  | Card testing    | 0.16 | 0.55  | 0.29   |
  | Bust-out        | 0.93 | 0.02  | 0.05   |
  | ATO             | 0.62 | 0.10  | 0.28   |
  | Triangulation   | 0.81 | 0.05  | 0.14   |
  | Refund fraud    | 0.91 | 0.03  | 0.06   |
  | First-party     | 0.95 | 0.02  | 0.03   |
  | Friendly fraud  | 0.94 | 0.02  | 0.04   |
  | Promo abuse     | 0.71 | 0.07  | 0.22   |

### 2.4 Per-archetype feature signature

- **D6.** `ARCH_FEATURE_PROFILE[arch] = { top: [feature_name…], weight: 0–1 }`.
  `top` is an ordered list of 5–6 features that should light up most for this
  archetype; `weight` controls overall signal strength (= predictability of
  the archetype by these features alone). Card-testing has the highest weight
  (0.88), first-party the lowest (0.50).
- **D7.** `FEATURE_POOL` is a fixed list of 25 fraud-style feature names
  (velocity, declines, 3DS, BIN risk, merchant risk, device, IP, behavioral,
  embedding, account-age, and interaction features) that the per-ring feature
  signal vectors are sampled from.

### 2.5 Per-ring fields

- **D8.** Each generated ring has:
  - `id` — `R000` … `R242` zero-padded
  - `arch`, `status`, `analyst`, `detection` (one of 8 detection methods)
  - Member counts: `n_acct`, `n_card`, `n_dev`, `n_ip`, `n_merch`
  - Member id arrays: `accts`, `cards`, `devs`, `ips`, `merchs` (entity ids
    with `A_`, `C_`, `D_`, `I_`, `M_` prefixes, zero-padded)
  - `$exp`, `$cap`, `captureRate`
  - `first_day`, `last_day`, `peak_day`, `lifetime` in days
  - `total_tx` — total transactions over lifetime
  - `daily_dollars[d]` — daily $ exposure (triangular ramp-to-peak-then-decay
    with 15% noise)
  - `daily_tx_count[d]` — daily transaction count (same shape, 20% noise)
  - `daily_alerts[d]` — daily alert count (derived)
  - `daily_status_counts[d] = { sent, error, denied }` — multinomial split of
    `daily_tx_count[d]` using `status_mix`
  - `total_sent`, `total_error`, `total_denied` — lifetime totals
  - `status_mix = { sent, error, denied }` — per-ring mix (archetype baseline
    + per-ring noise of σ≈0.05, renormalized to sum=1)
  - `model_auc` — `clip(baseline_auc + N(0, 0.06) − lifetime·0.0008, 0.50, 0.99)`
  - `model_recall_2pct` — derived: `clip((auc − 0.5)·1.5 + 0.05 + N(0, 0.04), 0, 0.95)`
  - `mean_score` — derived from `model_auc` with noise
  - `daily_avg_score[d]` — model score time series with per-ring drift
  - `daily_recall[d]` — recall time series with same drift, larger noise
  - `feature_signal` — array of `{ name, strength }` from archetype top-K with
    decay `weight·(1 − i·0.10) + N(0, 0.07)`
  - `shared_devs`, `shared_ips` — counts of cross-ring entity overlaps

### 2.6 Cross-ring overlap

- **D9.** ~10% of devices and ~8% of IPs get planted into a second random ring
  (post-generation). `dev2rings` and `ip2rings` maps track which rings each
  entity belongs to; `overlapEdges` is the set of `{ a, b, kind, count }` ring
  pairs that share at least one device or IP.

### 2.7 Statuses

- **D10.** Ring status values: `active` (32%), `monitoring` (28%),
  `investigating` (18%), `blocked` (22%) sampled at generation time.

### 2.8 Non-ring fraud

- **D11.** A separate `NON_RING` track represents uncoordinated single-actor
  fraud — opportunistic individual fraud transactions that aren't members of
  any ring. The dashboard surfaces this on portfolio-level plots (where the
  question is "what is happening across all fraud") and excludes it from
  ring-specific views (catalog, network graph, ring focus, overlap, queue).
- **D12.** `NON_RING` fields:
  - `label`: `'Non-ring fraud'`
  - `color`: `#64748b` (slate gray, distinct from any archetype color)
  - `status_mix`: `{ sent: 0.78, error: 0.08, denied: 0.14 }` — between the
    cleanest archetypes (bust-out / first-party) and the messy ones (card
    testing / ATO)
  - `model_auc`: `0.84` — model handles individual fraud reasonably well via
    velocity/risk features
  - `model_recall_2pct`: `0.42`, `mean_score`: `0.18`
  - `total_dollars`, `total_tx`, `total_sent`, `total_error`, `total_denied` —
    aggregate totals
  - `daily_dollars[d]`, `daily_tx_count[d]` — daily series with mid-week peak
    seasonality (`1 + 0.18 · sin((dow − 1.5) · π/3.5)`) and a slight upward
    drift (`1 + 0.0035·d`) so it isn't visually flat
  - `daily_status_counts[d] = { sent, error, denied }` — multinomial split of
    `daily_tx_count[d]` using the static `status_mix`
  - `daily_recall[d]`, `daily_avg_score[d]` — flat baselines with light noise

### 2.9 Aggregate KPIs

- **D13.** Computed once at generation:
  `N_RINGS`, `TOTAL_EXP` (rings only), `TOTAL_FRAUD_EXP` (rings + non-ring),
  `NON_RING_SHARE` (non-ring $ as fraction of total fraud $), `TOTAL_CAP`,
  `TOTAL_ACCTS`, `TOTAL_CARDS`, `TOTAL_DEVS` (unique), `TOTAL_IPS` (unique),
  `ACTIVE_COUNT`, `NEW_THIS_WEEK`, `SHARED_DEVS`, `SHARED_IPS`, `CAPTURE_RATE`.

---

## 3. Pages & routes

The hub has 12 routes. Routes are direct (one click) except `focus`, which is
parameterized by `?r=<ring_id>` and reached by clicking any ring-id row in any
table or any node in the macro graph.

### 3.1 `overview` (direct)

- **P1.** Page header explaining the hub and citing aggregate counts.
- **P2.** First KPI row (4 cards): active rings (bad), $ exposure (warn) with
  capture %, surfaced this week (accent), cross-ring overlap.
- **P3.** Second KPI row (4 cards): non-ring fraud $ (slate-gray border) with
  share of total fraud, non-ring transactions with model AUC and recall, total
  fraud $ (rings + non-ring), accounts in rings (with cards/devices/IPs in
  the subtitle line).
- **P4.** Side-by-side row: daily $ exposure stacked-area by archetype + daily
  transaction count stacked-area by archetype (same shape, very different per-
  archetype weighting — card-testing dominates the count chart, bust-out
  dominates the dollars chart). **Both charts include `Non-ring fraud` as the
  bottom-most layer of each stack** (slate-gray). **Both charts have their own
  legend** and the legends are **synced**: clicking a series in either chart's
  legend toggles the same series in the other chart, and double-click
  (Plotly's "isolate this series") mirrors across both charts as well. The
  sync is implemented via `plotly_legendclick` and `plotly_legenddoubleclick`
  handlers that match traces by `name`.
- **P5.** Daily transaction status across all rings — stacked area of sent /
  error / denied. **Counts include non-ring fraud transactions** so the chart
  represents the whole fraud-volume picture, not just rings. Subtitle calls
  out card-testing as the source of the gray layer.
- **P6.** Per-archetype model coverage: grouped bars of mean AUC per archetype
  and recall @ 2% on a secondary y-axis. Reference line at AUC=0.7.
- **P7.** Two-column row: status pie (rings by status, donut, hole=0.5) +
  archetype × $-exposure grouped bar.
- **P8.** Top 10 rings by exposure table — clickable rows drill into focus.

### 3.2 `catalog` (direct, badge: total ring count)

- **P9.** Page header.
- **P10.** Filter bar: archetype radio (`All` + 10 archetypes), status radio
  (`All` + 4 statuses), search box (matches ring id, analyst, detection method).
- **P11.** Table with sortable columns: Ring, Archetype, Status, # acct,
  # cards, # dev, # ip, $ exposure, Capture, First seen, Last seen, Detection,
  Analyst. Active sort column shows `▲`/`▼`.
- **P12.** Sort + filter persisted in `catalogState` for the lifetime of the
  page.
- **P13.** Row click → `focus?r=<id>`.
- **P14.** Display caps at 500 rows.

### 3.3 `graph` (direct, badge: `live`)

- **P15.** Page header.
- **P16.** Macro network graph: one node per ring, sized by exposure (capped
  diameter 40px), colored by archetype, positioned in a per-archetype
  concentric layout (radius grows by 5 per archetype, angular slot inside
  each archetype's circle proportional to ring index).
- **P17.** Edges: orange/pink lines connecting ring pairs that share at least
  one device or IP.
- **P18.** One trace per archetype so the legend filters by archetype.
- **P19.** Hover tooltip: ring id, archetype, status, # accounts, $ exposure.
- **P20.** Click on a node → `focus?r=<id>`.
- **P21.** Reading callout below explaining cross-cluster edges as the
  highest-priority signal.

### 3.4 `focus?r=<id>` (parameterized)

- **P22.** Hero card: monospace ring id, archetype + status + $ exposure +
  capture % + analyst + lifetime metadata, prev/next buttons disabled at the
  ends.
- **P23.** 5-tile entity grid: accounts / cards / devices / IPs / merchants
  with colored bullet dots matching entity color tokens.
- **P24.** Entity network graph: concentric layout — devices+IPs inside (r=0.5),
  accounts middle (r=1.0), cards outer (r=1.7), merchants outermost (r=2.3).
  Account↔card edges (one per account×card slot), account↔device, account↔ip,
  card↔merchant edges. Five separate node traces (one per entity kind) so the
  legend filters by kind. Hover shows `<kind> <id>`.
- **P25.** 4-KPI model-coverage row: model AUC (good/warn/bad coloring),
  recall @ 2%, mean model score, total transactions. Subtitle on the AUC tile
  cites the archetype baseline for comparison.
- **P26.** Side-by-side: daily $ exposure chart with alerts on secondary axis +
  daily transaction count chart. Both have a peak-day vertical reference line.
- **P27.** Model performance over ring lifetime: dual-axis chart with daily
  recall (left, %) and daily mean score (right, 0–1).
- **P28.** Daily transaction status — `<ring.id>`: stacked-area of sent / error
  / denied with a peak-day vertical reference line. Subtitle compares the
  ring's mix to the archetype baseline.
- **P29.** Top features lighting up on this ring: horizontal bar chart of the
  ring's `feature_signal` vector, colored by archetype.
- **P30.** Pattern callout: archetype-specific narrative about the operational
  fingerprint plus the detection method that surfaced the ring.

### 3.5 `archetype` (direct, badge: archetype count)

- **P31.** Page header.
- **P32.** KPI tiles per archetype (10 tiles): count of rings, $ exposure,
  account count, with a left-border in the archetype color.
- **P33.** Boxplot: cards-per-account by archetype, log-y. Card-testing pops
  off the top of the chart.
- **P34.** Side-by-side: accounts-per-device boxplot (log-y) + lifetime
  distribution boxplot.
- **P35.** Status mix by archetype — stacked bar (sent / error / denied).
- **P36.** Archetype reference table with columns: Archetype pill, Operational
  signature, Detection signal, Why it matters, Model AUC pill (good/warn/bad
  by 0.8 / 0.65 thresholds).

### 3.6 `coverage` (direct, badge: blind-spot count, kind: bad)

- **P37.** Page header.
- **P38.** 4 KPIs: well-covered (AUC ≥ 0.80, good), moderate (0.65–0.80, warn),
  blind spots (AUC < 0.65, bad) with $ exposure, portfolio mean AUC + recall.
- **P39.** Per-ring exposure × model AUC scatter, log-x. Marker size scales
  with `total_tx`. Two horizontal reference lines at 0.65 (bad) and 0.80
  (good). Click a node → `focus?r=<id>`. Hover shows ring id, $, AUC, tx count.
- **P40.** Side-by-side: per-archetype AUC distribution boxplot + daily
  portfolio recall @ 2% (volume-weighted, including non-ring fraud
  transactions in the weighting).
- **P41.** Top 25 blind-spots table (rings with AUC < 0.7, sorted by $ exposure):
  Ring, Archetype, $ exposure, total tx, AUC, recall %, mean score, status.
  Click to drill.

### 3.7 `signals` (direct)

- **P42.** Page header.
- **P43.** Feature × archetype heatmap (Purples colorscale, zmin=0, zmax=0.9).
  Rows = features sorted by total-signal descending. Cols = archetypes in
  `ARCH_ORDER`. Hover shows feature, archetype, signal strength.
- **P44.** Side-by-side: top-5 features per archetype (one trace per archetype,
  legend toggles between them, only first archetype visible by default,
  horizontal bottom legend) + feature breadth chart (number of archetypes
  where each feature has signal > 0.4, color-coded by breadth).
- **P45.** Closing callout linking back to the model-coverage page.

### 3.8 `status` (direct, badge: `live`)

- **P46.** Page header explaining sent/error/denied semantics with inline pills.
- **P47.** 4 KPIs: total sent (green border), error (gray border), denied (red
  border), **Non-ring share** (slate-gray border) showing what fraction of all
  fraud transactions are uncoordinated. Counts in all four KPIs include
  non-ring fraud.
- **P48.** Daily transaction status — portfolio total: stacked area of sent /
  error / denied across all rings **plus non-ring fraud**.
- **P49.** Daily status share — normalized: same data with `groupnorm: 'percent'`
  showing per-day share of each status.
- **P50.** Status mix by archetype: horizontal stacked bars, one row per
  archetype plus a final **`Non-ring fraud`** row, percent labels inside
  each segment.
- **P51.** Daily denied-rate trend by archetype: one line per archetype
  showing daily volume-weighted denied rate (`null` for days with no traffic),
  plus a **dashed slate-gray line** for non-ring fraud as a comparison
  baseline.
- **P52.** Side-by-side: top-20 leaky rings (high `sent% × $`) table + top-20
  noisy rings (high `(error+denied) × tx_total`) table. Both clickable to drill.
- **P53.** Closing callout on how Model coverage and Transaction status diverge:
  AUC says what's catchable, denied rate says what's actually caught.

### 3.9 `overlap` (direct, badge: shared-entities count, kind: warn)

- **P54.** Page header.
- **P55.** 4 KPIs: overlapping devices, overlapping IPs, ring pairs with
  overlap, cross-archetype pairs (most diagnostic).
- **P56.** Archetype × archetype overlap intensity heatmap (Reds colorscale).
- **P57.** Top 30 ring pairs by overlap count: Ring A, Archetype A, Ring B,
  Archetype B, Shared kind (device/IP), Count, Cross-archetype pill (`cross`
  in red, `same` in amber). Cross-archetype rows highlighted.

### 3.10 `growth` (direct)

- **P58.** Page header.
- **P59.** Active rings per day (line, fill-to-zero) with births (red bars)
  and deaths (green bars, plotted negative) on a secondary axis.
- **P60.** Cumulative $ exposure per archetype: one line per archetype plus
  a **dashed slate-gray line** for non-ring fraud cumulative exposure as a
  comparison baseline.
- **P61.** Ring lifetime distribution by current status (overlay histograms,
  bin size 5 days).

### 3.11 `detection` (direct)

- **P62.** Page header.
- **P63.** Detection method × archetype heatmap (Blues).
- **P64.** Side-by-side: rings per detection method (horizontal bar) + mean $
  exposure when detected per method (color-coded: red >$5K, amber >$2K,
  green ≤$2K — earlier detection is better).

### 3.12 `impact` (direct)

- **P65.** Page header.
- **P66.** Pareto chart: bars of per-ring $ exposure (sorted descending) +
  cumulative-share line on secondary axis (0–1, percent format).
- **P67.** Side-by-side: capture rate by status (bar with percent labels) +
  exposure vs captured by archetype (overlay bars in archetype color).
- **P68.** Reading-the-Pareto callout naming the top-N rings that drive 70%
  of cumulative exposure.

### 3.13 `queue` (direct, badge: investigating-count, kind: accent)

- **P69.** Page header.
- **P70.** Triage score per open ring (status `investigating` or `monitoring`)
  computed as `($exp/1000) × (1 + max(0, week-over-week velocity)) × (1 + 0.1·overlap_count)`.
- **P71.** 4 KPIs: open cases, open exposure, high-velocity (WoW lift > 30%),
  high-overlap (≥ 3 cross-ring ties).
- **P72.** Triage table sorted by score, top 25: Priority, Ring, Archetype,
  Status, Analyst, $ exposure, WoW growth (color-coded by sign and magnitude),
  Overlap ties, Lifetime. Click a row → focus.

---

## 4. Interactions

- **I1.** Click any nav item → route changes; existing Plotly elements purged
  before re-render.
- **I2.** Active nav item gets `.active` class with accent left border + tinted
  background; updated on every render.
- **I3.** Browser back/forward → `hashchange` re-resolves params and re-renders
  without full reload.
- **I4.** Cross-hub links in the sidebar are plain `<a href>` to the other
  three demo HTML files in the same directory.
- **I5.** Tables: any row whose `data-ring` attribute is set drills into the
  focus page. The catalog table additionally supports column-header sort
  toggling and live filter / search input.
- **I6.** Macro graph and coverage scatter both wire `plotly_click` to drill
  into focus.
- **I7.** Plotly charts are responsive (`responsive: true`, no modebar).
- **I8.** Focus page prev/next buttons disabled at index boundaries; otherwise
  navigate within `RINGS` order.
- **I9.** Hover tooltips on chart traces show formatted values; many use a
  `hovertemplate` with `customdata` to combine ring id + metrics.
- **I10.** Topbar stats reflect aggregate KPIs; rendered once at boot, not
  per-route.

---

## 5. Acceptance criteria

- **A1.** File loads in a modern browser with no JS console errors.
- **A2.** All 12 routes render distinct content with at least one chart or
  table each. Focus page renders for any valid `?r=<id>` and falls back to
  the first ring otherwise.
- **A3.** Determinism: two reloads produce visually identical charts.
- **A4.** Card-testing rings show >50% error rate in the per-archetype status
  mix bar (Status page) and dominate the gray layer of the portfolio status
  timeline. Bust-out and First-party rings show ≥90% sent in the same chart.
- **A5.** Per-archetype mean AUC matches the configured baselines within
  ±0.03; first-party (≈0.55) and refund (≈0.58) sit clearly below the 0.65
  blind-spot threshold; card-testing (≈0.93) sits clearly above 0.80.
- **A6.** The coverage scatter shows visible separation between archetypes:
  card-testing dots cluster top-left (low $, high AUC), bust-out dots cluster
  bottom-right (high $, mid AUC), first-party in the bottom-middle (low–mid $,
  low AUC).
- **A7.** Daily $ and daily transaction count charts have visibly different
  per-archetype proportions in the Overview row.
- **A8.** Cross-ring overlap heatmap has at least three off-diagonal cells
  with non-zero counts (cross-archetype reuse).
- **A9.** Focus page prev/next navigation traverses all 240+ rings without
  router errors.
- **A10.** Catalog filter+search reduces the table within 50ms; sort toggles
  re-rank without full re-render of the page.
- **A11.** No two charts overlap visually in the default 1280×900 viewport.
  Specifically: the Signals row's top-features and breadth charts each use a
  bottom horizontal legend (no right-anchored legend leaking into the next
  column).

---

## 6. Cross-hub integration

- **C1.** The fraud rings hub appears as a nav link in each of the other three
  demo hubs (Model Performance, Feature Analysis, Stability & Calibration) at
  the bottom of their cross-hub link list.
- **C2.** The landing page (`/index.html`) features the fraud rings hub as
  card 04 of 4, with a one-paragraph description summarizing its sections.
- **C3.** Cross-hub links are plain `<a href>` elements; no client-side state
  is preserved across hubs.
