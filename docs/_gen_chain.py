#!/usr/bin/env python3
"""Generate docs/depletion-chain.svg with clean orthogonal routing.

Layout choices:
- Single-bend (L-shape) where geometry allows; otherwise Z (2 bends).
- Parallel arrows in shared corridors are stacked with small y-offsets so they
  read as "slightly offset" bundles, with no overlap.
- Branching from one nuclide is routed symmetrically when geometry allows
  (Pm-147 -> Pm-148/Pm-148m, I-135 -> Xe-135m/Xe-135).
- Long alpha-decay corridors (Cm->Pu, Pu->U, Am->Np) live in dedicated bands so
  they never invade node text.
- Capture/decay tail- and head-points are offset (>=15 px apart) to avoid the
  arrow-in-arrow-out collision on shared edges (Pu-238 bottom, Np tops, ...).
- Ratio labels are rendered as a halo+text pair so they survive renderers that
  don't honour `paint-order:stroke` (e.g. cairosvg).
"""

from typing import List, Tuple

W, H = 1660, 1210

# ---------------------------------------------------------------------------
# Node positions
# ---------------------------------------------------------------------------
ACT_BOX_W, ACT_BOX_H = 88, 50
FP_BOX_W,  FP_BOX_H  = 110, 60
XE_BOX_W,  XE_BOX_H  = 86, 46

# Actinide column centers (mass 234 .. 245)
COLS = {234:110, 235:240, 236:370, 237:500, 238:630, 239:760,
        240:890, 241:1020, 242:1150, 243:1280, 244:1410, 245:1540}

# Row center y for each isotope row
ROW_Y = {
    "Cm":      150,
    "Am242m":  222,
    "Am":      270,
    "Am242":   318,
    "Pu":      390,
    "Np":      510,
    "U":       630,
}

actinides = {
    "U-234":   (COLS[234], ROW_Y["U"]),
    "U-235":   (COLS[235], ROW_Y["U"]),
    "U-236":   (COLS[236], ROW_Y["U"]),
    "U-238":   (COLS[238], ROW_Y["U"]),
    "Np-237":  (COLS[237], ROW_Y["Np"]),
    "Np-238":  (COLS[238], ROW_Y["Np"]),
    "Np-239":  (COLS[239], ROW_Y["Np"]),
    "Pu-238":  (COLS[238], ROW_Y["Pu"]),
    "Pu-239":  (COLS[239], ROW_Y["Pu"]),
    "Pu-240":  (COLS[240], ROW_Y["Pu"]),
    "Pu-241":  (COLS[241], ROW_Y["Pu"]),
    "Pu-242":  (COLS[242], ROW_Y["Pu"]),
    "Pu-243":  (COLS[243], ROW_Y["Pu"]),
    "Am-241":  (COLS[241], ROW_Y["Am"]),
    "Am-242m": (COLS[242], ROW_Y["Am242m"]),
    "Am-242":  (COLS[242], ROW_Y["Am242"]),
    "Am-243":  (COLS[243], ROW_Y["Am"]),
    "Am-244":  (COLS[244], ROW_Y["Am"]),
    "Cm-242":  (COLS[242], ROW_Y["Cm"]),
    "Cm-243":  (COLS[243], ROW_Y["Cm"]),
    "Cm-244":  (COLS[244], ROW_Y["Cm"]),
    "Cm-245":  (COLS[245], ROW_Y["Cm"]),
}

LAMBDA = {
    "U-234": "8.947e-14", "U-235": "3.121e-17", "U-236": "9.379e-16", "U-238": "4.916e-18",
    "Np-237": "1.024e-14", "Np-238": "3.790e-06", "Np-239": "3.405e-06",
    "Pu-238": "2.505e-10", "Pu-239": "9.110e-13", "Pu-240": "3.348e-12",
    "Pu-241": "1.537e-09", "Pu-242": "5.881e-14", "Pu-243": "3.885e-05",
    "Am-241": "5.077e-11", "Am-242m": "1.558e-10", "Am-242": "1.202e-05",
    "Am-243": "2.980e-12", "Am-244": "1.906e-05",
    "Cm-242": "4.924e-08", "Cm-243": "7.548e-10", "Cm-244": "1.213e-09", "Cm-245": "2.584e-12",
    "Nd-147": "7.307e-07", "Nd-148": "0", "Nd-149": "1.114e-04",
    "Pm-147": "8.373e-09", "Pm-148": "1.495e-06", "Pm-148m": "1.943e-07",
    "Pm-149": "3.627e-06",
    "Sm-147": "2.072e-19", "Sm-148": "3.138e-24", "Sm-149": "0",
    "I-135": "2.931e-05", "Xe-135m": "7.556e-04", "Xe-135": "2.107e-05",
}

fp_nodes = {
    "Nd-147":  (170, 765), "Nd-148":  (390, 765), "Nd-149":  (610, 765),
    "Pm-147":  (170, 910), "Pm-148":  (390, 845), "Pm-148m": (390, 975),
    "Pm-149":  (610, 910),
    # Sm row aligned to mass columns (Nd-147/Pm-147/Sm-147 share x=170,
    # Nd-148/Pm-148/Sm-148 share x=390, etc.).  Sm row y is pushed down to 1085
    # so the Pm-148m -> Sm-148 trunk has a 50 px visible run.
    "Sm-147":  (170, 1085), "Sm-148":  (390, 1085), "Sm-149":  (610, 1085),
}

ixe_nodes = {
    "I-135":   (1185, 815),
    "Xe-135m": (1460, 745),
    "Xe-135":  (1460, 895),
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def path(*pts) -> str:
    out = [f"M {pts[0][0]},{pts[0][1]}"]
    for p in pts[1:]:
        out.append(f"L {p[0]},{p[1]}")
    return " ".join(out)


def rect(cx, cy, w, h):
    return cx - w/2, cy - h/2, cx + w/2, cy + h/2


def act_sides(name):
    cx, cy = actinides[name]
    L, T, R, B = rect(cx, cy, ACT_BOX_W, ACT_BOX_H)
    return cx, cy, L, T, R, B


def fp_sides(name):
    cx, cy = fp_nodes[name]
    L, T, R, B = rect(cx, cy, FP_BOX_W, FP_BOX_H)
    return cx, cy, L, T, R, B


def xe_sides(name):
    cx, cy = ixe_nodes[name]
    L, T, R, B = rect(cx, cy, XE_BOX_W, XE_BOX_H)
    return cx, cy, L, T, R, B


edges: List[Tuple[str, str]] = []


def add(d: str, cls: str):
    if d:
        edges.append((d, cls))


# ============================================================================
# CAPTURE chain (black)
# ============================================================================
def cap_horiz(name_from: str, name_to: str, dy: float = 0.0):
    _, fy, _, _, fR, _ = act_sides(name_from)
    _, _, tL, _, _, _ = act_sides(name_to)
    add(path((fR, fy + dy), (tL, fy + dy)), "edge-cap")


cap_horiz("U-234", "U-235")
cap_horiz("U-235", "U-236")
cap_horiz("Np-238", "Np-239")
cap_horiz("Pu-238", "Pu-239")
cap_horiz("Pu-239", "Pu-240")
cap_horiz("Pu-240", "Pu-241")
cap_horiz("Pu-241", "Pu-242")
cap_horiz("Pu-242", "Pu-243")
cap_horiz("Am-243", "Am-244")
cap_horiz("Cm-242", "Cm-243")
cap_horiz("Cm-243", "Cm-244")
cap_horiz("Cm-244", "Cm-245")

# --- Synthesized captures (with implicit fast beta) -----------------------
# U-236 -> Np-237 (via U-237 fast beta).  Z up&right via column gap.
_, _, _, _, R236, _ = act_sides("U-236")
_, _, L237, _, _, _ = act_sides("Np-237")
add(path((R236, ROW_Y["U"]),
         (R236 + 21, ROW_Y["U"]),
         (R236 + 21, ROW_Y["Np"]),
         (L237, ROW_Y["Np"])), "edge-cap")

# U-238 -> Np-239 (via U-239 fast beta).  Up&right corridor at y=565.
_, _, _, T238, _, _ = act_sides("U-238")
_, _, _, _, _, B239 = act_sides("Np-239")
add(path((COLS[238], T238),
         (COLS[238], 565),
         (COLS[239], 565),
         (COLS[239], B239)), "edge-cap")

# Np-237 -> Pu-238 (via Np-238).  Enters Pu-238 from BOTTOM-CENTER.  Earlier
# we entered from LEFT, but that drew a horizontal segment between Pu-237 and
# Pu-238 that ran parallel to the Pu-238->Pu-239 capture and looked like a
# duplicate arrow.
_, _, _, T237, _, _ = act_sides("Np-237")
_, cy_pu238, _, _, _, B_pu238 = act_sides("Pu-238")
NP_PU_CORR = 476     # corridor in the Pu/Np gap, just below Np box tops
add(path((COLS[237] + 20, T237),
         (COLS[237] + 20, NP_PU_CORR),
         (COLS[238],      NP_PU_CORR),
         (COLS[238],      B_pu238)), "edge-cap")

# Np-239 -> Pu-240 (via Np-240).  Same: BOTTOM-CENTER entry to avoid the ghost
# duplicate alongside the Pu-239->Pu-240 horizontal.
_, _, _, T239, _, _ = act_sides("Np-239")
_, cy_pu240, _, _, _, B_pu240 = act_sides("Pu-240")
add(path((COLS[239] + 20, T239),
         (COLS[239] + 20, NP_PU_CORR),
         (COLS[240],      NP_PU_CORR),
         (COLS[240],      B_pu240)), "edge-cap")

# Am-241 -> Am-242 (n,gamma): out bot, drop to Am-242 row, then right.
_, _, _, _, _, B241_am = act_sides("Am-241")
_, _, L242_am, _, _, _ = act_sides("Am-242")
add(path((COLS[241], B241_am),
         (COLS[241], ROW_Y["Am242"]),
         (L242_am, ROW_Y["Am242"])), "edge-cap")

# Am-242 -> Am-243 (n,gamma): out top RIGHT-offset (clears Am-242m), into Am
# row at y BELOW center, so it doesn't collide with Am-243->Np-239 decay-out.
_, _, _, T242_am, _, _ = act_sides("Am-242")
_, _, L243, _, _, _ = act_sides("Am-243")
add(path((COLS[242] + 22, T242_am),
         (COLS[242] + 22, ROW_Y["Am"] + 12),
         (L243, ROW_Y["Am"] + 12)), "edge-cap")

# Am-242m -> Am-243 (n,gamma): out right, then down to Am row offset ABOVE
# center, paralleling the Am-242 cap on the other side of center.
_, _, _, _, R242m, _ = act_sides("Am-242m")
add(path((R242m, ROW_Y["Am242m"]),
         (R242m + 22, ROW_Y["Am242m"]),
         (R242m + 22, ROW_Y["Am"] - 12),
         (L243, ROW_Y["Am"] - 12)), "edge-cap")

# Am-243 -> Cm-244 (synthesized via fast Am-244 beta): up corridor at y=200.
_, _, _, T243, _, _ = act_sides("Am-243")
_, _, _, _, _, B244_cm = act_sides("Cm-244")
add(path((COLS[243], T243),
         (COLS[243], 200),
         (COLS[244], 200),
         (COLS[244], B244_cm)), "edge-cap")


# ============================================================================
# DECAYS inside same column (red, straight verticals with small offsets)
# ============================================================================
def vert_decay(name_from, name_to, dx: float = 0.0):
    fx, fy, _, fT, _, fB = act_sides(name_from)
    tx, ty, _, tT, _, tB = act_sides(name_to)
    if fy < ty:
        add(path((fx + dx, fB), (fx + dx, tT)), "edge-decay")
    else:
        add(path((fx + dx, fT), (fx + dx, tB)), "edge-decay")


# Np -> Pu (beta).  Source is below target. Offset RIGHT 20 so the arrow head
# at Pu BOT does not collide with Pu->U decay-out (offset LEFT 15).
vert_decay("Np-238", "Pu-238", dx=+20)
vert_decay("Np-239", "Pu-239", dx=+20)

# Pu -> Am (beta+/EC).  Offset LEFT 12 to avoid overlapping with Am->Pu cap-in.
vert_decay("Pu-241", "Am-241", dx=-14)
vert_decay("Pu-243", "Am-243", dx=-14)

# Am-242m -> Am-242 (IT, gamma).  Offset LEFT 14.
vert_decay("Am-242m", "Am-242", dx=-14)

# Am-242 -> Pu-242 (EC 82.7%): straight short vertical, offset LEFT 14.
vert_decay("Am-242", "Pu-242", dx=-14)

# Cm-243 -> Am-243 (small EC branch, drawn for chain completeness).
vert_decay("Cm-243", "Am-243", dx=+12)

# Am-244 -> Cm-244 (beta).  Offset LEFT 12.
vert_decay("Am-244", "Cm-244", dx=-12)


# Am-242 -> Cm-242 (beta- 17.3%): detour around Am-242m on its right.
_, _, _, _, R242_am, _ = act_sides("Am-242")
_, _, _, _, R242_cm, _ = act_sides("Cm-242")
DETOUR_X = max(R242_am, R242_cm) + 20
add(path((R242_am, ROW_Y["Am242"]),
         (DETOUR_X, ROW_Y["Am242"]),
         (DETOUR_X, ROW_Y["Cm"]),
         (R242_cm, ROW_Y["Cm"])), "edge-decay")


# ============================================================================
# Long alpha-decay corridors (red)
# ============================================================================
# Cm -> Pu alpha lives in a band ABOVE the Cm row (between legend y=86 and
# Cm box top y=125).  4 corridors stacked 7 px apart so they don't collide
# with the legend or each other.
CORR_CM = {
    "Cm-242": 119,
    "Cm-243": 112,
    "Cm-244": 105,
    "Cm-245":  98,
}


def cm_to_pu_alpha(cm_name, pu_name):
    cx_src, _, _, T_src, _, _ = act_sides(cm_name)
    cx_dst, _, _, T_dst, _, _ = act_sides(pu_name)
    y_corr = CORR_CM[cm_name]
    add(path((cx_src, T_src),
             (cx_src, y_corr),
             (cx_dst, y_corr),
             (cx_dst, T_dst)), "edge-decay")


cm_to_pu_alpha("Cm-242", "Pu-238")
cm_to_pu_alpha("Cm-243", "Pu-239")
cm_to_pu_alpha("Cm-244", "Pu-240")
cm_to_pu_alpha("Cm-245", "Pu-241")


# Pu -> U (alpha) and Am -> Np (alpha) share the Pu/Np gap (y in 415..485).
# Stacked staircase, ~7-8 px apart.
def pu_to_u_alpha(src, dst, y_corr, drop_x=None):
    sx, sy, sL, sT, sR, sB = act_sides(src)
    tx, ty, tL, tT, tR, tB = act_sides(dst)
    exit_x = sx - 15        # offset LEFT, away from Np->Pu decay arrow at +20
    if drop_x is None:
        # L-then-L: drop in target column from above
        add(path((exit_x, sB),
                 (exit_x, y_corr),
                 (tx, y_corr),
                 (tx, tT)), "edge-decay")
    else:
        # Detour drop via column gap, enter target from LEFT mid
        add(path((exit_x, sB),
                 (exit_x, y_corr),
                 (drop_x, y_corr),
                 (drop_x, ty),
                 (tL, ty)), "edge-decay")


def am_to_np_alpha(src, dst, y_corr):
    sx, sy, sL, sT, sR, sB = act_sides(src)
    tx, ty, tL, tT, tR, tB = act_sides(dst)
    drop_x_src = sL - 31     # gap between adjacent Pu columns
    drop_x_dst = tx - 20     # offset LEFT to clear Np->Pu decay/cap arrows
    add(path((sL, sy),
             (drop_x_src, sy),
             (drop_x_src, y_corr),
             (drop_x_dst, y_corr),
             (drop_x_dst, tT)), "edge-decay")


pu_to_u_alpha("Pu-238", "U-234", 422)
pu_to_u_alpha("Pu-239", "U-235", 430)
pu_to_u_alpha("Pu-240", "U-236", 438)
pu_to_u_alpha("Pu-242", "U-238", 468, drop_x=565)

am_to_np_alpha("Am-241",  "Np-237", 446)
am_to_np_alpha("Am-242m", "Np-238", 454)
am_to_np_alpha("Am-243",  "Np-239", 462)


# ============================================================================
# (n,2n) edges (green dashed) - loop UNDER the U row
# ============================================================================
def n2n_u(src, dst, corr_y):
    sx, _, _, _, _, sB = act_sides(src)
    dx, _, _, _, _, dB = act_sides(dst)
    add(path((sx + 18, sB),
             (sx + 18, corr_y),
             (dx + 18, corr_y),
             (dx + 18, dB)), "edge-n2n")


n2n_u("U-235", "U-234", 680)
n2n_u("U-238", "Np-237", 680)


# ============================================================================
# Fission product section
# ============================================================================
def fp_horiz(name_from, name_to, dy=0.0):
    _, fy, _, _, fR, _ = fp_sides(name_from)
    _, _, tL, _, _, _ = fp_sides(name_to)
    add(path((fR, fy + dy), (tL, fy + dy)), "edge-cap")


fp_horiz("Nd-147", "Nd-148")
fp_horiz("Nd-148", "Nd-149")

# --- Pm-147 (n,gamma) -> Pm-148 / Pm-148m: SYMMETRIC vertical branch ------
_, cy_147, _, _, R147, _ = fp_sides("Pm-147")
_, cy_148, L148, _, _, _ = fp_sides("Pm-148")
_, cy_148m, L148m, _, _, _ = fp_sides("Pm-148m")
KNEE_PM = R147 + 50

add(path((R147, cy_147 - 5), (KNEE_PM, cy_147 - 5),
         (KNEE_PM, cy_148),  (L148, cy_148)), "edge-cap")
add(path((R147, cy_147 + 5), (KNEE_PM, cy_147 + 5),
         (KNEE_PM, cy_148m), (L148m, cy_148m)), "edge-cap")

# Pm-148 (n,gamma) -> Pm-149.  Exit at -7 above Pm-148 center so the cap is
# visibly offset from the Pm-148 -> Sm-148 feeder which exits at +7.  Same
# offset pattern as Pm-147's branched exit.
_, _, _, _, R148R, _ = fp_sides("Pm-148")
_, cy_149, L149, _, _, _ = fp_sides("Pm-149")
KNEE_PM148 = R148R + 35
add(path((R148R,      cy_148 - 7),
         (KNEE_PM148, cy_148 - 7),
         (KNEE_PM148, cy_149 - 10),
         (L149,       cy_149 - 10)), "edge-cap")

# Pm-148m (n,gamma) -> Pm-149 (offset below center)
_, _, _, _, R148mR, _ = fp_sides("Pm-148m")
KNEE_PM148m = R148mR + 35
add(path((R148mR, cy_148m),
         (KNEE_PM148m, cy_148m),
         (KNEE_PM148m, cy_149 + 10),
         (L149, cy_149 + 10)), "edge-cap")

# Sm-147 (n,gamma) -> Sm-148.  With Sm-148 now in the mass-148 column,
# this is just a plain horizontal across the Sm row.
_, _, _, _, R_Sm147, _ = fp_sides("Sm-147")
cx_Sm148, cy_Sm148, L_Sm148, T_Sm148, R_Sm148, B_Sm148 = fp_sides("Sm-148")
add(path((R_Sm147, cy_Sm148), (L_Sm148, cy_Sm148)), "edge-cap")

# Sm-148 (n,gamma) -> Sm-149: plain horizontal across Sm row
_, _, _, _, R_Sm148b, _ = fp_sides("Sm-148")
cx_Sm149, cy_Sm149, L_Sm149, T_Sm149, _, _ = fp_sides("Sm-149")
add(path((R_Sm148b, cy_Sm149), (L_Sm149, cy_Sm149)), "edge-cap")


# --- FP decays (red) -----------------------------------------------------
def fp_vert_decay(name_from, name_to, dx=0.0):
    fx, fy, _, fT, _, fB = fp_sides(name_from)
    tx, ty, _, tT, _, tB = fp_sides(name_to)
    if fy < ty:
        add(path((fx + dx, fB), (fx + dx, tT)), "edge-decay")
    else:
        add(path((fx + dx, fT), (fx + dx, tB)), "edge-decay")


fp_vert_decay("Nd-147", "Pm-147")
fp_vert_decay("Nd-149", "Pm-149")
fp_vert_decay("Pm-147", "Sm-147")

# Pm-148m -> Pm-148 (IT, 4.2%): straight up, offset right.
add(path((fp_nodes["Pm-148m"][0] + 22, fp_sides("Pm-148m")[3]),
         (fp_nodes["Pm-148m"][0] + 22, fp_sides("Pm-148")[5])), "edge-decay")

# Pm-148m + Pm-148 both decay to Sm-148.  Now that Sm-148 sits in the mass-148
# column directly below Pm-148m, the trunk is a clean *vertical* down from
# Pm-148m bottom to Sm-148 top.  Pm-148 feeds into the trunk via a side-route
# around Pm-148m on its right.
#
#   Pm-148 ─┐
#           │
#  Pm-148m  │
#      │ <──┘   (feeder joins the trunk at y=1030)
#      ▼
#   Sm-148
#
_, _, _, _, R_Pm148, _  = fp_sides("Pm-148")
_, _, _, _, R_Pm148m, _ = fp_sides("Pm-148m")
_, _, _, _, _, B_Pm148m = fp_sides("Pm-148m")
cx_148m = fp_nodes["Pm-148m"][0]                # 390
TRUNK_X = cx_148m
TRUNK_Y_TOP = B_Pm148m                          # 1005
TRUNK_Y_BOT = T_Sm148                           # 1055
FEED_JOIN_Y = 1030                              # midway in the 50 px gap
FEED_X      = R_Pm148m + 20                     # right of Pm-148m's right edge

# Trunk: single arrow into Sm-148 TOP center
add(path((TRUNK_X, TRUNK_Y_TOP),
         (TRUNK_X, TRUNK_Y_BOT)), "edge-decay")

# Feeder from Pm-148.  Exits Pm-148 RIGHT at +7 below center so it is visibly
# offset from the Pm-148 -> Pm-149 capture exit (which we put at -7).
add(path((R_Pm148,    cy_148 + 7),
         (FEED_X,     cy_148 + 7),
         (FEED_X,     FEED_JOIN_Y),
         (TRUNK_X,    FEED_JOIN_Y)), "edge-decay-feed")

# Pm-149 -> Sm-149 (beta).  Same column now -> straight vertical, 0 bends.
_, _, _, _, _, B_Pm149 = fp_sides("Pm-149")
cx_Pm149 = fp_nodes["Pm-149"][0]
add(path((cx_Pm149, B_Pm149), (cx_Sm149, T_Sm149)), "edge-decay")


# --- FP feed (blue dashed) ------------------------------------------------
def fp_feed(target, length=78):
    _, ty, tL, _, _, _ = fp_sides(target)
    add(path((tL - length, ty), (tL, ty)), "edge-fp")


fp_feed("Nd-147")
fp_feed("Pm-149")


# ============================================================================
# I-135 / Xe-135m / Xe-135 section
# ============================================================================
ix_cx, ix_cy, ix_L, ix_T, ix_R, ix_B = xe_sides("I-135")
m_cx, m_cy, m_L, m_T, m_R, m_B = xe_sides("Xe-135m")
g_cx, g_cy, g_L, g_T, g_R, g_B = xe_sides("Xe-135")

KNEE_IXE = ix_R + 92  # vertical knee for both branch arrows (symmetric)

# I-135 -> Xe-135m (16.5%) — top branch
add(path((ix_R, ix_cy - 10),
         (KNEE_IXE, ix_cy - 10),
         (KNEE_IXE, m_cy),
         (m_L, m_cy)), "edge-decay")
# I-135 -> Xe-135 (83.5%) — bottom branch
add(path((ix_R, ix_cy + 10),
         (KNEE_IXE, ix_cy + 10),
         (KNEE_IXE, g_cy),
         (g_L, g_cy)), "edge-decay")

# Xe-135m -> Xe-135 (IT)
add(path((m_cx, m_B), (g_cx, g_T)), "edge-decay")

# I-135 capture loops (depletion topology) entering Xe-135m TOP and Xe-135 BOT
add(path((ix_cx, ix_T),
         (ix_cx, ix_T - 30),
         (m_cx - 8, ix_T - 30),
         (m_cx - 8, m_T)), "edge-cap")
add(path((ix_cx, ix_B),
         (ix_cx, ix_B + 32),
         (g_cx + 8, ix_B + 32),
         (g_cx + 8, g_B)), "edge-cap")

# Xe-135 -> loss (Cs-135 beta) and (n,gamma) absorption
add(path((g_R, g_cy - 8), (1595, g_cy - 8)), "edge-decay")
add(path((g_R, g_cy + 8),
         (1525, g_cy + 8),
         (1525, g_cy - 60),
         (1595, g_cy - 60)), "edge-loss")

# FP feeds into the I/Xe section.  Offset the Xe-135m / Xe-135 feeds away
# from the I-135 -> Xe branch arrows that already terminate at center-y, so
# the dashed feed line doesn't get hidden under the solid red branch line.
add(path((ix_L - 78, ix_cy), (ix_L, ix_cy)), "edge-fp")
add(path((m_L - 60, m_cy - 14), (m_L, m_cy - 14)), "edge-fp")
add(path((g_L - 60, g_cy + 14), (g_L, g_cy + 14)), "edge-fp")


# ============================================================================
# Render
# ============================================================================
def boxes_actinide():
    out = []
    for name, (cx, cy) in actinides.items():
        L, T, R, B = rect(cx, cy, ACT_BOX_W, ACT_BOX_H)
        out.append(f'<rect class="nodebox" x="{L}" y="{T}" width="{ACT_BOX_W}" height="{ACT_BOX_H}"/>')
        out.append(f'<text class="isoname" style="font-size:17.5px" x="{cx}" y="{cy - 4}">{name}</text>')
        out.append(f'<text class="lambda"  style="font-size:9.5px"  x="{cx}" y="{cy + 18}">λ = {LAMBDA[name]}</text>')
    return "\n".join(out)


def boxes_fp():
    out = []
    for name, (cx, cy) in fp_nodes.items():
        L, T, R, B = rect(cx, cy, FP_BOX_W, FP_BOX_H)
        out.append(f'<rect class="nodebox" x="{L}" y="{T}" width="{FP_BOX_W}" height="{FP_BOX_H}"/>')
        out.append(f'<text class="isoname" style="font-size:18px" x="{cx}" y="{cy - 4}">{name}</text>')
        out.append(f'<text class="lambda"  style="font-size:10.5px" x="{cx}" y="{cy + 18}">λ = {LAMBDA[name]}</text>')
    return "\n".join(out)


def boxes_xe():
    out = []
    for name, (cx, cy) in ixe_nodes.items():
        L, T, R, B = rect(cx, cy, XE_BOX_W, XE_BOX_H)
        out.append(f'<rect class="nodebox" x="{L}" y="{T}" width="{XE_BOX_W}" height="{XE_BOX_H}"/>')
        out.append(f'<text class="isoname" style="font-size:15.5px" x="{cx}" y="{cy - 4}">{name}</text>')
        out.append(f'<text class="lambda"  style="font-size:9.2px"  x="{cx}" y="{cy + 18}">λ = {LAMBDA[name]}</text>')
    return "\n".join(out)


# Branching ratio labels.  We render each as a halo (white stroke, no fill)
# *under* a fill (no stroke) so renderers without `paint-order` still show
# legible text.
ratio_labels = [
    ("0.173", 1218, 235, "ratio-red"),  # Am-242 -> Cm-242 detour
    ("0.827", 1156, 360, "ratio-red"),  # Am-242 -> Pu-242 short vertical
    ("0.528",  R147 + 60, 893, "ratio"),
    ("0.472",  R147 + 60, 933, "ratio"),
    ("0.042", fp_nodes["Pm-148m"][0] + 36, 910, "ratio-red"),
    # 0.958 sits just LEFT of the Pm-148m -> Sm-148 trunk, in the gap below
    # Pm-148m where it doesn't collide with the feeder horizontal at y=1030.
    ("0.958", fp_nodes["Pm-148m"][0] - 17, 1018, "ratio-red"),
    ("0.165", KNEE_IXE + 6, 800, "ratio-red"),
    ("0.835", KNEE_IXE + 6, 850, "ratio-red"),
]

def ratio_svg_for(text, x, y, cls):
    color = "#b91c1c" if cls == "ratio-red" else "#111827"
    halo = (f'<text class="ratio-halo" x="{x}" y="{y}">{text}</text>')
    fill = (f'<text class="ratio-fill" x="{x}" y="{y}" fill="{color}">{text}</text>')
    return halo + "\n" + fill


ratio_svg = "\n".join(ratio_svg_for(t, x, y, c) for (t, x, y, c) in ratio_labels)
edge_svg = "\n".join(f'<path class="{cls}" d="{d}"/>' for d, cls in edges)


SVG = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <defs>
    <marker id="arrow-cap"   markerWidth="4.2" markerHeight="4.2" refX="3.8" refY="2.1" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L4.2,2.1 L0,4.2 Z" fill="#1f2937"/></marker>
    <marker id="arrow-decay" markerWidth="4.2" markerHeight="4.2" refX="3.8" refY="2.1" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L4.2,2.1 L0,4.2 Z" fill="#dc2626"/></marker>
    <marker id="arrow-fp"    markerWidth="4.2" markerHeight="4.2" refX="3.8" refY="2.1" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L4.2,2.1 L0,4.2 Z" fill="#2563eb"/></marker>
    <marker id="arrow-n2n"   markerWidth="4.2" markerHeight="4.2" refX="3.8" refY="2.1" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L4.2,2.1 L0,4.2 Z" fill="#16a34a"/></marker>
    <marker id="arrow-loss"  markerWidth="4.2" markerHeight="4.2" refX="3.8" refY="2.1" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L4.2,2.1 L0,4.2 Z" fill="#6b7280"/></marker>
  </defs>
  <style>
    .title    {{ font: 700 26px Arial, sans-serif; fill:#111827; }}
    .subtitle {{ font: 13px Arial, sans-serif; fill:#4b5563; }}
    .rowlabel {{ font: 700 24px Arial, sans-serif; fill:#374151; }}
    .nodebox  {{ fill:#ffffff; stroke:#1f4e79; stroke-width:2.1; rx:3; }}
    .isoname  {{ font-family: Arial, sans-serif; font-weight:700; fill:#111827; text-anchor:middle; dominant-baseline:middle; }}
    .lambda   {{ font-family: Arial, sans-serif; fill:#4b5563; text-anchor:middle; dominant-baseline:middle; }}
    .edge-cap   {{ fill:none; stroke:#1f2937; stroke-width:1.9; marker-end:url(#arrow-cap);   stroke-linecap:square; stroke-linejoin:miter; }}
    .edge-decay      {{ fill:none; stroke:#dc2626; stroke-width:1.8; marker-end:url(#arrow-decay); stroke-linecap:square; stroke-linejoin:miter; }}
    .edge-decay-feed {{ fill:none; stroke:#dc2626; stroke-width:1.8; stroke-linecap:square; stroke-linejoin:miter; }}
    .edge-fp    {{ fill:none; stroke:#2563eb; stroke-width:1.7; stroke-dasharray:7 5; marker-end:url(#arrow-fp);  stroke-linecap:butt; }}
    .edge-n2n   {{ fill:none; stroke:#16a34a; stroke-width:1.8; stroke-dasharray:6 4; marker-end:url(#arrow-n2n); stroke-linecap:butt; }}
    .edge-loss  {{ fill:none; stroke:#6b7280; stroke-width:1.6; stroke-dasharray:4 4; marker-end:url(#arrow-loss); stroke-linecap:butt; }}
    .legend {{ font: 13px Arial, sans-serif; fill:#111827; }}
    .note   {{ font: 12px Arial, sans-serif; fill:#4b5563; }}
    .ratio-halo {{ font: 700 11px Arial, sans-serif; fill:none; stroke:#ffffff; stroke-width:4px; stroke-linejoin:round; text-anchor:middle; dominant-baseline:middle; }}
    .ratio-fill {{ font: 700 11px Arial, sans-serif; text-anchor:middle; dominant-baseline:middle; }}
  </style>
  <rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>

  <text class="title" x="32" y="38">Rasbery Depletion Chain</text>
  <text class="subtitle" x="32" y="60">Built from dep_decay.csv, dep_trans.csv, and XSSet::BuildTransitionMatrix. Arrow labels are intentionally omitted except selected branching ratios.</text>

  <!-- legend -->
  <path class="edge-cap"   d="M 32,86 L 86,86"/>     <text class="legend" x="98"  y="91">Capture / transmutation</text>
  <path class="edge-decay" d="M 270,86 L 324,86"/>   <text class="legend" x="336" y="91">Decay</text>
  <path class="edge-fp"    d="M 405,86 L 459,86"/>   <text class="legend" x="471" y="91">Fission product</text>
  <path class="edge-n2n"   d="M 585,86 L 639,86"/>   <text class="legend" x="651" y="91">XSSet (n,2n)</text>
  <path class="edge-loss"  d="M 760,86 L 814,86"/>   <text class="legend" x="826" y="91">Loss</text>

  <!-- section titles & row labels -->
  <text class="title" x="32" y="120">Actinide</text>
  <text class="rowlabel" x="35" y="156">Cm</text>
  <text class="rowlabel" x="35" y="276">Am</text>
  <text class="rowlabel" x="35" y="396">Pu</text>
  <text class="rowlabel" x="35" y="516">Np</text>
  <text class="rowlabel" x="35" y="636">U</text>

  <text class="title" x="32" y="710">Nd / Pm / Sm</text>
  <text class="rowlabel" x="35"  y="773">Nd</text>
  <text class="rowlabel" x="35"  y="918">Pm</text>
  <text class="rowlabel" x="35"  y="1093">Sm</text>

  <text class="title" x="1100" y="710">I / Xe</text>

  <!-- edges -->
  {edge_svg}

  <!-- branching-ratio labels (halo + fill, double-text for renderer compat) -->
  {ratio_svg}

  <!-- nodes -->
  {boxes_actinide()}
  {boxes_fp()}
  {boxes_xe()}

  <!-- footnotes -->
  <text class="note" x="1100" y="1050">Xe equilibrium overwrite remains in XSSet when xe_transient is false.</text>
  <text class="note" x="32" y="{H - 12}">Black: capture / transmutation (with implicit β where needed).  Red: decay branches.  Blue dashed: fission-product feed.  Green dashed: (n,2n).  Grey dashed: loss.</text>
</svg>
'''

import pathlib
out = pathlib.Path(__file__).resolve().parent / "depletion-chain.svg"
out.write_text(SVG)
print(f"wrote {out} ({len(SVG)} bytes)")
