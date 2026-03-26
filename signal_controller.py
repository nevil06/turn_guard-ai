"""
SafeTurn AI — Signal Controller Module
========================================
3-stage traffic signal state machine with ZONE-BASED logic.

States:  GREEN → ORANGE → RED (never skips)

Zone Logic:
  No pedestrians anywhere          → GREEN (free turn)
  Pedestrian in WAITING ZONE       → Wait 3s, then ORANGE → RED
  Pedestrian in CROSSING ZONE      → Immediate ORANGE → RED

Timing:
  ORANGE  = 2 seconds (fixed warning)
  RED min = 5 seconds
  RED → GREEN only when 0 pedestrians
"""

import time


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

ORANGE_DURATION  = 2.0    # ORANGE lasts exactly 2s
RED_MIN_DURATION = 5.0    # RED holds at least 5s (stable, realistic)
WAITING_PED_WAIT = 3.0    # Wait 3s before triggering for waiting pedestrian

# Colors (BGR for OpenCV)
COL_RED    = (0, 0, 255)
COL_GREEN  = (0, 200, 0)
COL_ORANGE = (0, 140, 255)


# ═══════════════════════════════════════════════════════════════════
# SIGNAL CONTROLLER — Zone-Aware
# ═══════════════════════════════════════════════════════════════════

class SignalController:
    """
    3-stage traffic signal with zone-based pedestrian logic:

        GREEN ──(crossing detected)──→ ORANGE ──(2s)──→ RED
        GREEN ──(waiting 3s)─────────→ ORANGE ──(2s)──→ RED
          ▲                                               │
          └──────(0 peds + timer done)────────────────────┘
    """

    def __init__(self):
        self.state            = "GREEN"
        self.state_start      = time.time()
        self.waiting_since    = None      # when waiting-zone ped was first seen

    def update(self, waiting_count=0, crossing_count=0):
        """
        Call every frame with zone-based pedestrian counts.

        Args:
            waiting_count:  number of peds in WAITING ZONE
            crossing_count: number of peds in CROSSING ZONE

        Returns: (state, color_bgr, remaining_seconds, status_text)
        """
        now = time.time()
        elapsed = now - self.state_start

        total_peds = waiting_count + crossing_count

        # ── GREEN ─────────────────────────────────────────────────
        if self.state == "GREEN":
            if total_peds == 0:
                # All clear
                self.waiting_since = None
                return "GREEN", COL_GREEN, 0.0, "No pedestrians — FREE TURN"

            elif crossing_count > 0:
                # Pedestrians ON the road → immediate ORANGE → RED
                self.waiting_since = None
                self._go("ORANGE", now)
                return "ORANGE", COL_ORANGE, ORANGE_DURATION, \
                    f"{crossing_count} crossing — WARNING"

            elif waiting_count > 0:
                # Pedestrians on footpath — start 3s wait timer
                if self.waiting_since is None:
                    self.waiting_since = now
                waited = now - self.waiting_since
                if waited >= WAITING_PED_WAIT:
                    self.waiting_since = None
                    self._go("ORANGE", now)
                    return "ORANGE", COL_ORANGE, ORANGE_DURATION, \
                        "Waiting pedestrian — WARNING"
                else:
                    wait_left = WAITING_PED_WAIT - waited
                    return "GREEN", COL_GREEN, wait_left, \
                        f"{waiting_count} waiting — watching ({wait_left:.1f}s)"

        # ── ORANGE ────────────────────────────────────────────────
        elif self.state == "ORANGE":
            if elapsed < ORANGE_DURATION:
                remaining = ORANGE_DURATION - elapsed
                return "ORANGE", COL_ORANGE, remaining, \
                    "Warning — prepare to stop"
            else:
                self._go("RED", now)
                return "RED", COL_RED, RED_MIN_DURATION, \
                    "STOP — pedestrians crossing"

        # ── RED ───────────────────────────────────────────────────
        elif self.state == "RED":
            if elapsed < RED_MIN_DURATION:
                remaining = RED_MIN_DURATION - elapsed
                return "RED", COL_RED, remaining, \
                    "STOP — pedestrians crossing"
            if total_peds == 0:
                self._go("GREEN", now)
                return "GREEN", COL_GREEN, 0.0, "No pedestrians — FREE TURN"
            else:
                return "RED", COL_RED, 0.0, \
                    f"{total_peds} pedestrian(s) — holding RED"

        return self.state, COL_GREEN, 0.0, ""

    def _go(self, new_state, now):
        self.state       = new_state
        self.state_start = now


class SimpleSignalController:
    """
    Simplified signal controller based purely on total pedestrian count.
    Used by dual camera mode without spatial zones.

    0 peds: GREEN
    1 ped: Wait 5s, then ORANGE(2s) -> RED
    2+ peds: Immediate ORANGE(2s) -> RED
    """
    def __init__(self):
        self.state            = "GREEN"
        self.state_start      = time.time()
        self.waiting_since    = None

    def update(self, ped_count):
        now = time.time()
        elapsed = now - self.state_start

        if self.state == "GREEN":
            if ped_count == 0:
                self.waiting_since = None
                return "GREEN", COL_GREEN, 0.0, "No pedestrians"
            
            elif ped_count >= 2:
                self.waiting_since = None
                self._go("ORANGE", now)
                return "ORANGE", COL_ORANGE, ORANGE_DURATION, f"{ped_count} pedestrians — WARNING"
                
            elif ped_count == 1:
                if self.waiting_since is None:
                    self.waiting_since = now
                waited = now - self.waiting_since
                if waited >= 5.0:
                    self.waiting_since = None
                    self._go("ORANGE", now)
                    return "ORANGE", COL_ORANGE, ORANGE_DURATION, "1 pedestrian — WARNING"
                else:
                    return "GREEN", COL_GREEN, 5.0 - waited, f"1 pedestrian — watching ({5.0 - waited:.1f}s)"

        elif self.state == "ORANGE":
            if elapsed < ORANGE_DURATION:
                return "ORANGE", COL_ORANGE, ORANGE_DURATION - elapsed, "Prepare to stop"
            else:
                self._go("RED", now)
                return "RED", COL_RED, RED_MIN_DURATION, "STOP — pedestrians"

        elif self.state == "RED":
            if elapsed < RED_MIN_DURATION:
                return "RED", COL_RED, RED_MIN_DURATION - elapsed, "STOP — pedestrians"
            if ped_count == 0:
                self._go("GREEN", now)
                return "GREEN", COL_GREEN, 0.0, "No pedestrians"
            else:
                return "RED", COL_RED, 0.0, f"{ped_count} pedestrian(s) — holding RED"

        return self.state, COL_GREEN, 0.0, ""

    def _go(self, new_state, now):
        self.state       = new_state
        self.state_start = now
