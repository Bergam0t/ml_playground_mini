"""What happened when the real tool was piloted: the figures the demo is set to match.

Telephone calls to patients the live model flagged as high risk, in the Physiotherapy
service, from 24 December 2025 to 2 February 2026. Staff phoned to confirm attendance,
rearrange if needed and reassure. These are the pilot's reported figures; the derived
values below are worked out from them.

There was no randomised comparison group: "before" is how often these high-risk patients
typically missed, not a group of similar patients who were not phoned at the same time.
"""

SERVICE = "Physiotherapy"
PERIOD = "24 December 2025 to 2 February 2026"
WEEKS = 6

CALLS = 273  # patients contacted ahead of their appointments
DNA_BEFORE = 0.84  # how often these high-risk patients typically missed
DNA_AFTER = 0.17  # missed after a call, everyone contacted
DNA_CONFIRMED = 0.08  # missed after answering and confirming they'd come
EARLY_CANCELLATIONS = 29  # slots freed early, to offer to other patients
EXTRA_ATTENDED = 182  # additional appointments attended over the 6 weeks
REVENUE = 21_000  # £ of revenue the trust would otherwise have lost

# ---- Derived
# The share of those who would have missed that a call got to attend: 182 of about 229
EFFECT = EXTRA_ATTENDED / (CALLS * DNA_BEFORE)
# £ per appointment saved, about £115
VALUE_PER_APPOINTMENT = REVENUE / EXTRA_ATTENDED
# Appointments saved per call: "two for every three calls"
SAVED_PER_CALL = EXTRA_ATTENDED / CALLS
# Calls that led to an early cancellation, about 1 in 9
CANCEL_RATE = EARLY_CANCELLATIONS / CALLS
