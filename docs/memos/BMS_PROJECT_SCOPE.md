# Project: Degradation-Aware BMS + DC-DC Testbed for Residential Grid Storage

**One-liner.** Open-source BMS + synchronous-buck DC-DC stack for a 4S Li-ion pack, controlled by an RL dispatcher that minimizes rainflow degradation. Bridges hardware (STM32 + analog frontend + MOSFET converter) with the existing `battery_gym` RL work.

## Why it hits the JD

| JD bullet | Project deliverable |
|---|---|
| BMS design + validation | 4S Li-ion BMS w/ bq76920 analog frontend, cell balance, OV/UV/OC trip |
| DC-DC architectures + control | Synchronous buck-boost; bench compare hard-switched vs ZVS; PI + Type-III compensator; bode plot from frequency-response analyzer |
| Prototyping | KiCad schematic + 4-layer PCB; BoM ~$300 |
| Test + FMEA + safety | Hazard analysis (ISO 26262 spirit), fault injection (sense break, cell short, MOSFET stuck), thermal IR-cam + waveform capture |
| HW/FW/SW debug | STM32 firmware (HAL), Python host tool, JTAG/SWD debug logs |
| Docs | Requirements doc, app note, datasheet-style manuscript, GitHub README |
| Battery chemistry knowledge | Electrochemical OCV-R model, SoC/SoH estimator, rainflow degradation tracker (already done in `battery_gym`) |
| Grid-tied storage | DC-load emulates inverter; PJM-style regulation signal driver from `battery_gym/reg_signal.py` |

## Scope (~5 months part-time)

### Phase 1 — Simulation (3 weeks)
- LTspice DC-DC topology comparison: synchronous buck, bidirectional buck-boost, dual active bridge. Loss decomposition (conduction / switching / magnetics).
- PLECS or PySpice closed-loop with battery model.
- Reuse `battery_gym` env: pack-level rainflow + DoD histograms.
- **Deliverable:** topology trade study PDF, .asc files, plots.

### Phase 2 — Control loop (3 weeks)
- Type-III compensator design for current-mode buck (Bode + phase margin >45°).
- Cell-balance state machine (passive shunt).
- SoC estimator: coulomb counting + EKF on OCV-R.
- RL dispatcher (port `agents.ELMQAgent`) sits above BMS, sets pack power setpoint.
- **Deliverable:** Python sim showing PI vs Type-III step response; closed-loop notebook.

### Phase 3 — Hardware build (8 weeks)
- 4S 18650 pack (~14.4V nominal, 10Ah, $80), holders + nickel strip.
- bq76920 analog frontend ($15) for cell sense + balance + protection.
- STM32G4 Nucleo ($25) — has built-in HRTIM for DC-DC.
- Power stage: 2× IRFB7430 + LM5106 driver + 22µH ferrite + 4× 220µF caps (~$50).
- 4-layer KiCad PCB ($60 JLCPCB + assembly).
- DC electronic load on output ($150 used Riden) emulates inverter sink.
- **Deliverable:** Working board, scope captures of switching node, efficiency curve.

### Phase 4 — Test + FMEA (3 weeks)
- Test plan doc: functional, performance, safety.
- FMEA matrix: 20+ failure modes ranked by RPN (severity × occurrence × detection).
- Fault injection: pull cell sense wire, short low-side MOSFET gate, OCP threshold validation.
- Thermal: FLIR or thermistor sweep, derating curve.
- **Deliverable:** Test report PDF with measured vs spec table, FMEA spreadsheet.

### Phase 5 — Documentation (2 weeks)
- Requirements spec (functional, electrical, safety, environmental).
- App note: "Degradation-Aware Dispatch for Residential BESS — combining ELM-RL control with bq76920 BMS."
- Datasheet-style manuscript for the board.
- arXiv preprint linking to `battery_gym` paper.

## Differentiators vs typical BMS portfolio project

1. **RL dispatch on top of BMS** — most BMS demos do balancing only. This one ties degradation-aware control (existing rainflow + RL work) into the physical dispatch loop. Novel angle for a hire pitch.
2. **Reuses arXiv:2601.22865v2 reproduction** — env, rainflow tracker, agents already present. Hardware just executes setpoints.
3. **Open-source full stack** — schematic, firmware, host tool, control sim on GitHub. Few BMS portfolio projects publish PCB + firmware + control theory together.

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| Limited hands-on EE experience | Stay LV (<48V DC). No grid AC. DC-load emulates inverter. |
| PCB respin needed | Budget 2 spins ($120 total). Thorough DRC + LTspice first. |
| Li-ion fire | LiFePO4 instead of NMC ($120 vs $80, much safer). Test in fire-safe box outdoors. |
| Scope explosion | Cut DAB topology, ship only sync-buck. Cut EKF, ship coulomb counting. |

## Min-viable cut (3 months)

If hardware time tight, ship:
- Sim phase complete.
- bq76920 eval board + STM32 dev board (no custom PCB).
- RL dispatcher controlling DC-load setpoint over UART.
- App note + arXiv preprint.

Drops PCB + Type-III + thermal characterization. Still hits BMS / FW/SW / battery / control / docs JD bullets. Misses pure DC-DC topology design + hands-on PCB.

## Budget

| Tier | Parts | Tools (have/borrow) |
|---|---|---|
| Min-viable | $250 (eval boards + cells + load) | Scope, soldering iron |
| Full | $500 (above + custom PCB ×2 spins + magnetics + extra MOSFETs) | + FLIR cam ($300 used), bench supply |

## Open questions

1. Hardware budget ceiling? $250 vs $500 changes scope.
2. Months-of-runway on this vs FAR.AI app prep? 5 months PT is real.
3. Target role: this JD specifically, or class of BMS roles? Tailoring app note matters.
4. Any existing LBL bench access (scope, load, iron)? Cuts capex 50%.
