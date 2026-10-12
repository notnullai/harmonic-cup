# Pump P-101 — Operation and Maintenance Manual

> **Synthetic document.** This manual was written for the Harmonic project. It
> is not a vendor document and carries no third-party text. The machine it
> describes is the reference machine of `sim/pump_sim.py` and the brain profile
> in `engine/tools/mint_brain.cpp`: a horizontal end-suction centrifugal pump,
> 2950 rpm nominal, ISO 10816-1 class III, deep-groove ball bearings at both
> ends. Section numbers are stable and are the citation tokens the GenAI layer
> is allowed to use.

Doc id: `P-101-OM` · Revision C · 34 sections

---

## 1. Machine description

### 1.1 Identification and duty

Pump P-101 is a horizontal end-suction centrifugal pump driven by a 45 kW
four-pole induction motor through a flexible jaw coupling. Nominal shaft speed
is 2950 rpm, which places the running frequency at 49.2 Hz. Rated duty is
120 m³/h against 4.2 bar discharge at 0.65 load factor. The pump is classed as
a machine under ISO 10816-1 class III, rigid foundation. All vibration
limits quoted in this manual are velocity RMS in mm/s measured radially at the
bearing housings over the 10–1000 Hz band.

### 1.2 Bearing arrangement

Both ends carry deep-groove ball bearings with nine rolling elements, a ball
diameter of 7.94 mm and a pitch diameter of 38.5 mm, mounted at a nominal
contact angle of zero degrees. The drive-end bearing takes the radial load from
the impeller; the non-drive end is the locating bearing and takes axial thrust.
These geometry figures are what set the defect frequencies in section 5.2, and
they must be re-entered in the monitoring configuration whenever a bearing of a
different type is fitted.

### 1.3 Instrumentation

The machine carries a radial accelerometer at each bearing housing sampled at
25.6 kHz, an axial accelerometer at the drive end, RTDs in both bearing
housings and in the motor winding, pressure transmitters on the suction and
discharge flanges, a magnetic flow meter on the discharge line, and a motor
current transducer. The slow channels are sampled once per second. Vibration is
integrated to velocity in the acquisition chain, so all amplitudes in this
manual are velocity unless a section says otherwise.

### 1.4 Operating regimes

The pump runs in four steady regimes and one transient. `IDLE` is below 500 rpm.
`STARTUP` is any period where shaft speed is changing by more than 100 rpm per
second, regardless of the speed reached. `NOMINAL` is 2800–3000 rpm at a load
factor between 0.5 and 0.8. `HIGH_LOAD` is the same speed band above 0.8 load.
`LOW_LOAD` is the same speed band below 0.5 load, which is the regime most
associated with recirculation and cavitation. Baselines are held separately per
regime because the same vibration level means different things in each.

---

## 2. Safety

### 2.1 Lockout and tagout

Isolate the motor at the local disconnect, apply a personal lock and tag, and
verify zero energy at the motor terminals before any mechanical work. The
locking-out step is not optional for inspection work: a pump on automatic level
control can start without warning. Prove the isolation by attempting a local
start after the lock is applied.

### 2.2 Pressure and fluid hazards

Close and lock the suction and discharge isolation valves, then vent the casing
through the high-point vent before breaking any flanged joint. The casing can
hold pressure after the valves are shut. Where the pumped fluid is hot, allow
the casing to cool below 40 °C before draining, and drain to a closed system
where the fluid is hazardous.

### 2.3 Hot surfaces and rotating parts

Bearing housings can exceed 80 °C in normal running and the motor frame can
exceed 70 °C. Never remove the coupling guard while the machine is energised.
The guard is also the alignment access cover, so alignment work always follows
the lockout in section 2.1.

### 2.4 Lifting

The bare pump weighs 310 kg and the motor 295 kg. Lift the pump by the casing
lugs, never by the suction nozzle or the coupling. Support the rotor during
bearing work so the shaft does not sit on the seal faces.

---

## 3. Lubrication

### 3.1 Lubrication schedule

Both bearings are grease-lubricated with a lithium-complex NLGI 2 grease.
Regrease every 2000 running hours with 18 g per bearing at the drive end and
14 g at the non-drive end. Purge the old grease through the relief port with
the machine running at nominal speed; regreasing a stationary bearing packs the
cavity and raises the running temperature. Record every regrease against the
running-hour counter, because the interval is the commonest thing to lose track
of after a shutdown.

### 3.2 Grease compatibility

Do not mix lithium-complex grease with polyurea or calcium-sulphonate grease.
Mixed thickeners soften, the base oil bleeds out and the bearing runs dry with
no visible change in the fill level. If the grease in service is unknown, flush
the cavity completely as in section 3.4 rather than topping up.

### 3.3 Over-greasing

An over-greased bearing churns, and the churning shows as a rising bearing
temperature with little change in vibration — typically 10 to 20 °C above the
regime baseline within an hour of the regrease. Relieve the excess by running
with the relief port open for thirty minutes. Persistent over-greasing damages
the cage and eventually produces the wear pattern in section 5.4.

### 3.4 Grease flush procedure

Isolate per section 2.1. Remove the relief plug and the fill plug, inject fresh
grease at the fill port until clean grease emerges at the relief port, refit
both plugs, and run at nominal for thirty minutes with the relief port open.
Allow two hours for the bearing temperature to settle before comparing it
against the baseline. Budget 45 minutes per bearing.

### 3.5 Lubricant contamination

Water or process fluid in the grease destroys the load-carrying film and
produces outer-race damage within a few hundred running hours. Suspect
contamination when the bearing temperature rises alongside broadband vibration
and the machine has recently run with a leaking seal. Contaminated grease is a
cause of outer-race wear, not a symptom of it, and replacing the bearing
without curing the ingress path repeats the failure.

---

## 4. Mechanical maintenance

### 4.1 Bearing inspection

Inspect the bearings whenever the overall vibration at a bearing housing
exceeds 4.5 mm/s RMS, when the bearing temperature runs more than 15 °C above
the regime baseline, or at 16000 running hours, whichever comes first. Listen
with a stethoscope at the housing: a regular tick at shaft speed indicates a
local defect, while a rough continuous hiss indicates general wear or
contamination.

### 4.2 Bearing replacement procedure

Isolate per section 2.1 and drain per section 2.2. Remove the coupling guard,
mark the coupling halves for reassembly and withdraw the coupling with a
two-leg puller. Remove the bearing end cover, release the locknut and tab
washer, and withdraw the bearing with a bearing puller applied to the inner
ring only. Never drive the replacement bearing on through the outer ring:
induction-heat the new bearing to 110 °C and slide it onto the shaft, then hold
it against the shoulder until it grips. Refit the tab washer and locknut, pack
the housing to one third with the grease in section 3.1, refit the end cover,
and refit the coupling to the marks. Realign per section 4.4 before removing
the lockout. Allow three hours, technician level, with a bearing puller and an
induction heater.

### 4.3 Bearing fitting tolerances

The shaft seat is 45 mm k5 and the housing bore is 85 mm H6. Check both before
fitting: a worn shaft seat lets the inner ring creep, which produces a
fretting pattern and a vibration signature that returns within weeks of a
bearing change. A housing bore above tolerance allows the outer ring to turn
and polishes the bore. Neither fault is cured by fitting another bearing.

### 4.4 Shaft alignment

Align cold with a laser alignment system, targeting 0.05 mm parallel offset and
0.05 mm per 100 mm angular at the coupling. Shim under all four motor feet and
check for soft foot before correcting the alignment: a soft foot makes the
readings change as the hold-down bolts are tightened, and correcting alignment
onto a soft foot builds the error into the machine. Recheck hot after two hours
at nominal duty and record both sets of readings. Allow two hours, technician
level.

### 4.5 Coupling inspection

Inspect the jaw coupling elastomer at every alignment. A hardened, cracked or
powdered element transmits misalignment forces straight into both bearings and
raises the 2X vibration component without any change in the alignment readings
themselves. Replace the element in pairs and never mix durometer grades.

### 4.6 Mechanical seal replacement

Isolate and drain per section 2. Remove the seal gland bolts, withdraw the
gland and the rotating assembly, and inspect the shaft sleeve for scoring.
Fit the new seal dry-handed, with the faces protected until the last moment.
Set the working length from the sleeve shoulder to the dimension on the seal
drawing. Pressure-test to 1.5 times working pressure before returning to
service. Allow two hours, technician level.

### 4.7 Impeller inspection

Withdraw the casing cover and inspect the impeller vanes for pitting on the
low-pressure side of the leading edge, which is the signature of cavitation
damage rather than of erosion by solids. Measure the wear-ring clearance: the
design clearance is 0.35 mm diametral and the replacement limit is 0.70 mm.
Worn wear rings raise internal recirculation, which lowers delivered flow at
constant speed and mimics a seal leak on the flow channel alone.

### 4.8 Foundation and hold-down

Check the grout for cracking and the hold-down bolts for torque annually.
A loose hold-down produces a vibration signature at 2X and 3X shaft speed with
a strongly directional character, which is easily confused with misalignment
and is distinguished by the fact that alignment readings are within tolerance.

---

## 5. Condition monitoring

### 5.1 Vibration severity zones

ISO 10816-1 class III sets four zones by velocity RMS over 10–1000 Hz. Zone A,
below 1.8 mm/s, is a newly commissioned machine. Zone B, 1.8 to 4.5 mm/s, is
acceptable for unrestricted long-term operation. Zone C, 4.5 to 11.2 mm/s, is
unsatisfactory for continuous operation and the machine should be scheduled for
correction. Zone D, above 11.2 mm/s, is severe enough to cause damage and the
machine should be stopped. A machine of 28 mm/s or more is in immediate danger.
These are absolute limits; a rise within a zone is still significant if it is a
rise against that machine's own baseline.

### 5.2 Bearing defect frequencies

For the bearing in section 1.2 at 2950 rpm, the ball-pass frequency of the
outer race (BPFO) is 151.3 Hz, the ball-pass frequency of the inner race (BPFI)
is 291.2 Hz, and the ball-spin frequency (BSF) is 118.7 Hz. These scale
linearly with shaft speed and must be recomputed whenever the speed changes.
Note that BPFO at 151.3 Hz sits within 4 Hz of the third shaft harmonic at
147.5 Hz: any spectrum used to separate them needs a resolution finer than
about 2 Hz, which at 25.6 kHz means a transform of at least 16384 samples.

### 5.3 Envelope analysis

Rolling-element defects are impacts, and impacts show far more clearly in the
envelope of the high-frequency band than in the raw velocity spectrum. Apply
envelope demodulation and look for a line at the defect frequency with
sidebands at shaft speed. An envelope peak at BPFO with shaft-speed sidebands
is the definitive indication of outer-race wear; the same peak without
sidebands more often indicates a lubrication problem than a spall.

### 5.4 Outer-race wear

Elevated amplitude at BPFO in the envelope spectrum indicates outer-race wear
with high confidence. The overall velocity RMS rises as the defect develops,
crossing into zone C in the later stages, and the bearing temperature follows
with a lag of days to weeks. Causes are loss or degradation of lubricant,
contamination of the grease with water or process fluid, and a shaft or housing
seat outside the tolerances in section 4.3. The mitigation is bearing
replacement per section 4.2, preceded where contamination is suspected by the
flush in section 3.4.

### 5.5 Inner-race wear

Elevated amplitude at BPFI indicates inner-race wear. The inner race rotates
with the shaft, so the defect passes in and out of the load zone once per
revolution and the envelope line carries strong sidebands spaced at shaft
speed. Inner-race defects usually follow a fitting fault or a shock load rather
than lubrication, and a machine that has been dropped or driven onto its
bearings is the usual history.

### 5.6 Misalignment

Misalignment raises the second shaft harmonic, and the diagnostic quantity is
the ratio of 2X to 1X rather than the amplitude of either: a 2X-to-1X ratio
above 0.5 indicates misalignment, and above 1.0 indicates it strongly. A third
harmonic rises with it when the misalignment is angular. The overall velocity
level may stay inside zone A throughout, which is why a machine can be
significantly misaligned and still pass an absolute vibration check. A strong
axial component at 1X and 2X distinguishes angular from parallel misalignment.
Causes are a settled or soft foot, thermal growth of the motor not allowed for
in the cold alignment, pipe strain on the casing nozzles, and a coupling
element that has hardened. Mitigation is the alignment in section 4.4, after
the soft-foot check and after the coupling inspection in section 4.5.

### 5.7 Imbalance

Imbalance raises the first shaft harmonic with little change at 2X, and the
response is radial and roughly equal in all radial directions. It is separated
from misalignment by the 2X-to-1X ratio staying low. Causes are product
build-up on the impeller, erosion of one vane, and a lost balance weight.
Mitigation is cleaning or in-situ balancing.

### 5.8 Cavitation

Cavitation is a random process, not a periodic one: collapsing vapour bubbles
ring the casing and raise a broadband, spiky noise floor. The tell is
impulsiveness rather than loudness — kurtosis above 4.0 and crest factor above
5.0 — accompanied by an oscillation on the suction pressure at roughly 0.3 Hz
and a fall in suction pressure relative to the regime baseline. Most of the
energy sits above the 1000 Hz band that the ISO overall level measures, which
is why overall velocity RMS is a poor cavitation detector and why the
monitoring set carries kurtosis and crest alongside it. Causes are insufficient
NPSH available, a blocked or partly closed suction strainer, a suction lift
that has increased with falling tank level, and operation far below the best
efficiency point. Mitigation is to restore NPSH: check the strainer, raise the
suction level, and throttle the discharge to move the duty point back toward
best efficiency. Prolonged cavitation pits the impeller as in section 4.7.

### 5.9 Seal leakage

A failing mechanical seal shows first on the process channels rather than on
vibration: delivered flow falls by ten to twenty per cent against the regime
baseline while discharge pressure falls more slowly, and motor current falls
with the load. Visible drip at the gland confirms it. Causes are face wear at
the end of seal life, dry running during a loss of suction, solids in the
pumped fluid scoring the faces, and a shaft sleeve worn by a previous seal.
Mitigation is the seal replacement in section 4.6. A leak that reaches the
drive-end bearing housing contaminates the grease and becomes a cause of the
outer-race wear in section 5.4.

### 5.10 Looseness

Mechanical looseness produces a spectrum with many harmonics of shaft speed,
often with half-order components, and a directional response that is much
higher in one radial axis than the other. Distinguish it from misalignment by
checking that the alignment readings are within the tolerances in section 4.4
and by checking the hold-down torque in section 4.8.

### 5.11 Trending and baselines

Judge every reading against the baseline for its own regime. A level that is
normal at high load can be a fifty per cent rise at low load, and a detector
that compares against a single machine-wide baseline will either miss the rise
or alarm on the regime change. Re-baseline after any work that changes the
machine's mechanical state, including bearing replacement and alignment.

### 5.12 Alarm response

At a first warning, confirm the reading with a second measurement at the same
duty point before acting, then inspect within the shift. At a critical level,
plan a shutdown rather than continuing to run: the damage rate above zone C
rises sharply and a bearing that is caught in zone C is usually a bearing
change, while one caught in zone D often takes the shaft and the seal with it.

---

## 6. Operation

### 6.1 Start-up

Prime the casing, open the suction valve fully, crack the discharge valve, and
start against a near-closed discharge. Open the discharge to duty within thirty
seconds: running against a closed discharge recirculates the whole flow and
heats the casing. Expect a vibration transient during run-up and do not judge
the machine's condition from it.

### 6.2 Minimum flow

Do not run below 30 m³/h, which is twenty-five per cent of rated flow.
Operation below the minimum-flow line causes suction and discharge
recirculation, which produces a cavitation-like signature on the vibration
channels without any loss of NPSH, and damages the impeller in the same way.

### 6.3 NPSH

The pump requires 3.2 m NPSH at rated flow and 4.6 m at 120 per cent of rated
flow. Available NPSH falls as the suction tank level falls, as the strainer
blocks, and as the fluid temperature rises toward its vapour pressure. Any of
the three can put the machine into cavitation without a change in the pump
itself.

### 6.4 Shutdown

Close the discharge valve, stop the motor, then close the suction valve. Leave
the casing full unless the machine is being opened. Where the machine will be
idle for more than a month, turn the shaft by hand a quarter turn weekly to
avoid false brinelling of the stationary bearings.

---

## 7. Records

### 7.1 What to record

Record every intervention against the running-hour counter: regreasing,
alignment readings cold and hot, bearing changes with the bearings' markings,
seal changes, and the vibration and temperature baselines measured after the
work. A condition-monitoring system that is re-baselined without a record of
why cannot distinguish an improvement from a reset.
