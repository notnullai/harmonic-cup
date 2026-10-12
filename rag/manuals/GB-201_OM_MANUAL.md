# Gearbox GB-201 — Operation and Maintenance Manual

> **Synthetic document.** This manual was written for the Harmonic project. It
> is not a vendor document and carries no third-party text. The machine it
> describes is the reference machine of `sim/gearbox_sim.py` and the profile
> `profiles/example_gearbox.profile`: a single-stage parallel-shaft gearbox,
> 1480 rpm input, 37-tooth pinion, a 6207 ball bearing on the input shaft.
> Section numbers are stable and are the citation tokens the GenAI layer is
> allowed to use.

Doc id: `GB-201-OM` · Revision A · 27 sections

---

## 1. Machine description

### 1.1 Identification and duty

Gearbox GB-201 is a single-stage parallel-shaft reducer rated at 30 kW. A
37-tooth helical pinion on the input shaft drives a 74-tooth wheel, a 2:1
reduction. The input runs at a nominal 1480 rpm, a running frequency of
24.7 Hz, so the output turns at 740 rpm. The gear mesh frequency — teeth times
input shaft frequency — is 912.7 Hz. The gearbox is assessed under ISO 10816-3
group 2 on a rigid foundation.

### 1.2 Gear and bearing arrangement

The input shaft runs in a 6207 deep-groove ball bearing at the drive end, with
nine rolling elements of 11.1 mm on a 46.0 mm pitch circle and a contact angle
of zero. At 1480 rpm its outer-race defect frequency (BPFO) is 84.2 Hz and its
cage frequency is 9.4 Hz, 0.379 times shaft speed. These figures set the
monitoring configuration and must be re-entered if a different bearing is
fitted, as must the tooth count if the gear set is changed (see 5.2).

### 1.3 Instrumentation

One accelerometer is mounted radially on the input bearing housing, reporting
acceleration in g, and a resistance thermometer sits in the oil sump. The
monitoring system derives from the accelerometer the overall level, the shaft
harmonics, the gear mesh frequency, its second harmonic and its sidebands, the
bearing defect frequency and the cage frequency in the envelope spectrum.

## 2. Safety

### 2.1 Lockout and tagout

Before any work inside the guard or on the gearbox, stop the drive, isolate
the motor at the local isolator, lock it and tag it with the name of the person
doing the work, and prove zero energy by attempting a start. The monitoring
system is advisory only: it never stops the gearbox and must never be relied on
to keep it stopped.

### 2.2 Hot oil and rotating parts

Sump oil can exceed 70 °C in service. Allow the gearbox to cool before opening
inspection covers or draining oil, and wear gloves and eye protection. Keep
hands, tools and loose clothing clear of the couplings while the drive is
running; inspections of rotating parts are done by stroboscope, not by touch.

### 2.3 Lifting

The gearbox weighs 310 kg with oil. Lift only by the two cast lifting eyes on
the upper casing, never by the shafts, and drain the oil first when the
gearbox is to be removed from its base.

## 3. Lubrication

### 3.1 Oil and oil changes

The gearbox is splash-lubricated with an ISO VG 220 mineral gear oil to the
level mark on the sight glass. Change the oil every 4000 running hours or
yearly, whichever comes first, and after the first 500 hours of a new gear
set. Never mix oils of different grades or base stocks.

### 3.2 Oil analysis

Take an oil sample from the drain valve with the gearbox warm, every 1000
hours and whenever an alarm is raised. Rising iron particle counts indicate
wear of the gear teeth or the bearing; a sudden rise in large particles after a
period of steady readings indicates a tooth or bearing surface breaking up.

### 3.3 Oil temperature

The sump runs at about 55 °C at nominal load. A sustained rise of 10 °C above
the regime baseline indicates increased friction — wear of the gear teeth, a
failing bearing, misalignment or low oil level — and is investigated with the
vibration evidence. Do not run the gearbox above 85 °C sump temperature.

## 4. Mechanical maintenance

### 4.1 Gear mesh inspection

With the drive isolated (see 2.1), remove the inspection cover and turn the
input by hand. Check every pinion tooth for pitting, scuffing, spalling and
cracks at the root. Apply marking compound to three wheel teeth and turn the
set through mesh to record the contact pattern; a pattern concentrated at one
end of the face indicates misalignment (see 4.4).

### 4.2 Gear set replacement

Replace the pinion and the wheel together; a new pinion run against a worn
wheel wears out quickly. Drain the oil, remove the upper casing, withdraw both
shafts, and press the gears off with a hydraulic puller. Fit new gears at the
interference stated on the drawing, reassemble, set the backlash (see 4.6), and
refill with fresh oil (see 3.1).

### 4.3 Input bearing replacement

With the oil drained, remove the input bearing cover and withdraw the shaft.
Pull the bearing with a puller acting on the inner ring only, heat the new
bearing by induction to 110 °C, and slide it on against the shoulder. Never
drive a bearing on with a hammer. Record the fitting date against the bearing
position.

### 4.4 Shaft alignment

Misalignment between the motor and the input shaft loads the input bearing and
the gear teeth unevenly. Align with a laser system to within 0.05 mm offset and
0.05 mm per 100 mm angularity, with the gearbox at operating temperature or with
the thermal growth allowed for. Check for soft foot under every hold-down
before aligning.

### 4.5 Coupling inspection

Inspect the elastomer coupling element for cracks and hardening at every oil
change. A hardened element transmits misalignment forces into the input shaft;
replace it whenever the alignment is corrected.

### 4.6 Backlash check

Measure backlash with a dial indicator on a wheel tooth while holding the
pinion. The specified range is 0.15–0.25 mm. Backlash above 0.35 mm indicates
tooth wear and is a reason to plan gear set replacement (see 4.2).

## 5. Condition monitoring

### 5.1 Vibration severity zones

Overall vibration is assessed as velocity RMS over 10–1000 Hz against ISO
10816-3 group 2, rigid foundation: zone A below 1.4 mm/s, zone B to 2.8 mm/s,
zone C to 4.5 mm/s, zone D above. The accelerometer signal is integrated to
velocity for this. The gear mesh at 912.7 Hz lies near the top of the band, so
gear faults show earlier in the mesh features than in the overall level.

### 5.2 Gear mesh frequency

The gear mesh frequency is the tooth count times the shaft frequency: 37 times
24.7 Hz, 912.7 Hz at nominal speed. Its amplitude and that of its second
harmonic at 1825 Hz reflect how smoothly the teeth transfer load. Both rise as
the tooth profiles wear, and the second harmonic rises faster, because wear
distorts the profile from the involute shape.

### 5.3 Sidebands

A gear that is not uniform around its circumference modulates the mesh once per
revolution, and the modulation appears as sidebands at the mesh frequency plus
and minus the shaft frequency. Low sidebands are normal. Sidebands that grow
quickly, or come to match the mesh amplitude, indicate damage localised on one
tooth or eccentricity of the gear.

### 5.4 Tooth wear

Distributed wear of the tooth flanks raises the gear mesh amplitude and, more
strongly, its second harmonic, with a modest rise in sidebands and a rise in
oil temperature. Confirm with oil analysis (see 3.2) and backlash (see 4.6).
Wear that is still in its early stage is managed by oil change and monitoring;
advanced wear calls for gear set replacement (see 4.2).

### 5.5 Broken or cracked tooth

A cracked or broken tooth strikes once per revolution of its shaft as it enters
mesh. The signal becomes impulsive: kurtosis, crest factor and peak rise, and
the mesh sidebands grow strongly while the mesh amplitude itself may change
little. Stop the gearbox at the next opportunity and inspect the teeth (see
4.1); running on risks the fragment damaging the other gear and the bearings.

### 5.6 Misalignment

Misalignment of the input shaft raises the amplitude at shaft speed and
especially at twice shaft speed, with a moderate rise in oil temperature. The
gear contact pattern moves to one end of the face (see 4.1). Correct with a
soft-foot check, coupling inspection and laser alignment (see 4.4 and 4.5).

### 5.7 Input bearing outer-race wear

Outer-race spalling of the input bearing produces impacts at BPFO, 84.2 Hz at
nominal speed, visible in the envelope spectrum rather than the plain spectrum,
and often modulated at the cage frequency. Kurtosis rises and the oil warms.
Confirm by listening and by oil analysis, then plan the bearing replacement
(see 4.3); a spalled race progresses faster once the oil carries its debris.

### 5.8 Oil temperature trend

Oil temperature is trended against the baseline of the same duty regime. A
rise with no vibration evidence points first to the oil level and the oil
itself (see 3.1); a rise together with vibration evidence supports the fault
that evidence names (see 5.4 to 5.7).

### 5.9 Alarm response

On any WATCH or WARNING, confirm the reading before acting: check that the
sensor is secure, that the regime is steady, and that the oil level is
correct. Then follow the section for the fault the evidence points to. A
CRITICAL state is an advisory to plan a stop under lockout (see 2.1); the plant
protection system remains in charge of stopping the drive.

## 6. Operation

### 6.1 Start-up

Check the oil level on the sight glass before every start. Start unloaded
where the process allows and bring the load up over five minutes; a cold oil
film carries load poorly. Readings during the first minutes after a start are
excluded from alarms.

### 6.2 Shutdown

Remove the load before stopping the drive. After a long stop, turn the input
by hand before restarting to redistribute the oil over the teeth.

## 7. Records

### 7.1 What to record

Record every oil change, oil sample result, backlash reading, alignment and
component replacement against the gearbox and the date, with the vibration
baseline taken after each intervention. The baseline after maintenance is what
later alarms are judged against.
