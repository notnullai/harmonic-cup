# Induction motor M-301 — Operation and Maintenance Manual

> **Synthetic document.** This manual was written for the Harmonic project. It
> is not a vendor document and carries no third-party text. The machine it
> describes is the reference machine of `sim/motor_sim.py` and the profile
> `profiles/motor_m301.profile`: a 4-pole three-phase induction motor, 2.2 kW,
> on a 60 Hz supply, with an SKF 6205-2RS JEM deep groove ball bearing at the
> drive end. Section numbers are stable and are the citation tokens the GenAI
> layer is allowed to use.

Doc id: `M-301-OM` · Revision A · 27 sections

---

## 1. Machine description

### 1.1 Identification and duty

Motor M-301 is a 4-pole three-phase induction motor rated 2.2 kW on a 60 Hz
supply. Synchronous speed is 1800 rpm; at full load the rotor runs at
1710 rpm, a slip of 5%, giving a shaft frequency of 28.5 Hz. The motor drives
its load through a direct coupling and is assessed for vibration under
ISO 10816-1 class I.

### 1.2 Rotor, stator and bearing arrangement

The squirrel-cage rotor turns in an SKF 6205-2RS JEM deep groove ball
bearing at the drive end, with nine rolling elements. At 1710 rpm its
outer-race defect frequency (BPFO) is 3.585 times shaft speed, 102.2 Hz, and
its cage frequency is 0.398 times shaft speed. The non-drive end carries a
matching sealed bearing that is not separately monitored.

### 1.3 Instrumentation

One accelerometer is mounted radially at the drive-end bearing housing,
reporting acceleration in g. A resistance temperature detector embedded in
the stator winding reads a healthy value of about 72 °C at rated load, and
the supply feeder reports phase current, healthy at about 8.2 A. These three
signals together carry the monitoring evidence for this motor (see 5.8).

## 2. Safety

### 2.1 Lockout and electrical isolation

Before any work on the motor or its terminal box, open and lock the local
isolator, tag it with the name of the person doing the work, and prove the
terminals dead with a voltage tester before touching them. The monitoring
system is advisory only: it never removes power from the motor and must
never be relied on to keep it isolated.

### 2.2 Rotating parts and hot surfaces

The coupling and shaft extension are guarded; never reach past a guard while
the motor is running. The frame and the drive-end bearing housing can reach
80 °C in normal service, hotter under a developing fault, so allow the motor
to cool before touching it and rely on the temperature reading rather than
touch for a check.

### 2.3 Lifting

The motor weighs 32 kg. Lift by the frame eyebolt only, never by the
terminal box or the shaft, and support the load with a sling rated for the
full weight before disconnecting the mounting bolts.

## 3. Electrical

### 3.1 Supply and protection

The motor is fed through a thermal-magnetic breaker and a contactor sized
for its full-load current. Confirm the supply voltage is balanced across the
three phases within 2% before starting; a sustained unbalance loads one
phase winding more than the others and behaves, electrically and thermally,
like a stator winding fault (see 5.5).

### 3.2 Insulation resistance test

With the motor isolated (see 2.1) and the terminals disconnected from the
supply, measure insulation resistance from each phase winding to frame with
a 500 V insulation tester. A reading below 1 MΩ, or one that has fallen
sharply since the last test, indicates moisture or winding degradation and
the motor should not be re-energised until the cause is found.

### 3.3 Winding temperature

The embedded RTD is read continuously and trended against the 72 °C
baseline at rated load. A rise with steady current and steady vibration
points to a cooling-path problem such as a blocked fan or fouled fins; a
rise together with a rising current points to an electrical cause (see 5.5
and 5.8).

## 4. Mechanical maintenance

### 4.1 Bearing lubrication

The drive-end bearing is sealed (2RS) and is not designed to be regreased in
service; do not attempt to add grease through the shield. Check instead for
grease discolouration and bearing temperature at scheduled inspections, and
plan replacement (see 4.2) once the sealed charge is judged exhausted rather
than topping it up.

### 4.2 Drive-end bearing replacement

With the motor isolated and the rotor withdrawn, pull the bearing from the
shaft with a puller acting on the inner ring, never through the balls. Heat
the replacement SKF 6205-2RS JEM by induction to about 100 °C before
fitting, seat it against the shaft shoulder, and record the fitting date
against the drive-end position.

### 4.3 Rotor inspection and bar testing

Where broken rotor bar evidence is present (see 5.4), an offline growler or
current-injection bar test locates the affected bars directly. Inspect
visible end-ring joints for cracking or discolouration from overheating. A
rotor with more than one confirmed broken bar is not economic to repair and
is normally replaced as an assembly.

### 4.4 Balancing

Residual rotor unbalance, or unbalance introduced by a coupling half or a
fan, is corrected by trim-balancing at the coupling end with the motor
running free. Balance to the grade specified on the nameplate. Repeated
unbalance after balancing points to a bent shaft or a loose rotor core and
is not solved by adding further trim weight.

### 4.5 Alignment and soft foot

Check for soft foot at each of the four feet, one bolt loosened at a time,
before aligning the motor to its load; a foot that moves more than 0.05 mm
must be shimmed flat before alignment proceeds. Align with a laser system to
within 0.05 mm offset and 0.05 mm per 100 mm angularity.

### 4.6 Stator winding repair

A confirmed stator winding fault, such as shorted turns found by the
insulation resistance test (see 3.2) or a growing surge-test difference
between phases, is not field-repairable on this frame size. Isolate the
motor, tag it out of service, and rewind or replace it; running on
risks a phase-to-phase or phase-to-earth failure.

## 5. Condition monitoring

### 5.1 Vibration severity zones

Overall vibration is assessed as velocity RMS over 10–1000 Hz against
ISO 10816-1 class I: zone A below 0.71 mm/s, zone B to 1.8 mm/s, zone C to
4.5 mm/s, zone D above. The drive-end accelerometer signal is integrated to
velocity for this assessment.

### 5.2 Running speed and slip

Shaft frequency at full load is 28.5 Hz, against a synchronous 30 Hz, a
slip of 5%. Slip rises with load and falls as the motor lightens, so the
running-speed peak in the spectrum is expected to drift slightly with duty
and is not itself a fault indication.

### 5.3 Twice line frequency

A component at twice the line frequency, 120 Hz, order 4.21 relative to
shaft speed, is normally low and steady. A rise in this component, taken
together with rising winding temperature and rising current, is the primary
vibration signature of an electrical fault in the stator (see 5.5) rather
than a mechanical one.

### 5.4 Broken rotor bars

A broken rotor bar, or broken rotor bars, unbalance the rotor's magnetic
field once per slip cycle, producing sidebands at shaft frequency plus and
minus the pole-pass frequency, four times the slip frequency, 6 Hz at full
load. The same fault shows as a slow swing in phase current at the 6 Hz
pole-pass rate. Confirm with a bar test (see 4.3).

### 5.5 Stator winding faults

Shorted turns or a developing stator winding fault raise the 120 Hz
vibration component, raise the winding temperature reading, and raise
phase current, together rather than singly. Supply voltage unbalance
produces the same three-part signature and should be ruled out first (see
3.1) before opening the motor for a winding fault.

### 5.6 Unbalance

Mechanical unbalance of the rotor or a coupled component raises the
vibration at shaft speed, 1X, with little change in current or temperature.
It is the most common mechanical fault on this motor class and is corrected
by trim-balancing (see 4.4) once other 1X causes, such as a bent shaft, are
ruled out.

### 5.7 Drive-end bearing outer-race wear

Outer-race wear of the drive-end bearing produces impacts at BPFO, 102.2 Hz
at full-load speed, visible in the envelope spectrum together with a rise
in kurtosis. Advanced outer-race spalling raises the plain-spectrum
overall level as well. Plan the bearing replacement (see 4.2) once envelope
impacts are confirmed and trending upward.

### 5.8 Current and temperature trends

Phase current and winding temperature are trended against their 8.2 A and
72 °C baselines at rated load. Both rising together points to an electrical
cause (see 5.5); temperature rising alone points to cooling; current alone
swinging at the pole-pass rate points to broken rotor bars (see 5.4).

### 5.9 Alarm response

On any WATCH or WARNING, confirm the reading first: check that the
accelerometer is secure, that the load is steady, and that the supply is
normal. Then follow the section for the fault the evidence points to. A
CRITICAL state is an advisory to plan a stop under lockout (see 2.1); the
plant protection system remains in charge of stopping the motor.

## 6. Operation

### 6.1 Starting

Confirm the supply is balanced (see 3.1) and the insulation resistance test
is current (see 3.2) before starting after an extended shutdown. Bring the
motor up to speed unloaded where the process allows, and exclude readings
taken during the first minute of running from alarm evaluation while
starting current settles.

### 6.2 Stopping

Remove the driven load before stopping where the process allows. After a
stop of more than a week, bar the rotor over by hand before restarting to
confirm it turns freely and to redistribute bearing grease across the
raceway.

## 7. Records

### 7.1 What to record

Record every insulation resistance test, bearing replacement, balancing
job, alignment check and winding repair against the motor and the date,
together with the vibration and current baseline taken after each
intervention. The baseline after maintenance is what later alarms are
judged against.
