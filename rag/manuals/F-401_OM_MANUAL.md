# Centrifugal fan F-401 — Operation and Maintenance Manual

> **Synthetic document.** This manual was written for the Harmonic project. It
> is not a vendor document and carries no third-party text. The machine it
> describes is the reference machine of `sim/fan_sim.py` and the profile
> `profiles/fan_f401.profile`: a direct-driven centrifugal fan with an
> 8-blade wheel at 1180 rpm, carried on an LDK UER204 insert bearing in a
> pillow block. Section numbers are stable and are the citation tokens the
> GenAI layer is allowed to use.

Doc id: `F-401-OM` · Revision A · 27 sections

---

## 1. Machine description

### 1.1 Identification and duty

Fan F-401 is a direct-driven centrifugal fan with an 8-blade wheel, driven
by a 6-pole motor on a 60 Hz supply with no belt or gearbox between them.
Running speed is 1180 rpm, a shaft frequency of 19.7 Hz. The fan is assessed
for vibration under ISO 10816-3 group 2, flexible support, because it is
mounted on a spring-isolated base.

### 1.2 Wheel, shaft and bearing arrangement

The wheel carries 8 blades and turns once per shaft revolution, so its
blade-pass frequency is 8 times shaft speed, 157 Hz. The shaft is carried
in an LDK UER204 insert bearing housed in a pillow block, with 8 rolling
elements. Its outer-race defect frequency (BPFO) is 3.083 times shaft
speed, 60.6 Hz.

### 1.3 Instrumentation

One accelerometer is mounted radially on the bearing housing, reporting
acceleration in g. The duct instrumentation reports airflow, healthy at
about 4.0 m³/s, and a temperature sensor on the bearing housing reads a
healthy value of about 45 °C. These signals together carry the monitoring
evidence for this fan (see 5.7 and 5.8).

## 2. Safety

### 2.1 Lockout and tagout

Before any work inside the fan casing or on the bearing, stop the fan,
isolate the drive motor at the local isolator, lock it and tag it with the
name of the person doing the work, and confirm the wheel has stopped
turning. The monitoring system is advisory only and must never be relied
on to keep the fan stopped.

### 2.2 Rotating wheel and guarding

The coupling and shaft are guarded at the drive end; never reach past a
guard while the fan is running. Keep hands, tools and loose clothing clear
of the inlet and outlet openings, since the wheel continues to turn under
its own inertia for some time after the motor is switched off.

### 2.3 Entry into the fan housing

Before anyone enters the fan housing, isolate and lock out the drive (see
2.1), close and lock any dampers that could admit a flow of air, and
confirm the wheel is fully stopped by direct observation through the
inspection door, not from the control room reading alone.

## 3. Airflow and system

### 3.1 Airflow and pressure

Airflow is measured by a duct sensor downstream of the fan outlet and
trended against the 4.0 m³/s baseline at the normal system resistance. A
falling reading at unchanged fan speed points first to the fan itself (see
5.4) rather than to the sensor, once damper and filter position are ruled
out (see 3.2).

### 3.2 Dampers and filters

Confirm damper position and filter condition before treating a low-airflow
reading as a fan fault: a partly closed damper or a loaded filter raises
system resistance and lowers flow with the fan itself unchanged. Airflow
readings are only compared against the baseline with dampers and filters
in their normal running condition.

### 3.3 Bearing temperature

The bearing housing temperature sensor is read continuously and trended
against the 45 °C baseline. A rise with steady vibration points to a
lubrication or cooling problem; a rise together with rising vibration and
BPFO impacts points to a developing bearing fault (see 5.6).

## 4. Mechanical maintenance

### 4.1 Wheel cleaning

Dirt and process residue build up unevenly on the wheel blades and shift
the wheel's mass distribution, producing unbalance (see 5.3). Wheel
cleaning is done with the fan isolated (see 2.1), by hand or with a
low-pressure wash as the process allows, working blade by blade rather than
spraying across the whole wheel at once.

### 4.2 Wheel inspection and blade repair

Inspect every blade for a cracked blade, a bent or eroded leading edge, and
a fouled blade carrying residue that cleaning has not removed. A cracked
blade is not repaired in place; replace the wheel. Minor erosion is
recorded and trended, and a fouled blade is scheduled for wheel cleaning
(see 4.1) rather than repair.

### 4.3 Balancing

Where unbalance is confirmed by vibration evidence (see 5.3) and wheel
cleaning has not resolved it, trim-balance the wheel in place with weights
fitted to the balancing rim. Balance to the grade specified for this fan
class. Unbalance that returns quickly after balancing points to ongoing
fouling or erosion rather than a one-off imbalance.

### 4.4 Bearing lubrication and replacement

Regrease the pillow block bearing on the interval set for its duty and
ambient temperature, using the grease grade specified for the LDK UER204.
To replace the bearing, isolate the fan (see 2.1), support the shaft,
withdraw the worn insert from its housing, fit the replacement to the
shaft, and lock its collar before removing the shaft support.

### 4.5 Mounting and base bolts

Check all base and pillow-block hold-down bolts for tightness at scheduled
inspection; a loose base or a hold-down bolt backed off produces mechanical
looseness in the vibration signature (see 5.5). Retorque to the specified
value and inspect the isolation springs for damage or uneven compression.

### 4.6 Inlet cone clearance

Check the running clearance between the wheel inlet and the inlet cone
against the drawing value at scheduled inspection. A clearance that has
closed up indicates shaft or bearing wear and is investigated with the
vibration evidence before the clearance is reset.

## 5. Condition monitoring

### 5.1 Vibration severity zones

Overall vibration is assessed as velocity RMS over 10–1000 Hz against
ISO 10816-3 group 2, flexible support: zone A below 2.3 mm/s, zone B to
4.5 mm/s, zone C to 7.1 mm/s, zone D above. The bearing-housing
accelerometer signal is integrated to velocity for this assessment.

### 5.2 Blade-pass frequency

Blade-pass frequency is the blade count times shaft speed: 8 times 19.7 Hz,
157 Hz at nominal speed. Its amplitude reflects how evenly the wheel
delivers flow to the volute and is the reference feature against which
sidebands from blade damage are measured (see 5.4).

### 5.3 Unbalance

Unbalance from dirt build-up or erosion on the wheel raises the vibration
at shaft speed, 1X, with little immediate change in airflow or bearing
temperature. It is the most common fault on this fan class. Confirm with
wheel cleaning (see 4.1) before resorting to balancing (see 4.3).

### 5.4 Blade damage and fouling

Blade damage — a cracked blade, a bent or eroded blade, or a fouled blade —
produces sidebands at blade-pass frequency plus and minus shaft speed, a
modest rise in the blade-pass amplitude itself, and a reduction in airflow.
Confirm with a wheel inspection (see 4.2) before the fan is returned to
service.

### 5.5 Mechanical looseness

Mechanical looseness, from a loose base, a backed-off hold-down bolt or a
deteriorated foundation, raises vibration at twice and three times shaft
speed and introduces a half-order subharmonic component not present in a
healthy spectrum. Correct by retorquing the mounting and base bolts (see
4.5) and inspecting the isolation springs.

### 5.6 Bearing outer-race wear

Outer-race wear of the pillow-block bearing produces impacts at BPFO,
60.6 Hz at nominal speed, visible in the envelope spectrum together with a
rise in kurtosis, and the bearing runs hot. Advanced outer-race spalling
raises the plain-spectrum level as well. Plan the bearing replacement (see
4.4) once envelope impacts are confirmed.

### 5.7 Airflow trend

Airflow is trended against the 4.0 m³/s baseline once damper and filter
position are confirmed normal (see 3.2). A falling trend with rising
blade-pass sidebands supports blade damage (see 5.4); a falling trend with
no vibration evidence points back to the duct system rather than the fan.

### 5.8 Bearing temperature trend

Bearing temperature is trended against the 45 °C baseline. A rise with
steady vibration points to lubrication; a rise together with rising BPFO
impacts and rising kurtosis supports bearing outer-race wear (see 5.6)
rather than a lubrication problem alone.

### 5.9 Alarm response

On any WATCH or WARNING, confirm the reading first: check that the
accelerometer is secure, that the fan is at steady load, and that the
supply is normal. Then follow the section for the fault the evidence points
to. A CRITICAL state is an advisory to plan a stop under lockout (see 2.1);
the plant protection system remains in charge of stopping the fan.

## 6. Operation

### 6.1 Start-up

Confirm dampers are in their start position and the wheel turns freely by
hand before starting after an extended shutdown (see 2.1 for isolation
during the check). Bring the fan up to speed and allow airflow to settle
before comparing the reading to baseline; readings during the first minute
are excluded from alarms.

### 6.2 Shutdown

Open dampers to their running position before stopping where the process
allows, to avoid a sudden pressure change across the wheel. After a stop of
more than a week, turn the wheel by hand before restarting to confirm it
turns freely and redistribute bearing grease across the insert.

## 7. Records

### 7.1 What to record

Record every wheel cleaning, blade inspection, balancing job, bearing
replacement and mounting check against the fan and the date, together with
the airflow and vibration baseline taken after each intervention. The
baseline after maintenance is what later alarms are judged against.
