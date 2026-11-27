# MPC status and claim boundaries

## Confirmed simulation evidence

At the pinned controller commit plus the simulation-hardening patch, all 12 upstream/hardening unit tests pass. A fresh eight-second headless validation on the Apple M4 completed 479 of 479 QPs with exact `solved` status, no torque clipping, 0.852 m forward displacement, 0.288 m/s mean forward speed while commanded, and 0.0129 m/s stopped planar speed. Height stayed between 0.251 and 0.277 m; maximum absolute roll and pitch were 0.0112 and 0.0365 rad. The recorded run had 2.76 ms MPC p95 against a 20.8 ms budget and 4.25 ms control-tick p95 against a 5 ms target. There were 29 individual Python deadline overruns among 1,600 ticks and a 25.5 ms maximum tick.

An independent clean clone with the exported patch also passed the 12 tests and validator. The final three-repeat validation completed 479 of 479 solved QPs with zero clipping in every repeat. MPC p95 varied from 2.275 to 2.396 ms, control-tick p95 varied from 3.809 to 3.869 ms, and individual deadline-overrun counts were 2, 1, and 4. The worst maximum control tick was 10.657 ms. The exact machine-readable results are generated under `runs/controller_validation/summary.json` by the documented command.

The implemented 12-episode matrix completed all 12 bearing-specific walks and all 240 intended frames with no fallback. The retained reference and fresh reproduction have byte-identical RGB and segmentation files, frame state after excluding only the run ID, and ground truth across all 240 frames. Detector outputs and evaluation also match after excluding runtime-profile fields. The fresh reproduction reports 2,916 of 2,916 QPs solved, zero clipping, 2.698 ms matrix-wide MPC p95 against 20.8 ms, and 4.361 ms control-tick p95 against 5.0 ms, so all declared simulation gates passed. The earlier retained run measured 3.390 ms MPC p95 and 5.381 ms control p95, narrowly failing the host-timing gate. No gains, gait, friction, force weights, prompts, or thresholds were changed between these results.

## Confirmed defects or limitations

The original controller checkout is dirty and cannot by itself serve as a reproducible artifact. The unpatched upstream model has no logged body RGB camera, scene schema, detector dependencies, frame/pose/ground-truth format, or experiment runner. The originally proposed camera x position of 0.30 m is inside the upstream front collision sphere; rendered validation showed the robot geometry obscuring the view, so the integration-owned camera uses 0.36 m. The offline 200 Hz Python loop has occasional individual deadline overruns, and its aggregate wall-clock timing changes with host load even when simulator state and rendered data are identical. The fresh isolated run passes the declared p95 gate, so there is no current reproduced simulation acceptance blocker; host-timing sensitivity remains a limitation.

## Plausible physical-hardware failure modes

Direct Jacobian-transpose realization of optimized contact forces may not transfer safely to the physical actuators. Position- and velocity-dependent actuator limits, temperature/current limits, state-estimator error, transport delay, time synchronization, stale commands, and watchdog behavior could all invalidate the simulation behavior. Zero torque can allow a quadruped to fall and is not assumed to be a safe stop.

## Speculation

It is plausible that added rendering load, different bearings, or scene changes could expose controller timing or initialization failures, but those are experiment hypotheses until a saved episode fails. It is also plausible that MPS inference differs numerically or operationally from CPU, Jetson, or desktop NVIDIA inference; no cross-platform equivalence is assumed.

## Not tested here

No physical Go2, Unitree SDK command, ROS 2 actuator path, onboard state estimator, hardware watchdog, emergency stop, stairs, obstacle contact, SLAM, learned locomotion, learned rejection module, or learned active-view policy is tested. MuJoCo success is not evidence of hardware safety.
