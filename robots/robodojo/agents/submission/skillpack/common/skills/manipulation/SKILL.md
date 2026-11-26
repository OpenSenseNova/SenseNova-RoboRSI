---
name: manipulation
description: Reusable observation-conditioned grasp, placement, support, contact and recovery principles.
---

# Observation-conditioned manipulation

Use current object geometry, measured poses and both-arm gripper intentions.
Choose subgoals and required endings from the current official Detail
Description. No task identity selects a workflow in this skill.

**Grasp and retention.** Identify accessible opposing grasp surfaces and a free
withdrawal direction. Choose a reachable approach without crossing occupied
space. Select orientation, opening and object-to-TCP offset from current RGB.
Inspect whether the intended surface is between the jaws before closing at the
measured pose. A closed command or shortened opening alone is insufficient.
When permitted, a small clearance lift can test whether object and gripper move
together. If the object stays, slides or tips, reobserve its new pose and change
the alignment or grasp hypothesis before another attempt. Transport requires
observed retention and clearance.

**Support and release.** Distinguish the held-object geometry from the TCP.
Align the intended support surfaces and account for the grasp offset and
required orientation. Inspect contact before release. An open command does not
prove release: confirm that the object separates from the jaws, remains
supported and is not dragged during withdrawal. When needed, inspect stability
after real native steps; repeated observe alone advances no physics. Whether
the final object must stay held, be released or return elsewhere is determined
by the official description, not this skill.

**Two-arm support.** Select a recipient grasp on an exposed surface rather than
the donor jaws or functional tip. Keep at least one verified support until
recipient retention is established. If support is uncertain, stabilize with
the supporting arm or use a visible stable staging surface when permitted.
Never open both grippers to resolve uncertainty. Balanced support may be useful
for a long or flexible object; inspect both arms' clearance and load effects.

**Contact, insertion and rotation.** Use visible mating geometry to choose
alignment and approach direction. Near uncertain contact, use small
feedback-controlled corrections. Unexpected lateral motion, tilt or residual
error can indicate an edge collision: withdraw or realign from observation
instead of forcing an unseen path. A rotation about the TCP can move the held
object's functional end along an arc because of grasp offset. Verify relative
motion and engagement; rotating an entire assembly does not establish that two
parts moved relative to one another. Regrasp only with adequate support.

**Articulation and deformable material.** Distinguish motion about a visible
hinge from translation of the whole body; unexpected sliding calls for
reobservation and a revised support hypothesis. For deformable material,
visible local displacement does not guarantee that the rest followed. Inspect
the resulting shape, support and exposed edges after an action; rigid-body pose
alone cannot establish its configuration.

**Flow, containment and occlusion.** Keep a receiver accessible and supported;
passing above an opening does not prove entry. Check visible transfer and any
external spill, and allow dynamic material to settle when needed. Hidden
contents require entry history rather than an empty source alone. If later
motion occludes a relevant region, retain episode-local visual evidence while
continuing to check what remains visible. Choose transfer order and final
container state only from the official description and current observations.

**Recovery.** After slip, collision, occlusion, uncertain execution or loss of
support, inspect RGB, measured state and native ACK before choosing the next
action. Replan from the actual configuration. Do not repeat a motion solely
because a parser failed, or infer grasp/task success from reached pose alone.
