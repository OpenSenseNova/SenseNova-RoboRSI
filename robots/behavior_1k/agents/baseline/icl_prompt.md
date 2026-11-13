Perform offline ICL preparation for the specified BEHAVIOR task using the 200
public demonstrations in training/. No simulator or evaluation episodes are
available. Web references are allowed. Preserve provenance and distinguish
observations from hypotheses.

Write learned/TASK.md and learned/SOURCES.md, plus useful reusable code. TASK.md
must name the task and describe a grounded manipulation strategy. SOURCES.md must
identify demonstrated episodes and any external references. Use the converted
21D actions and their FK provenance; RGB videos and annotations are available.
Do not copy demonstration shards into learned/. Finish only after saving both
nonempty files. These learned files are the only handoff to a fresh evaluation
session, which will not have access to the demonstration set or your transcript.

Use the supplied tools to inspect individual demonstrations:
`icl_demonstration` returns an episode's paths and valid video intervals.
`python tools/records.py --episode practice-1 --head 5`
`python tools/frames.py --episode practice-1 --camera head --modality rgb --timestamps 0 1 --output-dir frames`
Read training/index.json for episode IDs and valid time intervals. Extract only
RGB frames for the selected episode. Inspect the resulting PNGs with the CLI's
image viewer. The frame helper verifies the source video hash before decoding.
