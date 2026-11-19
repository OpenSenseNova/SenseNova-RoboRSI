# SIMPLE assets for the six evaluation tasks

Assets stay outside the image and source PR. Obtain them from the original public
`USC-PSI-Lab/SIMPLE` dataset at revision
`f5b58ab75210367c71ff7347155a656d403169a8`. Respect its and the constituent
assets' licenses. The seven archive hashes are recorded in `assets.sha256`.
Evaluation resets are a separate dataset pinned in `robots/simple/sources.json`.

From the RoboRSI checkout (the HF client belongs in a preparation environment):

```bash
hf download USC-PSI-Lab/SIMPLE --repo-type dataset \
  --revision f5b58ab75210367c71ff7347155a656d403169a8 \
  assets_graspnet.zip robots_g1.zip robots_g1_sonic.zip \
  scenes_hssd_102344022.zip scenes_hssd_107733960_175999701.zip \
  scenes_hssd_107734119_175999932.zip vMaterials_2.zip \
  --local-dir /data/simple-archive-cache
cp benchmarks/simple/assets.sha256 /data/simple-archive-cache/
(cd /data/simple-archive-cache && sha256sum -c assets.sha256)
mkdir -p /data/simple-registry/simple/assets
for archive in /data/simple-archive-cache/*.zip; do
  python -m zipfile -e "$archive" /data/simple-registry/simple/assets
done
```

Use a cache dedicated to these seven archives. The resulting asset root must
contain `assets/graspnet/`, `robots/g1/`, `robots/g1_sonic/`, `scenes/hssd/`, and
`vMaterials_2/`. Do not pass the ZIP directory as the simulator asset root.
Allow roughly 15 GB for archive downloads and extracted assets, in addition to
the image and per-service shader caches. Retain archives until extraction is
verified; they can then be moved to external storage.

The three scene IDs correspond to the released resets used by this adapter.
The renderer, robot geometry and object collision assets all come from these
public sources. No ground-truth scene information is forwarded to the Agent.
The Registry generator hashes the actual materialized tree, so any local asset
change produces a new asset identity; archive hashes alone do not certify a
particular extraction or historical experiment's asset set.
