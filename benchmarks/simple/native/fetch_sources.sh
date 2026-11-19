#!/usr/bin/env bash
set -euo pipefail
# All sources are public HTTPS clones at explicit revisions. No SSH or token mount.
export GIT_TERMINAL_PROMPT=0 GIT_LFS_SKIP_SMUDGE=1
clone_revision() {
    local url=$1 revision=$2 destination=$3
    git init "$destination"
    git -C "$destination" remote add origin "$url"
    git -C "$destination" fetch --depth 1 origin "$revision"
    git -C "$destination" checkout --detach FETCH_HEAD
    test "$(git -C "$destination" rev-parse HEAD)" = "$revision"
    git -C "$destination" lfs install --local
}
clone_revision https://github.com/physical-superintelligence-lab/SIMPLE.git \
    6d10628794d9c7de4596b4f2afb2c054a637c2bc /opt/simple
while read -r name url revision; do
    clone_revision "$url" "$revision" "/opt/simple/third_party/$name"
    expected=$(git -C /opt/simple ls-tree HEAD "third_party/$name" | awk '{print $3}')
    test "$expected" = "$revision"
done <<'EOF'
curobo https://github.com/songlin/curobo.git 5810bc9cb69129a0e1a8aad47704bd6da35370c0
openpi-client https://github.com/songlin/openpi-client.git 4b8a1dd6fd93a9e90d91658361d4d7f98098a44a
gear_sonic https://github.com/songlin/gear_sonic.git e61eeb1dec9678dcc02f97c5ac27ca6a4fe82128
unitree_sdk2_python https://github.com/songlin/unitree_sdk2_python.git cf26ccef0e615c9509140322bec154adc391c052
decoupled_wbc https://github.com/songlin/decoupled_wbc.git 5b304e4d643fe89cd000068dcbba7e7519539caa
GR00T-WholeBodyControl https://github.com/songlin/GR00T-WholeBodyControl.git 948022a6fd19bd4ad10186d587b632482453b9a2
EOF
unset GIT_LFS_SKIP_SMUDGE
git -C /opt/simple lfs pull --include='src/simple/robots/policy/*.pt'
git -C /opt/simple/third_party/GR00T-WholeBodyControl lfs pull \
    --include='gear_sonic/data/robot_model/model_data/g1/**'
git -C /opt/simple/third_party/decoupled_wbc lfs pull
clone_revision https://github.com/eclipse-cyclonedds/cyclonedds.git \
    2cdd114cbd18340c606573b4cc8dc20cc161ec5a /opt/cyclonedds-src
