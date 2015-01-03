#!/bin/bash
git checkout dev
git branch -D future
git checkout -b future
# deconv layer, coord maps, net pointer, crop layer
hub merge https://github.com/BVLC/caffe/pull/1639
# reshaping data layer
hub merge https://github.com/BVLC/caffe/pull/1313
# softmax missing values
hub merge https://github.com/BVLC/caffe/pull/1654
# python testing
hub merge https://github.com/BVLC/caffe/pull/1473
# gradient accumulation
hub merge https://github.com/BVLC/caffe/pull/1663
# solver stepping
hub merge --rerere-autoupdate https://github.com/BVLC/caffe/pull/1228
git add future.sh
git commit -m 'add creation script'

cat << 'EOF' > README.md
This is Caffe with several unmerged PRs and no guarantees.

Everything here is subject to change, including the history of this branch.

See `future.sh` for details.
EOF

git add README.md
git commit -m 'update readme'
