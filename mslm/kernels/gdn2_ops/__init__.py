"""GDN-2 Triton kernels, vendored unchanged from NVIDIA's GatedLinearAttention2
(via long-gdn/dsc/lit_gpt/gdn2_ops). Licensed under the NVIDIA Source Code
License-NC (see LICENSE in this folder): non-commercial research use only.

They import fla internals that changed after fla commit 4b02d15d (e.g. the
``use_exp2`` argument), so fla MUST be installed at exactly that commit
(setup/install_env.sh does this).
"""
