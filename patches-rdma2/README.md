# rdna-boosts RDNA2 (gfx1030) FA gate fixes

**1 patch** that applies on top of the 12-block rdna-boosts delivery set
(`release.json`, currently `v12-097cf33d9-r82`, applied tree `dc48b4a9`;
r81 `0c3a4413` is also supported - the three FA files are byte-identical
between the r81 and r82 bases, so the same patch applies cleanly on both).  It fixes the three bugs that made
`test-backend-ops -o FLASH_ATTN_EXT` abort with an illegal memory access on
gfx1030 (Radeon Pro V620), which blocked the MUL_MAT gate behind it:

1. **bf16 f16 scratch sizing** (`fattn.cu`): `ggml_cuda_flash_attn_ext_get_alloc_size`
   did not reserve the f16 conversion scratch for bf16 KV on gfx1030
   (`V_DOT2_F32_BF16` is gfx11xx/12xx only), so the whole-cache converter wrote
   past the node allocation (sentinel clobber, then illegal access).
2. **guard tail for the unchecked tile load** (`fattn-common.cuh`): the
   `ncols2 > 1` tile branch loads the last KV tile without OOB checks; for a KV
   length not a multiple of `nbatch_fa` the read can overrun the exact-size
   converted buffer by up to `nbatch_fa - 1` rows.  A 511-row guard tail is
   reserved after each converted operand.
3. **decode-band `q_h2` launch arg** (`fattn-tile.cuh`): the
   `Q->ne[1] <= 8` branch of `launch_fattn_tile_switch_ncols1` omitted the
   `q_h2` argument (default false) while the other six launch sites pass it.
   The wide RDNA2 kernels (DKQ <= 256, ncols >= 16,
   `GGML_CUDA_FATTN_TILE_Q_GLOBAL`) read Q from that global half2 buffer, so
   gqa-16 decode shapes (e.g. hsk=192, nr23=[16,1], n_q <= 8) faulted near
   0x1000.  The same condition as the other sites is now passed.

Apply with `../scripts/apply-rdma2.sh` **after** `../scripts/apply-all.sh`;
the result is committed on branch `rdna-boosts-rdna2-fa` and asserted against
`../release-rdma2.json`.

Every change is compile- or runtime-gated to RDNA2 (`GGML_CUDA_CC_IS_RDNA2` /
`GGML_CUDA_FATTN_TILE_Q_GLOBAL`) or sits on the bf16->f16 staging path that
only gfx1030 takes for the tile kernel; gfx1100/gfx1151/gfx1201 and NVIDIA
builds of the same tree are unaffected.

## Validation (ROCm 7.2.4, `GPU_TARGETS=gfx1030;gfx1100` dual build)

| card | op | result |
|------|----|--------|
| Radeon Pro V620 (gfx1030) | FLASH_ATTN_EXT | 6389/6389 passed |
| Radeon Pro V620 (gfx1030) | MUL_MAT | 2444/2444 passed |
| Radeon RX 7900 XTX (gfx1100) | FLASH_ATTN_EXT | 6389/6389 passed |
| Radeon RX 7900 XTX (gfx1100) | MUL_MAT | 2444/2444 passed |

Repro before the fix (V620): `test-backend-ops -o FLASH_ATTN_EXT -p
'hsk=192,hsv=128.*nr23=[16,1].*kv=512,.*type_K=bf16'` -> illegal memory
access; the same run passes after it.
