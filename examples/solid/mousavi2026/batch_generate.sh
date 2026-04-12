#!/bin/bash
# Batch generation script for all 18 datasets.
#
# Usage:
#   bash batch_generate.sh                # Default: 256 samples per dataset
#   bash batch_generate.sh 8448           # Full dataset: 8448 samples
#   bash batch_generate.sh 256 ./output   # Custom size and output dir
#
# For cluster submission, each dataset can be run as a separate job.

SIZE=${1:-256}
OUTPUT_DIR=${2:-./data}
SEED=42

echo "============================================="
echo "Generating 18 datasets, ${SIZE} samples each"
echo "Output directory: ${OUTPUT_DIR}"
echo "============================================="

# ---------------------------------------------------------------------------
# Poisson datasets (9)
# ---------------------------------------------------------------------------
for SHAPE in circle square boomerang; do
    for ID in bc1 bc4 bc5; do
        echo ""
        echo ">>> poisson-${SHAPE}-${ID} (${SIZE} samples)"
        python generate_dataset.py \
            --problem poisson \
            --shape ${SHAPE} \
            --id ${ID} \
            --size ${SIZE} \
            --seed ${SEED} \
            --output-dir ${OUTPUT_DIR} \
            --save-hdf5
    done
done

# ---------------------------------------------------------------------------
# Elasticity datasets (9)
# ---------------------------------------------------------------------------
for SHAPE in circlehollow squarehollow boomcircletri; do
    for ID in m1 m2 m3; do
        echo ""
        echo ">>> elasticity-${SHAPE}-${ID} (${SIZE} samples)"
        python generate_dataset.py \
            --problem elasticity \
            --shape ${SHAPE} \
            --id ${ID} \
            --size ${SIZE} \
            --seed ${SEED} \
            --output-dir ${OUTPUT_DIR} \
            --save-hdf5
    done
done

echo ""
echo "============================================="
echo "Done. Output in ${OUTPUT_DIR}/"
echo "============================================="
ls -lh ${OUTPUT_DIR}/*.nc 2>/dev/null
