#!/bin/bash
#PYTHON=.venv/bin/python
#
# Runs all 8 mitigation strategies, across all 4 languages, on all 9 models,
# dispatched on 2 GPUs. Resumable: any output file that's already fully
# populated is skipped, so a killed/interrupted run can just be restarted.

STRATEGIES=(baseline greedy self_refine dola rag nudging contrastive_decoding actlcd)
LANGUAGES=(Python JavaScript Rust Ruby)

LOG_DIR=output/logs
mkdir -p "$LOG_DIR"

run_model_on_gpu() {
    local short_name=$1
    local model_id=$2
    local gpu=$3
    local dola_layer=$4
    local dola_exit_layers=$5
    # ActLCD is the same layer-contrastive family as DoLa, so it reuses the
    # same mature_layer / early_exit_layers per model (not placeholders).
    local actlcd_layer=$dola_layer
    local actlcd_exit_layers=$dola_exit_layers

    echo "=== Starting $short_name on GPU $gpu ==="
    for language in "${LANGUAGES[@]}"; do
        for strategy in "${STRATEGIES[@]}"; do
            local outfile="output/adversarial/${short_name}/${language}/${strategy}.json"
            if [ -f "$outfile" ]; then
                local done
                done=$(python3 -c "
import json
d=json.load(open('$outfile'))
total=len(d)
done=sum(1 for x in d if 'answer' in x)
print(f'{done}/{total}')
if done==total: exit(0)
else: exit(1)
")
                if [ $? -eq 0 ]; then
                    echo "--- SKIP $short_name | $language | $strategy ($done done) ---"
                    continue
                else
                    echo "--- RESUME $short_name | $language | $strategy ($done) ---"
                fi
            fi
            echo "--- $short_name | $language | $strategy ---"
            if [ "$strategy" = "actlcd" ]; then
                CUDA_VISIBLE_DEVICES=$gpu python run_mitigation_infer_json.py \
                    --config config.yml \
                    --model "$model_id" \
                    --short-name "$short_name" \
                    --actlcd-layer "$actlcd_layer" \
                    --actlcd-early-exit-layers "$actlcd_exit_layers" \
                    --strategy "$strategy" \
                    --language "$language" \
                    --batch-size 1 \
                    --input data/adversarial/adversarial_${language}.json \
                    --output "$outfile"
            else
                CUDA_VISIBLE_DEVICES=$gpu python run_mitigation_infer_json.py \
                    --config config.yml \
                    --model "$model_id" \
                    --short-name "$short_name" \
                    --dola-layer "$dola_layer" \
                    --dola-early-exit-layers "$dola_exit_layers" \
                    --strategy "$strategy" \
                    --language "$language" \
                    --input data/adversarial/adversarial_${language}.json \
                    --output "$outfile"
            fi
        done
    done
    echo "=== Done: $short_name ==="
}

# GPU 0 (~21B total params, sequential): qwen3.5-9b, mistral-7b, gemma-3-4b, deepseek-1.3b
(
    run_model_on_gpu "09_qwen3_5_9b"    "Qwen/Qwen3.5-9B"                           0  32  "4,8,16,24"
    run_model_on_gpu "08_mistral_7b"    "mistralai/Mistral-7B-Instruct-v0.3"        0  32  "4,8,16,24"
    run_model_on_gpu "02_gemma_3_4b"    "google/gemma-3-4b-it"                      0  34  "4,8,16,24,28"
    run_model_on_gpu "05_deepseek_1_3b" "deepseek-ai/deepseek-coder-1.3b-instruct"  0  24  "4,8,12,16,20"
) &

# GPU 1 (~20B total params, sequential): llama3-8b, deepseek-coder-6.7b, qwen2.5-3b, qwen2.5-1.5b, gemma-3-1b
(
    run_model_on_gpu "07_llama3_8b"           "meta-llama/Meta-Llama-3-8B-Instruct"       1  32  "4,8,16,24"
    run_model_on_gpu "06_deepseek_coder_6_7b" "deepseek-ai/deepseek-coder-6.7b-instruct"  1  32  "4,8,16,24"
    run_model_on_gpu "04_qwen_2_5_3b"         "Qwen/Qwen2.5-3B-Instruct"                  1  36  "4,8,16,24,32"
    run_model_on_gpu "03_qwen_2_5_1_5b"       "Qwen/Qwen2.5-1.5B-Instruct"                1  28  "4,8,12,16,24"
    run_model_on_gpu "01_gemma_3_1b"          "google/gemma-3-1b-it"                      1  26  "4,8,12,16,20"
) &

wait
echo "=== All models done ==="
