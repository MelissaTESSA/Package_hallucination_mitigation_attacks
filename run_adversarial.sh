#!/bin/bash
#PYTHON=.venv/bin/python

STRATEGIES=(baseline greedy self_refine dola rag nudging contrastive_decoding)
LANGUAGES=(Python JavaScript Rust Ruby)

run_model_on_gpu() {
    local short_name=$1
    local model_id=$2
    local gpu=$3
    local dola_layer=$4
    local dola_exit_layers=$5

    echo "=== Starting $short_name on GPU $gpu ==="
    for language in "${LANGUAGES[@]}"; do
        for strategy in "${STRATEGIES[@]}"; do
            echo "--- $short_name | $language | $strategy ---"
            CUDA_VISIBLE_DEVICES=$gpu python run_mitigation_infer_json.py \
                --config config.yml \
                --model "$model_id" \
                --short-name "$short_name" \
                --dola-layer "$dola_layer" \
                --dola-early-exit-layers "$dola_exit_layers" \
                --strategy "$strategy" \
                --language "$language" \
                --input data/adversarial/adversarial_${language}.json \
                --output output/adversarial/${short_name}/${language}/${strategy}.json
        done
    done
    echo "=== Done: $short_name ==="
}

# GPU 0: sequential (one model at a time to avoid OOM)
(
    run_model_on_gpu "llama3_8b"  "meta-llama/Meta-Llama-3-8B-Instruct" 0  32  "4,8,16,24"
    run_model_on_gpu "qwen3_5_9b" "Qwen/Qwen3.5-9B"                     0  32  "4,8,16,24"
) &

# GPU 1: sequential (one model at a time to avoid OOM)
(
    run_model_on_gpu "mistral_7b"          "mistralai/Mistral-7B-Instruct-v0.3"       1  32  "4,8,16,24"
    run_model_on_gpu "deepseek_coder_6_7b" "deepseek-ai/deepseek-coder-6.7b-instruct" 1  32  "4,8,16,24"
) &

wait
echo "=== All models done ==="
