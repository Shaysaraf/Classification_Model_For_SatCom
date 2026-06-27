import os
import json
import time
import argparse
import numpy as np
import xir
import vart

def run_zero_preprocessing_validation(model_path, data_dir):
    """
    Validates pre-packaged binary satellite signals directly on the Versal DPU.
    Zero CPU preprocessing occurs inside this loop.
    """
    # 1. Verify necessary files exist
    manifest_path = os.path.join(data_dir, "versal_manifest.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Compiled DPU model file (.xmodel) not found at: {model_path}")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Preprocessed data manifest not found at: {manifest_path}")

    # 2. Deserialize the graph and locate the target DPU Subgraph
    graph = xir.Graph.deserialize(model_path)
    root_subgraph = graph.get_root_subgraph()
    child_subgraphs = root_subgraph.get_children()
    
    # Filter out CPU/Shape subgraphs to bind directly to the DPU hardware accelerator
    dpu_subgraphs = [s for s in child_subgraphs if s.has_attr("device") and s.get_attr("device") == "DPU"]
    if not dpu_subgraphs:
        # Fallback for alternative compilation hierarchies
        dpu_subgraph = root_subgraph
    else:
        dpu_subgraph = dpu_subgraphs[0]
        
    # 3. Instantiate the Vitis AI VART Runner
    runner = vart.Runner.create_runner(dpu_subgraph, "run")

    # 4. Extract Input and Output Tensor Dimensionality 
    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()
    
    input_shape = tuple(input_tensors[0].dims)   # Expected: (1, 4, 512)
    output_shape = tuple(output_tensors[0].dims) # Expected: (1, 3)

    # 5. Pre-allocate Strict C-Contiguous RAM Buffers for VART Execution
    # Pre-allocating prevents runtime heap allocations inside the inference loop
    input_buffer = np.empty(input_shape, dtype=np.float32, order='C')
    output_buffer = np.empty(output_shape, dtype=np.float32, order='C')

    # 6. Load Master Label Database
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Setup Metric Containers
    total_samples = 0
    correct_predictions = 0
    latencies_ms = []
    
    class_names = ['16APSK', '8PSK', 'QPSK']
    per_class_total = {0: 0, 1: 0, 2: 0}
    per_class_correct = {0: 0, 1: 0, 2: 0}

    print(f"\n========================================================")
    print(f"  VERSAL DPU ZERO-PREPROCESSING ACCURACY VALIDATION     ")
    print(f"========================================================")
    print(f" Model Path         : {model_path}")
    print(f" Data Directory     : {data_dir}")
    print(f" Target Tensor Shape: {input_shape}")
    print(f" Total Manifest Size: {len(manifest)} samples")
    print(f"--------------------------------------------------------\n")

    # 7. Hardware Inference Loop
    for bin_filename, metadata in manifest.items():
        bin_path = os.path.join(data_dir, bin_filename)
        if not os.path.exists(bin_path):
            continue

        true_label = metadata["label"]

        # --- STEP A: DIRECT MEMORY MAP READ (0% PREPROCESSING) ---
        # Reads the exact float32 sequence directly into the specific input shape
        raw_signal_tensor = np.fromfile(bin_path, dtype=np.float32).reshape(input_shape)
        
        # Blit data directly into the mapped hardware input buffer
        np.copyto(input_buffer, raw_signal_tensor)

        # --- STEP B: ASYNCHRONOUS DPU EXECUTION & HARDWARE WAIT ---
        start_time = time.perf_counter()
        job_id = runner.execute_async([input_buffer], [output_buffer])
        runner.wait(job_id)
        end_time = time.perf_counter()

        # Capture core NPU latency
        latencies_ms.append((end_time - start_time) * 1000.0)

        # --- STEP C: ARGMAX EXTRACT & STATS COLLECTION ---
        predicted_label = np.argmax(output_buffer[0])
        
        total_samples += 1
        per_class_total[true_label] += 1
        if predicted_label == true_label:
            correct_predictions += 1
            per_class_correct[true_label] += 1

    # Clean up runner instance to release dynamic hardware channels safely
    del runner

    # 8. Compute and Output Final Engineering Metrics
    if total_samples == 0:
        print("[!] Execution failed: Zero valid binary files were processed.")
        return

    overall_accuracy = (correct_predictions / total_samples) * 100.0
    avg_latency = sum(latencies_ms) / len(latencies_ms)
    throughput_fps = 1000.0 / avg_latency if avg_latency > 0 else 0

    print("\n" + "="*56)
    print("                FINAL BENCHMARK SUMMARY                 ")
    print("="*56)
    print(f" Total Evaluated Signals: {total_samples}")
    print(f" Final Hardware Accuracy: {overall_accuracy:.2f}%")
    print(f" Average DPU Latency    : {avg_latency:.4f} ms")
    print(f" Sustained Throughput   : {throughput_fps:.2f} FPS")
    print("-"*56)
    print(" Per-Class Accuracy Breakdown:")
    for idx, name in enumerate(class_names):
        total = per_class_total[idx]
        correct = per_class_correct[idx]
        acc = (correct / total * 100.0) if total > 0 else 0.0
        print(f"   {name:<10} -> {acc:.2f}% ({correct}/{total})")
    print("="*56 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Versal ACAP Zero-Preprocessing Validation Script")
    parser.add_argument("--model", type=str, default="resnet18_16bf.xmodel", help="Path to compiled .xmodel file")
    parser.add_argument("--data_dir", type=str, default="versal_ready_bins", help="Path to directory containing processed .bin files and JSON manifest")
    
    args = parser.parse_args()
    run_zero_preprocessing_validation(args.model, args.data_dir)