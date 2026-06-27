import os
import json
import numpy as np

# =====================================================================
# --- ENVIRONMENT SWITCH ---
# Set to True to test on your PC. Set to False when deploying to the Versal.
# =====================================================================
ON_PC = True 

if not ON_PC:
    import xir
    import vart

# ==========================================
# CONFIGURATION
# ==========================================
BIN_DIR = r"D:\versal_ready_bins"  # Use your local path on PC, change to "./" on board
MANIFEST_PATH = os.path.join(BIN_DIR, "versal_manifest.json")
MODULATIONS = ('16apsk', '8psk', 'qpsk')
MODEL_XMODEL = "./best_resnet18_dpu.xmodel"

# ==========================================
# ISOLATED INFERENCE ENGINE
# ==========================================
class DPUExecutionEngine:
    def __init__(self):
        if ON_PC:
            print("[System] Running in PC Simulation Mode (Pure NumPy).")
            # Seed the random generator so your test results are stable
            np.random.seed(42)
            self.input_dim = [1, 4, 1, 512]
            self.output_dim = [1, 3]
        else:
            print("[System] Initializing Native Versal DPU Hardware Context...")
            self.graph = xir.Graph.deserialize(MODEL_XMODEL)
            subgraphs = self.graph.get_root_subgraph().get_children()
            dpu_subgraph = [s for s in subgraphs if s.has_attr("device") and s.get_attr("device") == "DPU"][0]
            self.runner = vart.Runner.create_runner(dpu_subgraph, "run")
            
            self.input_dim = self.runner.get_input_tensors()[0].dims
            self.output_dim = self.runner.get_output_tensors()[0].dims

        # Pre-allocate reusable memory buffers 
        self.input_buffer = np.empty(self.input_dim, dtype=np.float32)
        self.output_buffer = np.empty(self.output_dim, dtype=np.float32)

    def run_inference(self, raw_binary_data, true_label):
        """Processes a single binary frame using zero external framework dependencies."""
        # Force data into the exact hardware input shape layout
        self.input_buffer[0] = raw_binary_data.reshape(self.input_dim[1:])

        if ON_PC:
            # SIMULATION: Mimics your 96% accurate model outputs using pure NumPy math
            mock_logits = np.zeros(self.output_dim[1], dtype=np.float32)
            if np.random.rand() < 0.96:
                mock_logits[true_label] = 5.0  # Correct classification
            else:
                wrong_class = (true_label + np.random.choice([1, 2])) % 3
                mock_logits[wrong_class] = 5.0  # Missed classification
            return mock_logits
        else:
            # HARDWARE: Direct asynchronous execution on the Versal DPU core
            job_id = self.runner.execute_async([self.input_buffer], [self.output_buffer])
            self.runner.wait(job_id)
            return self.output_buffer[0]

    def shutdown(self):
        if not ON_PC:
            del self.runner

# ==========================================
# MAIN VALIDATION LOOP
# ==========================================
def main():
    print("--- Starting Clean Framework Verification Loop ---")

    if not os.path.exists(MANIFEST_PATH):
        print(f"[!] Error: Preprocessed dataset manifest missing at {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
        
    filenames = list(manifest.keys())
    total_samples = len(filenames)
    print(f"Tracking {total_samples} target binary verification matrices.")

    # Initialize engine
    engine = DPUExecutionEngine()

    correct_predictions = 0
    num_classes = len(MODULATIONS)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

    print("Executing binary stream validation...")
    for filename in filenames:
        filepath = os.path.join(BIN_DIR, filename)
        true_label = manifest[filename]["label"]

        # Read the raw byte sequence straight from the drive
        raw_data = np.fromfile(filepath, dtype=np.float32)

        # Run calculation
        output_data = engine.run_inference(raw_data, true_label)

        # Extract classification index using pure NumPy
        predicted_label = np.argmax(output_data)

        # Accumulate internal tracking arrays
        if predicted_label == true_label:
            correct_predictions += 1
        confusion_matrix[true_label, predicted_label] += 1

    engine.shutdown()

    # Calculate final pure accuracy metric
    accuracy = (correct_predictions / total_samples) * 100

    print("\n" + "="*50)
    print(f" VERIFICATION ACCURACY: {accuracy:.2f}%")
    print(f" SAMPLES PROCESSED:     {correct_predictions} / {total_samples}")
    print("="*50 + "\n")

    # Clean Console-Based Confusion Matrix (Zero Plotting Libs Required)
    print("Distribution Array Matrix (Rows = True Class, Columns = Predicted Class):")
    print(f"        {MODULATIONS[0]:>8} {MODULATIONS[1]:>8} {MODULATIONS[2]:>8}")
    for i, mod in enumerate(MODULATIONS):
        print(f"{mod:>7}: {confusion_matrix[i, 0]:8d} {confusion_matrix[i, 1]:8d} {confusion_matrix[i, 2]:8d}")

if __name__ == "__main__":
    main()