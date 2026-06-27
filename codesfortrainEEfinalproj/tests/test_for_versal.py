import os                                                                                                                
import json                                                                                                                              
import argparse                                                                                                          
import time                                                                                                                              
import numpy as np                                                                                                  
import VART                                                                                
                                                                                                                         
def main():                                                                                                                              
    parser = argparse.ArgumentParser(description="Versal Edge DPU Benchmarking Test With Latency and Confidence Metrics")
    parser.add_argument("--snapshot", default="/run/media/mmcblk0p1/snapshot.VE2802_NPU_IP_O00_A304_M3.modulation_model_resnet18_int8")
    parser.add_argument("--net_name", default="snapshot.VE2802_NPU_IP_O00_A304_M3.modulation_model_resnet18_int8")
    parser.add_argument("--bin_dir", default="/run/media/mmcblk0p1/versal_ready_equal_bins")
    parser.add_argument("--manifest", default="/run/media/mmcblk0p1/versal_ready_equal_bins/versal_manifest.json")
    args = parser.parse_args()                  
                                         
    # 1. Load the Pre-processed Dataset Manifest                            
    if not os.path.exists(args.manifest):
        print(f"[ERROR] Pre-processed manifest missing at: {args.manifest}")                          
        return                        
                                                                                                       
    with open(args.manifest, "r") as f:                
        manifest = json.load(f)                                                
                                                       
    filenames = list(manifest.keys())                                                                  
    total_samples = len(filenames)    
    print(f"[INFO] Successfully loaded manifest. Found {total_samples} pre-processed target binaries.")
                                                       
    # 2. Initialize VART Runner                                                
    print("[INFO] Initializing VART Hardware Runner...")                  
    model = VART.Runner(snapshot_dir=args.snapshot, network_name=args.net_name)
    target_dtype = model.input_types[0]                                    
    print(f"[INFO] Target DPU Hardware Input Type: {target_dtype}")
                                           
    # Meta definitions for zero-dependency scoring
    modulations = ("16apsk", "8psk", "qpsk")                              
    num_classes = len(modulations)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
                                                                 
    # Performance Metric Registers              
    correct = 0                                                  
    processed_count = 0                          
    total_latency_ms = 0.0                                                
    total_confidence = 0.0

# 3. Direct Hardware Inference Loop                                                                                  
    print("[INFO] Launching pure binary evaluation execution...")                                                                        
    for idx, filename in enumerate(filenames, 1):                                                                    
        filepath = os.path.join(args.bin_dir, filename)                                                
                                                                                                                 
        # Safely attempt to read file; skip gracefully if it doesn't exist                
        try:                                                      
            raw_data = np.fromfile(filepath, dtype=np.float32)                
        except (FileNotFoundError, IsADirectoryError):                    
            continue                                                                                  
        except Exception as e:                                    
            print(f"  -> [WARNING] Skipping {filename} due to unexpected read error: {e}")            
            continue                                              
                                                                                                 
        # Increment evaluated counter only after successful read          
        processed_count += 1                                                                          
        true_label = manifest[filename]["label"]                  
                                                                                                       
        # Reshape to match format [Batch=1, Channels=4, Width=512]
        input_data = raw_data.reshape(1, 4, 512)                                                  
                                                                           
        # Dynamic Scaling Check for INT8 vs Float targets                                
        if "int8" in str(target_dtype).lower():                            
            input_data = np.round(input_data * 127).astype(np.int8)                              
        else:                                                  
            input_data = input_data.astype(target_dtype)                                          
                                                                                       
        # Execute on hardware and measure latency                                                
        start_time = time.perf_counter()                                                                                                                                            
        logits = model([input_data])[0]                                                          
        end_time = time.perf_counter()                                                                                                                                              
                                                                                                 
        # Accumulate inference duration metrics                                        
        total_latency_ms += (end_time - start_time) * 1000.0                                    
                                                                                                                                                                                   
        # Calculate Software Softmax to establish clear prediction confidence percentages        
        logits_1d = logits[0]                                                                                                                                                      
        exp_logits = np.exp(logits_1d - np.max(logits_1d))  # Subtract max for numerical stability
        probabilities = exp_logits / np.sum(exp_logits)                                
                                                                                                 
        pred_label = int(np.argmax(probabilities))                                                                                                                                  
        confidence = float(probabilities[pred_label])                                            
        total_confidence += confidence

        # Track accuracy mappings                                
        if pred_label == true_label:                                                              
            correct += 1                                                  
        confusion_matrix[true_label, pred_label] += 1                                                  
                                                                 
        # Periodic status updates to the terminal console                                              
        if idx % 10 == 0 or idx == total_samples:                
            running_acc = (correct / processed_count) * 100 if processed_count > 0 else 0.0      
            avg_lat = total_latency_ms / processed_count if processed_count > 0 else 0.0
            avg_conf = (total_confidence / processed_count) * 100 if processed_count > 0 else 0.0
            print(f"  -> Progressed: {idx:04d}/{total_samples:04d} | Evaluated: {processed_count:04d} | Acc: {running_acc:.2f}% | Latency: {avg_lat:.2f}ms | Conf: {avg_conf:.2f}%")
                                                                                                 
    # 4. Render Final Performance Matrix                        
    final_accuracy = (correct / processed_count) * 100 if processed_count > 0 else 0.0            
    final_latency = total_latency_ms / processed_count if processed_count > 0 else 0.0  
    final_confidence = (total_confidence / processed_count) * 100 if processed_count > 0 else 0.0
                                                                                                                                                                                   
    print("\n" + "="*65)                                                                          
    print(f" FINAL HARDWARE METRICS REPORT")                                                                                                                                        
    print("="*65)                                                                                
    print(f" ACCURACY:          {final_accuracy:.2f}% ({correct} / {processed_count})")
    print(f" AVERAGE LATENCY:   {final_latency:.2f} ms")                                        
    print(f" MEAN CONFIDENCE:   {final_confidence:.2f}%")                                                                                                                          
    print("="*65 + "\n")                                                                          
                                                                                                                                                                                   
    print("Distribution Array Matrix (Rows = True Class, Columns = Predicted Class):")            
    print(f"        {modulations[0]:>8} {modulations[1]:>8} {modulations[2]:>8}")      
    for i, mod in enumerate(modulations):                                                        
        print(f"{mod:>7}: {confusion_matrix[i, 0]:8d} {confusion_matrix[i, 1]:8d} {confusion_matrix[i, 2]:8d}")                                                                    
    print("")                                                                                    
                                                                                                                                                                                   
if __name__ == "__main__":                                                                
    main()                          