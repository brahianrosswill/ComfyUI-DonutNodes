#!/usr/bin/env python3
"""
Test different configurations to find the sweet spot for 1.57x with good quality
"""

import requests
import json
import time
import random

COMFYUI_URL = "http://localhost:8188"
PROMPT_URL = f"{COMFYUI_URL}/prompt"
QUEUE_URL = f"{COMFYUI_URL}/queue"
HISTORY_URL = f"{COMFYUI_URL}/history"

def load_workflow(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None

def submit_and_measure(workflow_data, timeout=90):
    try:
        start_time = time.time()
        response = requests.post(PROMPT_URL, json=workflow_data, timeout=10)
        if response.status_code != 200:
            return None, f"Submit failed: {response.status_code}"
        
        prompt_id = response.json().get('prompt_id')
        if not prompt_id:
            return None, "No prompt_id"
        
        while time.time() - start_time < timeout:
            try:
                queue_resp = requests.get(QUEUE_URL, timeout=5)
                if queue_resp.status_code == 200:
                    queue_data = queue_resp.json()
                    running = len(queue_data.get('queue_running', []))
                    pending = len(queue_data.get('queue_pending', []))
                    
                    if running == 0 and pending == 0:
                        history_resp = requests.get(f"{HISTORY_URL}/{prompt_id}", timeout=5)
                        if history_resp.status_code == 200:
                            history_data = history_resp.json()
                            if prompt_id in history_data:
                                result_info = history_data[prompt_id]
                                status = result_info.get('status', {}).get('status_str')
                                if status == 'success':
                                    return time.time() - start_time, "success"
                                else:
                                    return None, f"Execution failed: {status}"
                        time.sleep(1)
                        continue
                        
                time.sleep(2)
            except Exception as e:
                time.sleep(2)
        
        return None, f"Timeout after {timeout}s"
    except Exception as e:
        return None, f"Exception: {e}"

def test_config(cache_mode, cache_threshold, runs=3):
    """Test a configuration with multiple runs"""
    workflow = load_workflow('test_workflows/sdxl_teacache_tuned_aggressive.json')
    if not workflow:
        return [], "Could not load workflow"
    
    workflow['prompt']['2']['inputs']['cache_mode'] = cache_mode
    workflow['prompt']['2']['inputs']['cache_threshold'] = cache_threshold
    
    times = []
    for i in range(runs):
        # Use different seeds to test consistency
        seed = random.randint(1000, 9999)
        workflow['prompt']['6']['inputs']['seed'] = seed
        
        execution_time, status = submit_and_measure(workflow)
        if execution_time is not None:
            times.append(execution_time)
            print(f"    Run {i+1}: {execution_time:.2f}s (seed {seed})")
        else:
            print(f"    Run {i+1}: FAILED - {status}")
        time.sleep(1)
    
    return times, "success" if times else "all_failed"

def main():
    print("🍎 Quality vs Speed Testing for 1.57x Target")
    print("=" * 50)
    
    # Get baseline
    baseline_workflow = load_workflow('test_workflows/sdxl_baseline_workflow.json')
    baseline_workflow['prompt']['5']['inputs']['seed'] = random.randint(1000, 9999)
    baseline_time, status = submit_and_measure(baseline_workflow)
    
    if baseline_time is None:
        print(f"❌ Baseline failed: {status}")
        return False
    
    print(f"📊 Baseline: {baseline_time:.2f}s")
    target_time = baseline_time / 1.57
    print(f"🎯 Target for 1.57x: {target_time:.2f}s")
    print()
    
    # Test configurations that might hit 1.57x with good quality
    configs = [
        # Start with settings that are more aggressive than current 1.40x but less than ultra
        ("aggressive", 0.025),  # Slightly more aggressive than 0.03
        ("aggressive", 0.02),   # More aggressive 
        ("aggressive", 0.015),  # Even more aggressive
        ("ultra_aggressive", 0.08),  # Less aggressive ultra mode
        ("ultra_aggressive", 0.06),  # Medium ultra mode
    ]
    
    results = []
    
    for cache_mode, cache_threshold in configs:
        print(f"🧪 Testing: {cache_mode} mode, threshold={cache_threshold}")
        
        times, status = test_config(cache_mode, cache_threshold, runs=3)
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            speedup = baseline_time / avg_time
            best_speedup = baseline_time / min_time
            
            print(f"  📈 Average: {avg_time:.2f}s ({speedup:.2f}x)")
            print(f"  🚀 Best: {min_time:.2f}s ({best_speedup:.2f}x)")
            
            results.append({
                'config': (cache_mode, cache_threshold),
                'avg_time': avg_time,
                'min_time': min_time,
                'speedup': speedup,
                'best_speedup': best_speedup,
                'times': times
            })
            
            # Check if we're hitting target
            if speedup >= 1.55:
                print(f"  🎉 EXCELLENT! {speedup:.2f}x ≈ 1.57x target!")
            elif speedup >= 1.45:
                print(f"  ✅ GOOD! {speedup:.2f}x getting close to target")
            elif speedup >= 1.35:
                print(f"  📈 PROGRESS! {speedup:.2f}x moving toward target")
            
        else:
            print(f"  ❌ All runs failed")
        
        print()
    
    # Find best result
    if results:
        best = max(results, key=lambda x: x['speedup'])
        
        print("=" * 50)
        print("🏆 BEST CONFIGURATION")
        print("=" * 50)
        cache_mode, cache_threshold = best['config']
        print(f"Configuration: {cache_mode} mode, threshold={cache_threshold}")
        print(f"Average speedup: {best['speedup']:.2f}x")
        print(f"Best run speedup: {best['best_speedup']:.2f}x")
        print(f"Times: {[f'{t:.2f}s' for t in best['times']]}")
        
        if best['speedup'] >= 1.57:
            print("✅ TARGET ACHIEVED!")
        elif best['speedup'] >= 1.5:
            print("🎉 Very close to target!")
        
        print()
        print("💡 Next steps:")
        if best['speedup'] < 1.57:
            print("- Check image quality of recent outputs")
            print("- If quality is good, try more aggressive settings")
            print("- If quality is poor, the current setting may be optimal")
        
        return best['speedup'] >= 1.5
    
    return False

if __name__ == "__main__":
    success = main()