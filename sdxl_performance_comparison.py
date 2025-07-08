#!/usr/bin/env python3
"""
SDXL Performance Comparison Script
Compares different optimization methods for SDXL image generation.
"""

import json
import time
import requests
import websocket
import threading
import uuid
from typing import Dict, List, Any, Optional
import statistics
import os

class ComfyUIAPI:
    def __init__(self, server_address: str = "localhost:8188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        
    def queue_prompt(self, prompt: dict) -> str:
        """Queue a prompt and return the prompt ID."""
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = requests.post(f"http://{self.server_address}/prompt", data=data, 
                          headers={'Content-Type': 'application/json'})
        return req.json()["prompt_id"]
    
    def get_history(self, prompt_id: str) -> dict:
        """Get execution history for a prompt."""
        with requests.get(f"http://{self.server_address}/history/{prompt_id}") as response:
            return response.json()
    
    def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> dict:
        """Wait for prompt completion and return execution info."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            history = self.get_history(prompt_id)
            if prompt_id in history:
                return history[prompt_id]
            time.sleep(1)
        raise TimeoutError(f"Prompt {prompt_id} timed out after {timeout} seconds")
    
    def measure_prompt_execution(self, prompt: dict, runs: int = 3, warmup: bool = True) -> dict:
        """Execute a prompt multiple times and measure performance."""
        times = []
        errors = []
        
        # Warmup run to load models
        if warmup:
            print("  🔥 Warmup run (loading models)...")
            try:
                prompt_id = self.queue_prompt(prompt)
                self.wait_for_completion(prompt_id)
                print("  ✅ Warmup completed")
            except Exception as e:
                print(f"  ⚠️  Warmup failed: {str(e)}")
        
        for run in range(runs):
            print(f"  Run {run + 1}/{runs}...")
            try:
                # Use different seed for each run to avoid caching but keep quality comparable
                test_prompt = prompt.copy()
                if "5" in test_prompt:  # KSampler node
                    test_prompt["5"]["inputs"]["seed"] = 123456 + run
                
                start_time = time.time()
                prompt_id = self.queue_prompt(test_prompt)
                
                # Wait for completion
                result = self.wait_for_completion(prompt_id)
                end_time = time.time()
                
                execution_time = end_time - start_time
                times.append(execution_time)
                
                # Check for errors in execution
                if result.get("status", {}).get("status_str") == "error":
                    errors.append(result.get("status", {}).get("messages", []))
                    
            except Exception as e:
                errors.append(str(e))
                
        if not times:
            return {"error": "All runs failed", "errors": errors}
            
        return {
            "times": times,
            "average_time": statistics.mean(times),
            "min_time": min(times),
            "max_time": max(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
            "errors": errors,
            "success_rate": len(times) / runs
        }

def load_workflow(file_path: str) -> dict:
    """Load a workflow JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    """Main performance comparison function."""
    api = ComfyUIAPI()
    
    # Define workflows to test
    workflows = {
        "Baseline SDXL": "workflow_baseline_sdxl.json",
        "DonutNodes TeaCache": "workflow_teacache_sdxl.json", 
        "Wavespeed FBCache": "workflow_fbcache_sdxl.json",
        "StableFast": "workflow_stablefast_sdxl.json",  # Now working after installing polygraphy
        # "Wavespeed Velocator": "workflow_wavespeed_sdxl.json",  # Disabled - velocator not available
    }
    
    # Check if we can test stable-fast (may not be available)
    # workflows["Stable-Fast"] = "workflow_stable_fast_sdxl.json"
    
    results = {}
    
    print("🚀 Starting SDXL Performance Comparison")
    print("=" * 50)
    
    for name, workflow_file in workflows.items():
        print(f"\n📊 Testing {name}...")
        
        if not os.path.exists(workflow_file):
            print(f"  ❌ Workflow file {workflow_file} not found, skipping...")
            continue
            
        try:
            workflow = load_workflow(workflow_file)
            result = api.measure_prompt_execution(workflow, runs=3)
            results[name] = result
            
            if "error" in result:
                print(f"  ❌ {name} failed: {result['error']}")
                if result.get("errors"):
                    for error in result["errors"]:
                        if isinstance(error, list) and len(error) > 0:
                            # Look for specific error types
                            for err_item in error:
                                if isinstance(err_item, list) and len(err_item) > 1:
                                    if err_item[0] == "execution_error":
                                        err_data = err_item[1]
                                        print(f"    Node Error: {err_data.get('node_type', 'Unknown')} - {err_data.get('exception_message', 'Unknown error')}")
                                        break
                        else:
                            print(f"    Error: {error}")
            else:
                print(f"  ✅ {name} completed successfully")
                print(f"    Average time: {result['average_time']:.2f}s")
                print(f"    Min time: {result['min_time']:.2f}s")
                print(f"    Max time: {result['max_time']:.2f}s")
                print(f"    Success rate: {result['success_rate']:.1%}")
                
        except Exception as e:
            print(f"  ❌ {name} failed with exception: {str(e)}")
            results[name] = {"error": str(e)}
    
    print("\n" + "=" * 50)
    print("📈 PERFORMANCE SUMMARY")
    print("=" * 50)
    
    # Sort results by average time
    successful_results = {k: v for k, v in results.items() if "average_time" in v}
    sorted_results = sorted(successful_results.items(), key=lambda x: x[1]["average_time"])
    
    if sorted_results:
        baseline_time = None
        for name, result in sorted_results:
            if name == "Baseline SDXL":
                baseline_time = result["average_time"]
                break
        
        for i, (name, result) in enumerate(sorted_results):
            speedup = ""
            if baseline_time and name != "Baseline SDXL":
                speedup_factor = baseline_time / result["average_time"]
                speedup = f" (🚀 {speedup_factor:.2f}x speedup)"
            
            print(f"{i+1}. {name}: {result['average_time']:.2f}s ± {result['std_dev']:.2f}s{speedup}")
    
    # Show failed tests
    failed_results = {k: v for k, v in results.items() if "error" in v}
    if failed_results:
        print("\n❌ Failed Tests:")
        for name, result in failed_results.items():
            print(f"  - {name}: {result['error']}")
    
    # Save detailed results
    results_file = "sdxl_performance_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📋 Detailed results saved to {results_file}")

if __name__ == "__main__":
    main()