#!/usr/bin/env python3
"""Monitor all 3 pods and terminate each when complete"""
import subprocess
import time
import os
import runpod

runpod.api_key = os.environ.get('RUNPOD_API_KEY')

pods = {
    "xazayczh7qhs6j": {"name": "Pod1-CNN", "ip": "213.173.105.9", "port": 46570, "log": "part1.log", "done": False},
    "5wbob5mn7vy3hu": {"name": "Pod2-ViT", "ip": "213.173.105.6", "port": 30001, "log": "part2.log", "done": False},
    "t75wuo2w6vfi8g": {"name": "Pod3-SOTA", "ip": "213.173.105.10", "port": 14602, "log": "sota.log", "done": False},
}

def check_complete(ip, port, log):
    """Check if test completed by looking for 'saved to' in log"""
    cmd = f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i ~/.ssh/runpod_ed25519 -p {port} root@{ip} 'grep -c \"saved to\" /workspace/{log} 2>/dev/null || echo 0'"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20)
        return int(result.stdout.strip()) > 0
    except:
        return False

def save_results(ip, port, log, local_name):
    """Download results before terminating"""
    cmd = f"scp -o StrictHostKeyChecking=no -i ~/.ssh/runpod_ed25519 -P {port} root@{ip}:/workspace/{log} /home/runner/workspace/lucasnet/{local_name}"
    subprocess.run(cmd, shell=True, timeout=30)
    # Also get JSON if exists
    json_log = log.replace('.log', '_results.json')
    cmd2 = f"scp -o StrictHostKeyChecking=no -i ~/.ssh/runpod_ed25519 -P {port} root@{ip}:/workspace/*results*.json /home/runner/workspace/lucasnet/ 2>/dev/null"
    subprocess.run(cmd2, shell=True, timeout=30)

print("=== Monitoring 3 pods for completion ===")
while not all(p["done"] for p in pods.values()):
    for pod_id, info in pods.items():
        if info["done"]:
            continue
        if check_complete(info["ip"], info["port"], info["log"]):
            print(f"\n{info['name']} COMPLETE! Saving results and terminating...")
            save_results(info["ip"], info["port"], info["log"], f"{info['name'].lower()}.log")
            runpod.terminate_pod(pod_id)
            print(f"  {info['name']} terminated.")
            info["done"] = True
    
    # Status update
    done_count = sum(1 for p in pods.values() if p["done"])
    print(f"[{time.strftime('%H:%M:%S')}] {done_count}/3 pods complete", end="\r")
    time.sleep(60)

print("\n\n=== ALL PODS COMPLETE AND TERMINATED ===")
