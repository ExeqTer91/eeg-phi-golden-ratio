#!/usr/bin/env python3
"""
RunPod Pod Creator - Continuously tries to create pods for EEG processing
"""

import os
import time
import json
import requests

RUNPOD_API_KEY = os.environ.get('RUNPOD_API_KEY')
GRAPHQL_ENDPOINT = 'https://api.runpod.io/graphql'

GPU_TYPES = [
    "NVIDIA RTX A4000",
    "NVIDIA RTX A5000",
    "NVIDIA RTX 4090",
    "NVIDIA RTX 3090",
    "NVIDIA RTX 3080",
    "NVIDIA RTX A6000",
    "NVIDIA L4",
    "NVIDIA A40",
    "NVIDIA RTX 4080",
    "NVIDIA RTX 3070",
    "NVIDIA RTX 4070 Ti",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 3090",
]

POD_SPECS = {
    "image": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    "volumeInGb": 50,
    "containerDiskInGb": 30,
    "startSsh": True,
    "cloudType": "ALL"
}

PODS_TO_CREATE = [
    {"name": "eeg-physionet"},
    {"name": "eeg-lemon"}
]

def graphql_request(query, variables=None):
    """Execute GraphQL request to RunPod API"""
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {RUNPOD_API_KEY}'
    }
    payload = {'query': query}
    if variables:
        payload['variables'] = variables
    
    try:
        response = requests.post(GRAPHQL_ENDPOINT, json=payload, headers=headers, timeout=60)
        return response.json()
    except Exception as e:
        return {'errors': [{'message': str(e)}]}

def get_available_gpus():
    """Query available GPU types"""
    query = """
    query GpuTypes {
        gpuTypes {
            id
            displayName
            memoryInGb
        }
    }
    """
    result = graphql_request(query)
    if 'data' in result and 'gpuTypes' in result['data']:
        return result['data']['gpuTypes']
    return []

def create_pod(name, gpu_type_id):
    """Create a pod with specified GPU type"""
    mutation = """
    mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $input) {
            id
            name
            desiredStatus
            imageName
            machine {
                podHostId
            }
            runtime {
                ports {
                    ip
                    isIpPublic
                    privatePort
                    publicPort
                    type
                }
            }
        }
    }
    """
    
    variables = {
        "input": {
            "name": name,
            "gpuTypeId": gpu_type_id,
            "imageName": POD_SPECS["image"],
            "volumeInGb": POD_SPECS["volumeInGb"],
            "containerDiskInGb": POD_SPECS["containerDiskInGb"],
            "startSsh": POD_SPECS["startSsh"],
            "cloudType": POD_SPECS["cloudType"],
            "gpuCount": 1,
            "volumeMountPath": "/workspace"
        }
    }
    
    result = graphql_request(mutation, variables)
    return result

def get_pod_details(pod_id):
    """Get SSH connection details for a pod"""
    query = """
    query Pod($podId: String!) {
        pod(input: {podId: $podId}) {
            id
            name
            desiredStatus
            runtime {
                ports {
                    ip
                    isIpPublic
                    privatePort
                    publicPort
                    type
                }
            }
        }
    }
    """
    result = graphql_request(query, {"podId": pod_id})
    return result

def main():
    print("="*60)
    print("RunPod Pod Creator for EEG Processing")
    print("="*60)
    
    if not RUNPOD_API_KEY:
        print("ERROR: RUNPOD_API_KEY not found!")
        return
    
    print(f"\nAPI Key: {RUNPOD_API_KEY[:8]}...{RUNPOD_API_KEY[-4:]}")
    
    print("\n[1] Querying available GPU types...")
    available_gpus = get_available_gpus()
    if available_gpus:
        print(f"Found {len(available_gpus)} GPU types available")
        gpu_ids = [g['id'] for g in available_gpus]
        print(f"GPU IDs: {gpu_ids[:5]}...")
    else:
        print("Using predefined GPU type list")
        gpu_ids = GPU_TYPES
    
    created_pods = []
    pods_remaining = PODS_TO_CREATE.copy()
    max_attempts = 10
    attempt = 0
    
    while pods_remaining and attempt < max_attempts:
        attempt += 1
        print(f"\n{'='*60}")
        print(f"ATTEMPT {attempt}/{max_attempts}")
        print(f"Pods remaining to create: {len(pods_remaining)}")
        print(f"{'='*60}")
        
        for pod_config in pods_remaining[:]:
            pod_name = pod_config["name"]
            print(f"\n[Attempting to create pod: {pod_name}]")
            
            for gpu_id in gpu_ids:
                print(f"  Trying GPU: {gpu_id}...")
                result = create_pod(pod_name, gpu_id)
                
                if 'errors' in result:
                    error_msg = result['errors'][0].get('message', 'Unknown error')
                    if 'no available' in error_msg.lower() or 'out of stock' in error_msg.lower() or 'insufficient' in error_msg.lower():
                        print(f"    GPU not available: {error_msg[:50]}...")
                        continue
                    else:
                        print(f"    Error: {error_msg[:80]}...")
                        continue
                
                if 'data' in result and result['data'].get('podFindAndDeployOnDemand'):
                    pod_data = result['data']['podFindAndDeployOnDemand']
                    pod_id = pod_data['id']
                    print(f"\n  ✓ SUCCESS! Pod created!")
                    print(f"    Pod ID: {pod_id}")
                    print(f"    Pod Name: {pod_data['name']}")
                    print(f"    Status: {pod_data['desiredStatus']}")
                    print(f"    GPU: {gpu_id}")
                    
                    time.sleep(5)
                    details = get_pod_details(pod_id)
                    if 'data' in details and details['data'].get('pod'):
                        pod_info = details['data']['pod']
                        runtime = pod_info.get('runtime', {})
                        ports = runtime.get('ports', []) if runtime else []
                        
                        ssh_info = None
                        for port in ports:
                            if port.get('privatePort') == 22:
                                ssh_info = port
                                break
                        
                        if ssh_info:
                            print(f"    SSH: ssh root@{ssh_info.get('ip')} -p {ssh_info.get('publicPort')}")
                        else:
                            print(f"    SSH: (initializing, check RunPod dashboard)")
                    
                    created_pods.append({
                        'id': pod_id,
                        'name': pod_data['name'],
                        'gpu': gpu_id
                    })
                    pods_remaining.remove(pod_config)
                    break
            else:
                print(f"  Could not create {pod_name} with any GPU type this attempt")
        
        if pods_remaining:
            if attempt < max_attempts:
                wait_time = 120
                print(f"\nWaiting {wait_time} seconds before next attempt...")
                time.sleep(wait_time)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    if created_pods:
        print(f"\n✓ Successfully created {len(created_pods)} pod(s):")
        for pod in created_pods:
            print(f"\n  Pod ID: {pod['id']}")
            print(f"  Name: {pod['name']}")
            print(f"  GPU: {pod['gpu']}")
            
            details = get_pod_details(pod['id'])
            if 'data' in details and details['data'].get('pod'):
                pod_info = details['data']['pod']
                runtime = pod_info.get('runtime', {})
                ports = runtime.get('ports', []) if runtime else []
                
                for port in ports:
                    if port.get('privatePort') == 22:
                        print(f"  SSH: ssh root@{port.get('ip')} -p {port.get('publicPort')}")
                        break
    else:
        print("\n✗ No pods were created after all attempts")
    
    if pods_remaining:
        print(f"\n✗ Failed to create {len(pods_remaining)} pod(s):")
        for pod in pods_remaining:
            print(f"  - {pod['name']}")

if __name__ == "__main__":
    main()
