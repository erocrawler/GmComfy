#!/usr/bin/env python3
"""
Auto-download script for ComfyUI models, LoRAs, VAEs, and text encoders.
This script downloads all necessary files for the WAN image-to-video workflow.
"""

import os
import sys
import requests
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import tempfile
import hashlib


# Model configurations organized by model set
MODELS = {
    "wan": {
        # Text Encoder / CLIP
        "clip": {
            "umt5_xxl_fp8_e4m3fn_scaled.safetensors": {
                "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "path": "models/clip",
                "size_gb": 6.7,
                "sha256": "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68",
            },
            "umt5_xxl_fp16.safetensors": {
                "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors",
                "path": "models/clip",
                "size_gb": 11.4,
            }
        },

        
        
        # VAE
        "vae": {
            "wan_2.1_vae.safetensors": {
                "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
                "path": "models/vae",
                "size_gb": 0.3,
                "sha256": "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b",
            }
        },
        
        # Diffusion Models
        "diffusion_models": {
            "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors": {
                "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
                "path": "models/diffusion_models",
                "size_gb": 7.3,
            },
            "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors": {
                "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
                "path": "models/diffusion_models",
                "size_gb": 7.3,
            }
        },
        
        # UNet models (GGUF quantized)
        "unet": {
            "wan2.2_i2v_high_noise_14B_Q8_0.gguf": {
                "url": "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_high_noise_14B_Q8_0.gguf",
                "path": "models/unet",
                "size_gb": 14.5,
                "sha256": "619a66032c28e1b27882dfccc0bf93e51edb1491e8d4e4c6f291726abe4de8aa",
            },
            "wan2.2_i2v_low_noise_14B_Q8_0.gguf": {
                "url": "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_low_noise_14B_Q8_0.gguf",
                "path": "models/unet",
                "size_gb": 14.5,
                "sha256": "029c7adc74de4f7804905c5e4fb9335d0862cd2fc37191df526aeac13b64425e",
            },
            # CivitAI GGUF models
            "DasiwaWAN22I2V14BTastysinV8_q8High.gguf": {
                "url": "https://civitai.com/api/download/models/2466604",
                "path": "models/unet",
                "size_gb": 14.5,
                "sha256": "5ff00d11264ae130503cf1954f89cedbf00a1db628f3524120bc966943152172",
            },
            "DasiwaWAN22I2V14BTastysinV8_q8Low.gguf": {
                "url": "https://civitai.com/api/download/models/2466822",
                "path": "models/unet",
                "size_gb": 14.5,
                "sha256": "dd743594590e61f619bba3e80f7a6c44dba20dfaa8e123047de2b21495b8ecdd",
            }
        },
        
        # Upscale Models
        "upscale_models": {
            "RealESRGAN_x2plus.pth": {
                "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                "path": "models/upscale_models",
                "size_gb": 0.064,
                "sha256": "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
            }
        },
        
        # LoRAs
        "loras": {
            "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors": {
                "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
                "path": "models/loras",
                "size_gb": 0.5,
                "sha256": "d176c808d6fc461999b68e321efcb7501b20b8c3797523ed0df14f7d1deff11e",
            },
            "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors": {
                "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
                "path": "models/loras",
                "size_gb": 0.5,
                "sha256": "024f21de095bc8fad9809ded3e9e49a2e170dcf27075da8145ba7d60d8aab7f9",
            },
            "wan22-video10-arcshot-16-sel-7-high.safetensors": {
                "url": "https://huggingface.co/UnifiedHorusRA/ArcShot-Wan2.2_2.1-I2V-A14B/resolve/main/wan22-video10-arcshot-16-sel-7-high.safetensors",
                "path": "models/loras",
                "size_gb": 0.3,
                "sha256": "c53ea534d03549edcf6dfed426f41224568f9cd095095ef0a70e3d95ec26e616",
            },
            "DR34ML4Y_I2V_14B_HIGH_V2.safetensors": {
                "url": "https://civitai.com/api/download/models/2553151",
                "path": "models/loras",
                "size_gb": 0.3,
                "sha256": "d9931756c202bd8d4946c0d163c1269231a6352b51bb4235f6a19894c9ad8c68",
            },
            "DR34ML4Y_I2V_14B_LOW_V2.safetensors": {
                "url": "https://civitai.com/api/download/models/2553271",
                "path": "models/loras",
                "size_gb": 0.3,
                "sha256": "066ee4bfafb685c85f08174c8283cd11bc6d36f4845347f20d633ab44581601f",
            },
            "NSFW-22-H-e8.safetensors": {
                "url": "https://huggingface.co/wiikoo/WAN-LORA/resolve/main/wan2.2/NSFW-22-H-e8.safetensors",
                "path": "models/loras",
                "size_gb": 0.1,
                "sha256": "34e2144d3cd65360f97d09ccbe03e1c39a096df6c9234af5fe3899d1b63cda39",
            },
            "NSFW-22-L-e8.safetensors": {
                "url": "https://huggingface.co/wiikoo/WAN-LORA/resolve/main/wan2.2/NSFW-22-L-e8.safetensors",
                "path": "models/loras",
                "size_gb": 0.1,
                "sha256": "d6b783742f4d5fd63a0223ae1d5bf64fc995a6b408480ac2a00528ae0d4146db",
            },
            "cumshot_wan22_high.safetensors": {
                "url": "https://huggingface.co/seraphimzz/wan22/resolve/main/cumshot_wan22_high.safetensors",
                "path": "models/loras",
                "size_gb": 0.3,
                "sha256": "0b08425be78cbd8bea9041d67318b8a1a9f3de43674af669dc2ba6e3f3897a1b",
            },
            "cumshot_wan22_low.safetensors": {
                "url": "https://huggingface.co/seraphimzz/wan22/resolve/main/cumshot_wan22_low.safetensors",
                "path": "models/loras",
                "size_gb": 0.3,
                "sha256": "6d0fbe82a3757ed30760e7f973d15b3250659c7dd20823dff830d55a4e8755cf",
            },
            "creampie_wan22_e50_high.safetensors": {
                "url": "https://wan.sg-sin-1.linodeobjects.com/creampie_wan22_e50_high.safetensors",
                "path": "models/loras",
                "size_gb": 0.3,
                "sha256": "d411a64bad89e124ad49b68cd2d8590e864d355604acc3f101e088da35de7add",
            },
            "creampie_wan22_e50_low.safetensors": {
                "url": "https://wan.sg-sin-1.linodeobjects.com/creampie_wan22_e50_low.safetensors",
                "path": "models/loras",
                "size_gb": 0.3,
                "sha256": "a1e3add35093017ee4365bedd6279588051185d58f184d0b3aac98ff16981db7",
            },
            "Penis_HN_v1.0.safetensors": {
                "url": "https://civitai.com/api/download/models/2308249",
                "path": "models/loras",
                "size_gb": 0.3,
                "sha256": "b70e1a908ba02a2908fb8a5d3b3be6ddc1f249b0907df224f020f985448481c5",
            },
            "Penis_LN_v1.0.safetensors": {
                "url": "https://civitai.com/api/download/models/2308253",
                "path": "models/loras",
                "size_gb": 0.3,
                "sha256": "7e9ec2a51a34867574a10570b13fc46ee6bfb17db03fbc9182f3d3fa56e9a062",
            },
            "2D_animation_effects.safetensors": {
                "url": "https://civitai.com/api/download/models/2174159",
                "path": "models/loras",
                "size_gb": 0.3,
            },
            "WAN-2.2-I2V_st0mach_bulge_HIGH.safetensors": {
                "url": "https://civitai.com/api/download/models/2424257",
                "path": "models/loras",
                "size_gb": 0.3,
                "sha256": "9b44dd837247363f4e82968ef40b2406f57fae291a9d4b8b1544442cfe8937e2",
            },
            "WAN-2.2-I2V_st0mach_bulge_LOW.safetensors": {
                "url": "https://civitai.com/api/download/models/2424273",
                "path": "models/loras",
                "size_gb": 0.3,
                "sha256": "51e89e2cdbcec9c60540e0f1b9b07d2ffd139edd4d1c4b059d29b29f85eaed9d",
            },
            "wan22-video6-crashzoom-16-sel-6-000150.safetensors": {
                "url": "https://civitai.com/api/download/models/2146673",
                "path": "models/loras",
                "size_gb": 0.3,
                "sha256": "8899d52c7aedf8f96395a2a4281da396c0df8fda70fab7347b7adff96b894c74",
            }
        }
    },
    
    "qwen": {
        # Diffusion Models
        "diffusion_models": {
            "Qwen-Rapid-AIO-SFW-v22.safetensors": {
                "url": "https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/resolve/main/v22/Qwen-Rapid-AIO-SFW-v22.safetensors",
                "path": "models/diffusion_models",
                "size_gb": 27,
                "sha256": "8d419320329eef7dce757cef6eca89e670766bfbbe8c346dd78b3def1e109b0d",
            },
            "Qwen-Rapid-AIO-NSFW-v22.safetensors": {
                "url": "https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/resolve/main/v22/Qwen-Rapid-AIO-NSFW-v22.safetensors",
                "path": "models/diffusion_models",
                "size_gb": 27,
                "sha256": "4e8d7689ceaca6e60305c07c3cfb697a70b312da2d3827598c7e932930da3362",
            }
        },
        
        # Text Encoder / CLIP (GGUF)
        "clip": {
            "Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf": {
                "url": "https://huggingface.co/Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-abliterated/Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf",
                "path": "models/clip",
                "size_gb": 7.8,
                "sha256": "669a5604c47c90c20110c6db5fe10ad7e8ec99b553a785d7d20492f7d5b3e7d0",
            },
            "Qwen2.5-VL-7B-Instruct-abliterated.mmproj-Q8_0.gguf": {
                "url": "https://huggingface.co/Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-abliterated/Qwen2.5-VL-7B-Instruct-abliterated.mmproj-Q8_0.gguf",
                "path": "models/clip",
                "size_gb": 0.6,
                "sha256": "e2ab4e60dd3e174f3d2d6d0c0979c058827699d0085cdcbadada0a5c609ec43f",
            }
        },
        
        # VAE
        "vae": {
            "qwen_image_vae.safetensors": {
                "url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors",
                "path": "models/vae",
                "size_gb": 0.3,
                "sha256": "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
            }
        }
    }
}


def download_segment(url: str, start: int, end: int, output_path: Path) -> Tuple[int, int]:
    """
    Download a segment of a file using Range requests.
    
    Args:
        url: URL to download from (with token already appended if needed)
        start: Start byte position
        end: End byte position
        output_path: Output file path
        
    Returns:
        Tuple of (bytes_downloaded, segment_id)
    """
    try:
        headers = {'Range': f'bytes={start}-{end}'}
        response = requests.get(url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()
        
        bytes_written = 0
        with open(output_path, 'r+b') as f:
            f.seek(start)
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bytes_written += len(chunk)
        
        return (bytes_written, start)
    except Exception as e:
        print(f"✗ Failed to download segment {start}-{end}: {e}")
        raise


def compute_file_sha256(file_path: Path) -> str:
    """
    Compute SHA256 checksum of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        SHA256 hash as hexadecimal string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def download_file(url: str, dest_path: Path, desc: str = None, civitai_token: str = None, num_segments: int = 4, sha256: str = None) -> Tuple[bool, Path]:
    """
    Download a file with progress bar, using multi-segment download for large files.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        desc: Description for progress bar
        civitai_token: CivitAI API token for authenticated downloads
        num_segments: Number of segments for multi-threaded download (only for large files)
        sha256: Expected SHA256 checksum (optional, for verification)
        
    Returns:
        Tuple of (success: bool, path: Path)
    """
    try:
        # Check if file already exists
        if dest_path.exists():
            print(f"✓ {dest_path.name} already exists, skipping")
            # Verify checksum if provided
            if sha256:
                computed = compute_file_sha256(dest_path)
                if computed.lower() != sha256.lower():
                    print(f"⚠️  Checksum mismatch! Deleting and will redownload")
                    dest_path.unlink()
                else:
                    print(f"✓ Checksum verified")
                    return (True, dest_path)
            else:
                return (True, dest_path)
            
        # Create parent directory if it doesn't exist
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add CivitAI token if provided and URL is from CivitAI
        download_url = url
        if civitai_token and 'civitai.com' in url:
            separator = '&' if '?' in url else '?'
            download_url = f"{url}{separator}token={civitai_token}"
        
        # First request to get file size and check for range support
        response = requests.head(download_url, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        supports_range = response.headers.get('accept-ranges', '').lower() == 'bytes'
        
        # Use multi-segment download for large files (>500MB) if server supports ranges
        if total_size > 500 * 1024 * 1024 and supports_range and num_segments > 1:
            print(f"⚡ Using {num_segments}-segment download for large file")
            return _download_file_multipart(download_url, dest_path, total_size, desc, num_segments, sha256)
        else:
            # Standard single-threaded download
            return _download_file_standard(download_url, dest_path, total_size, desc, sha256)
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to download {dest_path.name}: {e}")
        # Clean up partial download
        if dest_path.exists():
            dest_path.unlink()
        return (False, dest_path)
    except KeyboardInterrupt:
        print(f"\n✗ Download interrupted for {dest_path.name}")
        # Clean up partial download
        if dest_path.exists():
            dest_path.unlink()
        raise
    except Exception as e:
        print(f"✗ Unexpected error downloading {dest_path.name}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return (False, dest_path)


def _download_file_standard(url: str, dest_path: Path, total_size: int, desc: str = None, sha256: str = None) -> Tuple[bool, Path]:
    """
    Standard single-threaded file download.
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(dest_path, 'wb') as f:
            with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=desc or dest_path.name,
                ncols=100
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Verify checksum if provided
        if sha256:
            print(f"\n🔍 Verifying checksum...")
            computed = compute_file_sha256(dest_path)
            if computed.lower() != sha256.lower():
                print(f"✗ Checksum mismatch!")
                print(f"  Expected: {sha256}")
                print(f"  Got:      {computed}")
                dest_path.unlink()
                return (False, dest_path)
            print(f"✓ Checksum verified: {computed[:16]}...")
        
        print(f"✓ Downloaded {dest_path.name}")
        return (True, dest_path)
    except Exception as e:
        print(f"✗ Standard download failed: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return (False, dest_path)


def _download_file_multipart(url: str, dest_path: Path, total_size: int, desc: str = None, num_segments: int = 4, sha256: str = None) -> Tuple[bool, Path]:
    """
    Multi-segment file download using Range requests.
    """
    try:
        # Create empty file with correct size
        with open(dest_path, 'wb') as f:
            f.seek(total_size - 1)
            f.write(b'\0')
        
        # Calculate segment size
        segment_size = total_size // num_segments
        segments = []
        
        for i in range(num_segments):
            start = i * segment_size
            if i == num_segments - 1:
                end = total_size - 1
            else:
                end = start + segment_size - 1
            segments.append((start, end))
        
        # Download segments in parallel
        downloaded = 0
        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc=desc or dest_path.name, ncols=100) as pbar:
            with ThreadPoolExecutor(max_workers=num_segments) as executor:
                futures = [
                    executor.submit(download_segment, url, start, end, dest_path)
                    for start, end in segments
                ]
                
                for future in as_completed(futures):
                    bytes_downloaded, _ = future.result()
                    pbar.update(bytes_downloaded)
                    downloaded += bytes_downloaded
        
        # Verify checksum if provided
        if sha256:
            print(f"\n🔍 Verifying checksum...")
            computed = compute_file_sha256(dest_path)
            if computed.lower() != sha256.lower():
                print(f"✗ Checksum mismatch!")
                print(f"  Expected: {sha256}")
                print(f"  Got:      {computed}")
                dest_path.unlink()
                return (False, dest_path)
            print(f"✓ Checksum verified: {computed[:16]}...")
        
        print(f"✓ Downloaded {dest_path.name}")
        return (True, dest_path)
    except Exception as e:
        print(f"✗ Multi-segment download failed: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return (False, dest_path)


def calculate_total_size(model_set: str = "wan", categories: List[str] = None) -> float:
    """Calculate total download size in GB."""
    total_gb = 0.0
    
    if model_set not in MODELS:
        return total_gb
    
    for category, files in MODELS[model_set].items():
        if categories and category not in categories:
            continue
        for file_name, file_info in files.items():
            if not (Path(file_info["path"]) / file_name).exists():
                total_gb += file_info["size_gb"]
    
    return total_gb


def download_models(
    base_path: Path,
    model_set: str = "wan",
    categories: List[str] = None,
    dry_run: bool = False,
    civitai_token: str = None,
    max_workers: int = 4,
    segments_per_file: int = 4
) -> Dict[str, int]:
    """
    Download all configured models using multi-threading.
    
    Args:
        base_path: Base path for ComfyUI installation
        model_set: Model set to download ("wan" or "qwen")
        categories: List of categories to download (None = all)
        dry_run: If True, only show what would be downloaded
        civitai_token: CivitAI API token for authenticated downloads
        max_workers: Maximum number of concurrent file downloads
        segments_per_file: Number of segments for large file multi-segment downloads
        
    Returns:
        Dictionary with success/failure counts
    """
    stats = {"success": 0, "failed": 0, "skipped": 0}
    stats_lock = Lock()
    
    if model_set not in MODELS:
        print(f"❌ Invalid model set: {model_set}")
        print(f"   Available model sets: {', '.join(MODELS.keys())}")
        return stats
    
    # Calculate and display total size
    total_size = calculate_total_size(model_set, categories)
    if total_size > 0:
        print(f"\n📦 Total download size: ~{total_size:.1f} GB")
        if dry_run:
            print("🔍 Dry run mode - no files will be downloaded\n")
        else:
            response = input("Continue? [y/N]: ")
            if response.lower() not in ['y', 'yes']:
                print("Cancelled")
                return stats
    else:
        print("\n✓ All files already downloaded!")
        return stats
    
    print()
    
    # Collect all download tasks
    download_tasks = []
    
    for category, files in MODELS[model_set].items():
        if categories and category not in categories:
            continue
            
        print(f"\n{'='*60}")
        print(f"📂 {category.upper()}")
        print(f"{'='*60}")
        
        for file_name, file_info in files.items():
            dest_path = base_path / file_info["path"] / file_name
            
            if dest_path.exists():
                print(f"✓ {file_name} already exists")
                with stats_lock:
                    stats["skipped"] += 1
                continue
            
            if dry_run:
                print(f"Would download: {file_name} ({file_info['size_gb']:.1f} GB)")
                print(f"  → {dest_path}")
                continue
            
            print(f"⬇️  Queued: {file_name} (~{file_info['size_gb']:.1f} GB)")
            download_tasks.append({
                "url": file_info["url"],
                "dest_path": dest_path,
                "file_name": file_name,
                "civitai_token": civitai_token,
                "segments": segments_per_file,
                "sha256": file_info.get("sha256")
            })
    
    if dry_run or not download_tasks:
        return stats
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting multi-threaded downloads ({max_workers} workers)")
    print(f"{'='*60}\n")
    
    # Execute downloads in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        futures = {
            executor.submit(
                download_file,
                task["url"],
                task["dest_path"],
                task["file_name"],
                task["civitai_token"],
                task["segments"],
                task["sha256"]
            ): task for task in download_tasks
        }
        
        # Process completed downloads
        for future in as_completed(futures):
            try:
                success, path = future.result()
                with stats_lock:
                    if success:
                        stats["success"] += 1
                    else:
                        stats["failed"] += 1
            except KeyboardInterrupt:
                print("\n✗ Download interrupted")
                raise
            except Exception as e:
                print(f"✗ Error in download thread: {e}")
                with stats_lock:
                    stats["failed"] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download models, LoRAs, VAEs, and encoders for ComfyUI WAN workflow"
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path.cwd() / "src",
        help="Base path for ComfyUI installation (default: ./src)"
    )
    parser.add_argument(
        "--model-set",
        type=str,
        default="wan",
        choices=["wan", "qwen"],
        help="Model set to download: wan (WAN v2.2 image-to-video) or qwen (Qwen image edit) (default: wan)"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        choices=["clip", "vae", "diffusion_models", "unet", "upscale_models", "loras"],
        help="Specific categories to download (default: all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all models and exit"
    )
    parser.add_argument(
        "--civitai-token",
        type=str,
        help="CivitAI API token for downloading models from CivitAI (required for CivitAI models)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of concurrent file downloads (default: 4)"
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=4,
        help="Number of segments for large file multi-threaded downloads (default: 4)"
    )
    parser.add_argument(
        "--compute-checksums",
        action="store_true",
        help="Compute and display SHA256 checksums for all files in the model-set"
    )
    
    args = parser.parse_args()
    
    # Compute checksums mode
    if args.compute_checksums:
        if args.model_set not in MODELS:
            print(f"❌ Invalid model set: {args.model_set}")
            sys.exit(1)
        
        print(f"\n🔍 Computing checksums for {args.model_set.upper()} model-set\n")
        print(f"{'='*80}")
        
        for category, files in MODELS[args.model_set].items():
            if args.categories and category not in args.categories:
                continue
            
            print(f"\n{category.upper()}:")
            print("-" * 80)
            
            for file_name, file_info in files.items():
                file_path = args.base_path / file_info["path"] / file_name
                
                if not file_path.exists():
                    print(f"⚠️  {file_name}")
                    print(f"    Path: {file_path}")
                    print(f"    Status: FILE NOT FOUND\n")
                    continue
                
                print(f"📄 {file_name}")
                sha256_hash = compute_file_sha256(file_path)
                print(f"    SHA256: {sha256_hash}")
                print(f"    \"sha256\": \"{sha256_hash}\",")
                print()
        
        print(f"{'='*80}\n")
        return
    
    # List mode
    if args.list:
        print("\n📋 Available Models:\n")
        for model_set, categories_dict in MODELS.items():
            print(f"\n{'='*60}")
            print(f"Model Set: {model_set.upper()}")
            print(f"{'='*60}")
            for category, files in categories_dict.items():
                print(f"\n{category.upper()}:")
                for file_name, file_info in files.items():
                    print(f"  • {file_name} ({file_info['size_gb']:.1f} GB)")
                    print(f"    Path: {file_info['path']}")
                    print(f"    URL: {file_info['url']}")
        return
    
    # Verify base path exists
    if not args.base_path.exists():
        print(f"❌ Base path does not exist: {args.base_path}")
        print(f"   Please ensure ComfyUI is installed or specify correct --base-path")
        sys.exit(1)
    
    print("="*60)
    print("🎨 ComfyUI Model Downloader")
    print("="*60)
    print(f"\n📁 Base path: {args.base_path.absolute()}")
    print(f"🎯 Model set: {args.model_set}")
    
    if args.categories:
        print(f"📦 Categories: {', '.join(args.categories)}")
    else:
        print(f"📦 Categories: all")
    
    print(f"⚙️  Max workers: {args.max_workers}")
    print(f"⚙️  Segments per file: {args.segments}")
    
    try:
        stats = download_models(
            args.base_path,
            model_set=args.model_set,
            categories=args.categories,
            dry_run=args.dry_run,
            civitai_token=args.civitai_token,
            max_workers=args.max_workers,
            segments_per_file=args.segments
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print("📊 Summary")
        print(f"{'='*60}")
        
        if args.dry_run:
            print("Dry run completed")
        else:
            print(f"✓ Successfully downloaded: {stats['success']}")
            print(f"⊘ Skipped (already exist): {stats['skipped']}")
            if stats['failed'] > 0:
                print(f"✗ Failed: {stats['failed']}")
                sys.exit(1)
            else:
                print(f"\n🎉 All downloads completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
