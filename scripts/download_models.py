#!/usr/bin/env python3
"""
Auto-download script for ComfyUI models, LoRAs, VAEs, and text encoders.
This script downloads all necessary files for the WAN image-to-video workflow.
"""

import os
import sys
import requests
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import argparse


# Model configurations organized by model set
MODELS = {
    "wan": {
        # Text Encoder / CLIP
        "clip": {
            "umt5_xxl_fp8_e4m3fn_scaled.safetensors": {
                "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "path": "models/clip",
                "size_gb": 6.7,
            },
            "": {
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
            },
            "wan2.2_i2v_low_noise_14B_Q8_0.gguf": {
                "url": "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_low_noise_14B_Q8_0.gguf",
                "path": "models/unet",
                "size_gb": 14.5,
            },
            # CivitAI GGUF models
            "DasiwaWAN22I2V14BTastysinV8_q8High.gguf": {
                "url": "https://civitai.com/api/download/models/2466604",
                "path": "models/unet",
                "size_gb": 14.5,
            },
            "DasiwaWAN22I2V14BTastysinV8_q8Low.gguf": {
                "url": "https://civitai.com/api/download/models/2466822",
                "path": "models/unet",
                "size_gb": 14.5,
            }
        },
        
        # Upscale Models
        "upscale_models": {
            "RealESRGAN_x2plus.pth": {
                "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                "path": "models/upscale_models",
                "size_gb": 0.064,
            }
        },
        
        # LoRAs
        "loras": {
            "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors": {
                "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
                "path": "models/loras",
                "size_gb": 0.5,
            },
            "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors": {
                "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
                "path": "models/loras",
                "size_gb": 0.5,
            },
            "wan22-video10-arcshot-16-sel-7-high.safetensors": {
                "url": "https://huggingface.co/UnifiedHorusRA/ArcShot-Wan2.2_2.1-I2V-A14B/resolve/main/wan22-video10-arcshot-16-sel-7-high.safetensors",
                "path": "models/loras",
                "size_gb": 0.3,
            },
            "DR34ML4Y_I2V_14B_HIGH_V2.safetensors": {
                "url": "https://civitai.com/api/download/models/2553151",
                "path": "models/loras",
                "size_gb": 0.3,
            },
            "DR34ML4Y_I2V_14B_LOW_V2.safetensors": {
                "url": "https://civitai.com/api/download/models/2553271",
                "path": "models/loras",
                "size_gb": 0.3,
            },
            "NSFW-22-H-e8.safetensors": {
                "url": "https://huggingface.co/wiikoo/WAN-LORA/resolve/main/wan2.2/NSFW-22-H-e8.safetensors",
                "path": "models/loras",
                "size_gb": 0.1,
            },
            "NSFW-22-L-e8.safetensors": {
                "url": "https://huggingface.co/wiikoo/WAN-LORA/resolve/main/wan2.2/NSFW-22-L-e8.safetensors",
                "path": "models/loras",
                "size_gb": 0.1,
            },
            "cumshot_wan22_high.safetensors": {
                "url": "https://huggingface.co/seraphimzz/wan22/resolve/main/cumshot_wan22_high.safetensors",
                "path": "models/loras",
                "size_gb": 0.3,
            },
            "cumshot_wan22_low.safetensors": {
                "url": "https://huggingface.co/seraphimzz/wan22/resolve/main/cumshot_wan22_low.safetensors",
                "path": "models/loras",
                "size_gb": 0.3,
            },
            "creampie_wan22_e50_high.safetensors": {
                "url": "https://wan.sg-sin-1.linodeobjects.com/creampie_wan22_e50_high.safetensors",
                "path": "models/loras",
                "size_gb": 0.3,
            },
            "creampie_wan22_e50_low.safetensors": {
                "url": "https://wan.sg-sin-1.linodeobjects.com/creampie_wan22_e50_low.safetensors",
                "path": "models/loras",
                "size_gb": 0.3,
            },
            "Penis_HN_v1.0.safetensors": {
                "url": "https://civitai.com/api/download/models/2308249",
                "path": "models/loras",
                "size_gb": 0.3,
            },
            "Penis_LN_v1.0.safetensors": {
                "url": "https://civitai.com/api/download/models/2308253",
                "path": "models/loras",
                "size_gb": 0.3,
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
            },
            "WAN-2.2-I2V_st0mach_bulge_LOW.safetensors": {
                "url": "https://civitai.com/api/download/models/2424273",
                "path": "models/loras",
                "size_gb": 0.3,
            },
            "wan22-video6-crashzoom-16-sel-6-000150.safetensors": {
                "url": "https://civitai.com/api/download/models/2146673",
                "path": "models/loras",
                "size_gb": 0.3,
            }
        }
    },
    
    "qwen": {
        # Diffusion Models
        "diffusion_models": {
            "Qwen-Rapid-AIO-SFW-v22.safetensors": {
                "url": "https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/resolve/main/v22/Qwen-Rapid-AIO-SFW-v22.safetensors",
                "path": "models/diffusion_models",
                "size_gb": 3.5,
            },
            "Qwen-Rapid-AIO-NSFW-v22.safetensors": {
                "url": "https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/resolve/main/v22/Qwen-Rapid-AIO-NSFW-v22.safetensors",
                "path": "models/diffusion_models",
                "size_gb": 3.5,
            }
        },
        
        # Text Encoder / CLIP (GGUF)
        "clip": {
            "Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf": {
                "url": "https://huggingface.co/Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-abliterated/Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf",
                "path": "models/clip",
                "size_gb": 7.8,
            },
            "Qwen2.5-VL-7B-Instruct-abliterated.mmproj-Q8_0.gguf": {
                "url": "https://huggingface.co/Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-abliterated/Qwen2.5-VL-7B-Instruct-abliterated.mmproj-Q8_0.gguf",
                "path": "models/clip",
                "size_gb": 0.6,
            }
        },
        
        # VAE
        "vae": {
            "qwen_image_vae.safetensors": {
                "url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors",
                "path": "models/vae",
                "size_gb": 0.3,
            }
        }
    }
}


def download_file(url: str, dest_path: Path, desc: str = None, civitai_token: str = None) -> bool:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        desc: Description for progress bar
        civitai_token: CivitAI API token for authenticated downloads
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if file already exists
        if dest_path.exists():
            print(f"✓ {dest_path.name} already exists, skipping")
            return True
            
        # Create parent directory if it doesn't exist
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add CivitAI token if provided and URL is from CivitAI
        download_url = url
        if civitai_token and 'civitai.com' in url:
            separator = '&' if '?' in url else '?'
            download_url = f"{url}{separator}token={civitai_token}"
        
        # Start download with streaming
        response = requests.get(download_url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
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
        
        print(f"✓ Downloaded {dest_path.name}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to download {dest_path.name}: {e}")
        # Clean up partial download
        if dest_path.exists():
            dest_path.unlink()
        return False
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
        return False


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
    civitai_token: str = None
) -> Dict[str, int]:
    """
    Download all configured models.
    
    Args:
        base_path: Base path for ComfyUI installation
        model_set: Model set to download ("wan" or "qwen")
        categories: List of categories to download (None = all)
        dry_run: If True, only show what would be downloaded
        civitai_token: CivitAI API token for authenticated downloads
        
    Returns:
        Dictionary with success/failure counts
    """
    stats = {"success": 0, "failed": 0, "skipped": 0}
    
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
                stats["skipped"] += 1
                continue
            
            if dry_run:
                print(f"Would download: {file_name} ({file_info['size_gb']:.1f} GB)")
                print(f"  → {dest_path}")
                continue
            
            print(f"\n⬇️  Downloading {file_name} (~{file_info['size_gb']:.1f} GB)")
            print(f"   URL: {file_info['url']}")
            print(f"   Destination: {dest_path}\n")
            
            if download_file(file_info["url"], dest_path, file_name, civitai_token):
                stats["success"] += 1
            else:
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
    
    args = parser.parse_args()
    
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
    
    try:
        stats = download_models(
            args.base_path,
            model_set=args.model_set,
            categories=args.categories,
            dry_run=args.dry_run,
            civitai_token=args.civitai_token
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
