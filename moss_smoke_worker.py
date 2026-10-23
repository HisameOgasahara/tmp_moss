"""Run one MOSS candidate inside the Colab ComfyUI environment."""

import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return importlib.import_module(name + ".nodes") if Path(path).name == "__init__.py" else module


def run_candidate(config):
    started = time.monotonic()
    result = {
        "candidate": config["id"], "mode": config.get("mode", "smoke"),
        "status": "RUNNING", "stage": "import", "timings_s": {},
        "config": config, "stage_started": started,
    }
    torch = None
    write_json(config["state_path"], result)

    def stage(name, function):
        result.update(stage=name, stage_started=time.monotonic())
        write_json(config["state_path"], result)
        begin = time.monotonic()
        try:
            value = function()
            if torch is not None and torch.cuda.is_initialized():
                torch.cuda.synchronize()
            return value
        finally:
            result["timings_s"][name] = round(time.monotonic() - begin, 3)

    try:
        root = Path(config["comfy_root"])
        os.chdir(root)
        sys.path.insert(0, str(root))
        sys.argv = ["moss-smoke", *config.get("flags", [])]
        import comfy.options
        comfy.options.enable_args_parsing()
        from comfy.cli_args import args, enables_dynamic_vram
        import cuda_malloc  # Configure the allocator before importing torch.
        import comfy_aimdo.control as aimdo
        if enables_dynamic_vram():
            headroom = None if args.reserve_vram is None else int(args.reserve_vram * 2**30)
            aimdo.init(simple_vram_headroom=headroom, nvml_pressure=not args.disable_nvml_pressure)
        import torch as torch_module
        torch = torch_module
        import nodes
        import comfy.memory_management as dynamic_memory
        import comfy.model_management as memory
        import comfy.model_patcher as patchers
        import comfy.sampler_helpers as helpers
        import comfy.samplers as samplers
        import comfy.utils

        if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
            raise RuntimeError("이 노트북의 실행 대상은 NVIDIA T4입니다.")
        # Match main.py's native NVIDIA DynamicVRAM initialization.
        if enables_dynamic_vram() and memory.torch_version_numeric >= (2, 8):
            devices = [(device.index, int(args.vram_headroom * 2**30))
                       for device in memory.get_all_torch_devices()]
            if aimdo.init_devices(iter(devices)):
                patchers.CoreModelPatcher = patchers.ModelPatcherDynamic
                dynamic_memory.aimdo_enabled = True
        result["dynamic_vram"] = dynamic_memory.aimdo_enabled
        torch.cuda.reset_peak_memory_stats()
        result["gpu"] = torch.cuda.get_device_name(0)
        result["torch"] = torch.__version__
        result["cuda"] = torch.version.cuda

        model = stage("load_dit", lambda: nodes.UNETLoader().load_unet(
            config["dit"], config.get("weight_dtype", "default"))[0])
        clip = stage("load_te", lambda: nodes.CLIPLoader().load_clip(
            config["te"], "moss_image", config.get("text_device", "default"))[0])
        vae = stage("load_vae", lambda: nodes.VAELoader().load_vae(config["vae"])[0])

        def check_tensor(tensor, expected=None):
            if expected is not None and tuple(tensor.shape) != tuple(expected):
                raise ValueError(f"shape={tuple(tensor.shape)}, expected={tuple(expected)}")
            if not bool(torch.isfinite(tensor).all().item()):
                raise FloatingPointError("NaN/Inf가 발생했습니다.")

        def encode(text, label):
            output = clip.encode_from_tokens(clip.tokenize(text), return_dict=True)
            tensor = output["cond"]
            check_tensor(tensor)
            if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[-1] != 4096 or tensor.shape[1] < 1:
                raise ValueError(f"잘못된 conditioning shape: {tuple(tensor.shape)}")
            result[label + "_tokens"] = tensor.shape[1]
            output["cond"] = tensor.cpu()
            return output

        with torch.inference_mode():
            positive = stage("encode_positive", lambda: encode(config["prompt"], "positive"))
            negative = stage("encode_negative", lambda: encode("", "negative"))
            options_enabled = set(config.get("options", [config.get("option", "none")]))
            spectrum = None

            if "lora8" in options_enabled:
                def patch_lora():
                    patched = nodes.LoraLoaderModelOnly().load_lora_model_only(
                        model, config["lora"], config.get("lora_strength", 1.0))[0]
                    count = len(patched.patches)
                    if count == 0:
                        raise ValueError("적용된 LoRA patch가 없습니다.")
                    result["lora_patch_keys"] = count
                    return patched
                model = stage("patch_lora", patch_lora)
            if "compile" in options_enabled:
                from comfy_extras.nodes_torch_compile import TorchCompileModel
                model = stage("patch_compile", lambda: TorchCompileModel.execute(model, "inductor")[0])
            if "spectrum" in options_enabled:
                spectrum = stage("import_spectrum", lambda: load_module(
                    "moss_spectrum_probe", config["spectrum_path"]))
                settings = spectrum.SpectrumConfig(**config["spectrum_config"])
                model = stage("patch_spectrum", lambda: spectrum.apply_spectrum_moss(model, settings))

            device = memory.get_torch_device()
            shape = (1, 128, config["height"] // 16, config["width"] // 16)
            latent = torch.randn(shape, generator=torch.Generator("cpu").manual_seed(config["seed"]))
            options = patchers.create_model_options_clone(model.model_options)
            helpers.prepare_model_patcher(model, {}, options)

            def prepare_model():
                required = model.model.memory_required(shape)
                memory.load_models_gpu([model], memory_required=required)
                samplers.cast_to_load_options(options, device=device, dtype=model.model_dtype())
                model.pre_run()
                result["dit_compute_dtype"] = str(model.model.get_dtype_inference())
                result["patcher"] = type(model).__name__
            stage("prepare_dit", prepare_model)

            try:
                noise = latent.to(device)
                sigmas = torch.linspace(1.0, 0.05, config.get("spectrum_calls", 12), device=device)
                # The same schedule object identifies a single Spectrum probe.
                schedule = torch.cat((sigmas, sigmas.new_zeros(1)))

                def forward(conditioning, branch, sigma=0.5):
                    transformer_options = dict(options.get("transformer_options", {}))
                    transformer_options["cond_or_uncond"] = [branch]
                    transformer_options["sample_sigmas"] = schedule
                    output = model.model.apply_model(
                        noise, torch.tensor([sigma], device=device),
                        c_crossattn=conditioning["cond"], transformer_options=transformer_options)
                    check_tensor(output, shape)

                if spectrum is None:
                    stage("forward_positive", lambda: forward(positive, 0))
                    stage("forward_negative", lambda: forward(negative, 1))
                    if "compile" in options_enabled:
                        stage("forward_repeat", lambda: forward(positive, 0))
                else:
                    def probe_spectrum():
                        for sigma in sigmas.tolist():
                            forward(positive, 0, sigma)
                            forward(negative, 1, sigma)
                        wrappers = model.model_options["transformer_options"]["wrappers"]
                        roots = [wrapper.root_state for group in wrappers.values()
                                 for values in group.values() for wrapper in values
                                 if hasattr(wrapper, "root_state")]
                        actual = sum(branch.actual_count for state in roots for branch in state.branches.values())
                        forecast = sum(branch.forecast_count for state in roots for branch in state.branches.values())
                        result.update(spectrum_actual=actual, spectrum_forecast=forecast)
                        if actual == 0 or forecast == 0:
                            raise ValueError(f"Spectrum 경로 미통과: actual={actual}, forecast={forecast}")
                    stage("forward_spectrum", probe_spectrum)
            finally:
                model.cleanup()
                samplers.cast_to_load_options(options, device=model.offload_device)

            # Record ComfyUI's built-in automatic tiling, if selected at runtime.
            result["vae_actual_path"] = "tiled" if "tiled_vae" in options_enabled else "default"
            if "tiled_vae" not in options_enabled:
                for method_name in ("decode_tiled_", "decode_tiled_1d", "decode_tiled_3d", "_decode_tiled_owned"):
                    original = getattr(vae, method_name)
                    def tracked_tiling(*args, _original=original, **kwargs):
                        result["vae_actual_path"] = "auto_tiled"
                        return _original(*args, **kwargs)
                    setattr(vae, method_name, tracked_tiling)

            def decode(samples):
                if "tiled_vae" in options_enabled:
                    pixels = nodes.VAEDecodeTiled().decode(vae, {"samples": samples}, 512, 64)[0]
                else:
                    pixels = vae.decode(samples)
                check_tensor(pixels, (1, config["height"], config["width"], 3))
                return pixels

            stage("decode_vae", lambda: decode(latent * 0.1))
            if config.get("mode", "smoke") == "generate":
                def generate():
                    from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple
                    sampling_model = model
                    if "lora8" not in options_enabled:
                        sampling_model = SkipLayerGuidanceDiTSimple.execute(
                            model, 0.0, 1.0, double_layers="", single_layers="22")[0]
                    pos = [[positive["cond"], {k: v for k, v in positive.items() if k != "cond"}]]
                    neg = [[negative["cond"], {k: v for k, v in negative.items() if k != "cond"}]]
                    count = config["steps"]
                    cfg = config["cfg"]
                    samples = nodes.common_ksampler(
                        sampling_model, config["seed"], count, cfg, "euler", "sgm_uniform",
                        pos, neg, {"samples": torch.zeros_like(latent)}, denoise=1.0)[0]["samples"]
                    pixels = decode(samples)
                    from PIL import Image
                    import numpy as np
                    image_path = Path(config["image_path"])
                    image = Image.fromarray((pixels[0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8))
                    image.save(image_path)
                    result["image_path"] = str(image_path)
                stage("generate", generate)
            result["status"] = "PASS_GENERATION" if config.get("mode") == "generate" else "PASS_SMOKE"
            result["stage"] = "done"
    except Exception as exc:
        detail = str(exc)
        if isinstance(exc, FloatingPointError):
            status = "NONFINITE"
        elif torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError):
            status = "CUDA_OOM"
        elif isinstance(exc, MemoryError) or "Cannot allocate memory" in detail or "DefaultCPUAllocator" in detail:
            status = "HOST_OOM"
        else:
            status = "FAIL"
        result.update(status=status, error=f"{type(exc).__name__}: {detail}")
        traceback.print_exc()
    finally:
        result["total_s"] = round(time.monotonic() - started, 3)
        if torch is not None and torch.cuda.is_initialized():
            result["torch_peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            result["torch_peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
        write_json(config["state_path"], result)
        write_json(config["result_path"], result)
    return 0 if result["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    configuration = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    raise SystemExit(run_candidate(configuration))
